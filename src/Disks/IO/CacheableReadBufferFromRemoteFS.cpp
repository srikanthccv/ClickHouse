#include "CacheableReadBufferFromRemoteFS.h"
#include <IO/createReadBufferFromFileBase.h>

namespace ProfileEvents
{
    extern const Event RemoteFSReadBytes;
    extern const Event RemoteFSCacheReadBytes;
    extern const Event RemoteFSCacheDownloadBytes;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_SEEK_THROUGH_FILE;
    extern const int LOGICAL_ERROR;
}

CacheableReadBufferFromRemoteFS::CacheableReadBufferFromRemoteFS(
    const String & path_,
    FileCachePtr cache_,
    SeekableReadBufferPtr reader_,
    const ReadSettings & settings_,
    size_t read_until_position_)
    : SeekableReadBuffer(nullptr, 0)
    , log(&Poco::Logger::get("CacheableReadBufferFromRemoteFS" + path_ + ""))
    , key(cache_->hash(path_))
    , cache(cache_)
    , reader(reader_)
    , settings(settings_)
    , read_until_position(read_until_position_)
{
}

void CacheableReadBufferFromRemoteFS::initialize(size_t offset, size_t size)
{
    file_segments_holder.emplace(cache->getOrSet(key, offset, size));

    /**
     * Segments in returned list are ordered in ascending order and represent a full contiguous
     * interval (no holes). Each segment in returned list has state: DOWNLOADED, DOWNLOADING or EMPTY.
     * DOWNLOADING means that either the segment is being downloaded by some other thread or that it
     * is going to be downloaded by the caller (just space reservation happened).
     * EMPTY means that the segment not in cache, not being downloaded and cannot be downloaded
     * by the caller (because of not enough space or max elements limit reached). E.g. returned list is never empty.
     */
    if (file_segments_holder->file_segments.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "List of file segments cannot be empty");

    LOG_TEST(log, "Having {} file segments to read", file_segments_holder->file_segments.size());
    current_file_segment_it = file_segments_holder->file_segments.begin();

    initialized = true;
}

SeekableReadBufferPtr CacheableReadBufferFromRemoteFS::createCacheReadBuffer(size_t offset) const
{
    return createReadBufferFromFileBase(cache->path(key, offset), settings);
}

SeekableReadBufferPtr CacheableReadBufferFromRemoteFS::createReadBuffer(FileSegmentPtr file_segment)
{
    auto range = file_segment->range();

    assert((impl && range.left == file_offset_of_buffer_end) || (!impl && range.left <= file_offset_of_buffer_end));

    SeekableReadBufferPtr implementation_buffer;

    auto download_state = file_segment->state();
    while (true)
    {
        switch (download_state)
        {
            case FileSegment::State::DOWNLOADED:
            {
                read_type = ReadType::CACHE;
                implementation_buffer = createCacheReadBuffer(range.left);

                break;
            }
            case FileSegment::State::DOWNLOADING:
            {
                auto downloader_id = file_segment->getOrSetDownloader();
                if (downloader_id == file_segment->getCallerId())
                {
                    read_type = ReadType::REMOTE_FS_READ_AND_DOWNLOAD;
                    implementation_buffer = reader;

                    break;
                }
                else
                {
                    download_state = file_segment->wait();
                    continue;
                }
            }
            case FileSegment::State::EMPTY:
            {
                auto downloader_id = file_segment->getOrSetDownloader();
                if (downloader_id == file_segment->getCallerId())
                {
                    /// Note: setDownloader() sets file_segment->state = State::DOWNLOADING under segment mutex.
                    /// After setDownloader() succeeds, current thread remains a downloader until
                    /// file_segment->complete() is called by downloader or until downloader's
                    /// FileSegmentsHolder is destructed.

                    download_state = FileSegment::State::DOWNLOADING;
                }
                else
                    download_state = file_segment->wait();

                continue;
            }
            case FileSegment::State::NO_SPACE:
            {
                read_type = ReadType::REMOTE_FS_READ;
                implementation_buffer = reader;

                break;
            }
        }

        break;
    }

    LOG_TEST(log, "Current file segment: {}, read type: {}", range.toString(), toString(read_type));
    download_current_segment = read_type == ReadType::REMOTE_FS_READ_AND_DOWNLOAD;

    /// TODO: Add seek avoiding for s3 on the lowest level.
    implementation_buffer->setReadUntilPosition(range.right + 1); /// [..., range.right]
    implementation_buffer->seek(range.left, SEEK_SET);

    return implementation_buffer;
}

void CacheableReadBufferFromRemoteFS::completeFileSegmentAndGetNext()
{
    auto file_segment_it = current_file_segment_it++;
    auto range = (*file_segment_it)->range();
    assert(file_offset_of_buffer_end > range.right);

    if (download_current_segment)
        (*current_file_segment_it)->complete();

    /// Do not hold pointer to file segment if it is not needed anymore
    /// so can become releasable and can be evicted from cache.
    file_segments_holder->file_segments.erase(file_segment_it);
}

bool CacheableReadBufferFromRemoteFS::nextImpl()
{
    if (!initialized)
        initialize(file_offset_of_buffer_end, getTotalSizeToRead());

    if (current_file_segment_it == file_segments_holder->file_segments.end())
        return false;

    if (impl)
    {
        auto current_read_range = (*current_file_segment_it)->range();
        assert(current_read_range.left <= file_offset_of_buffer_end);

        if (file_offset_of_buffer_end > current_read_range.right)
        {
            completeFileSegmentAndGetNext();

            if (current_file_segment_it == file_segments_holder->file_segments.end())
                return false;

            impl = createReadBuffer(*current_file_segment_it);
        }
    }
    else
    {
        impl = createReadBuffer(*current_file_segment_it);
    }

    auto current_read_range = (*current_file_segment_it)->range();
    size_t remaining_size_to_read = std::min(current_read_range.right, read_until_position - 1) - file_offset_of_buffer_end + 1;

    assert(current_read_range.left <= file_offset_of_buffer_end);
    assert(current_read_range.right >= file_offset_of_buffer_end);

    swap(*impl);

    bool result;
    auto & file_segment = *current_file_segment_it;

    try
    {
        result = impl->next();

        if (result && download_current_segment)
        {
            size_t size = impl->buffer().size();

            if (file_segment->reserve(size))
                file_segment->write(impl->buffer().begin(), impl->buffer().size());
            else
                file_segment->complete();
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);

        if (download_current_segment)
            file_segment->complete();

        /// Note: If exception happens in another place -- out of scope of this buffer, then
        /// downloader's FileSegmentsHolder is responsible to set ERROR state and call notify.

        /// (download_path (if exists) is removed from inside cache)
        throw;
    }

    if (result)
    {
        /// TODO: This resize() is needed only for local fs read, so it is better to
        /// just implement setReadUntilPosition() for local filesysteam read buffer?

        impl->buffer().resize(std::min(impl->buffer().size(), remaining_size_to_read));
        file_offset_of_buffer_end += impl->buffer().size();

        switch (read_type)
        {
            case ReadType::CACHE:
            {
                ProfileEvents::increment(ProfileEvents::RemoteFSCacheReadBytes, working_buffer.size());
                break;
            }
            case ReadType::REMOTE_FS_READ:
            {
                ProfileEvents::increment(ProfileEvents::RemoteFSReadBytes, working_buffer.size());
                break;
            }
            case ReadType::REMOTE_FS_READ_AND_DOWNLOAD:
            {
                ProfileEvents::increment(ProfileEvents::RemoteFSReadBytes, working_buffer.size());
                ProfileEvents::increment(ProfileEvents::RemoteFSCacheDownloadBytes, working_buffer.size());
                break;
            }
        }
    }

    swap(*impl);

    if (file_offset_of_buffer_end > current_read_range.right)
        completeFileSegmentAndGetNext();

    LOG_TEST(log, "Returning with {} bytes, last range: {}, current offset: {}",
             working_buffer.size(), current_read_range.toString(), file_offset_of_buffer_end);
    return result;
}

off_t CacheableReadBufferFromRemoteFS::seek(off_t offset, int whence)
{
    if (initialized)
        throw Exception(ErrorCodes::CANNOT_SEEK_THROUGH_FILE,
                        "Seek is allowed only before first read attempt from the buffer");

    if (whence != SEEK_SET)
        throw Exception(ErrorCodes::CANNOT_SEEK_THROUGH_FILE, "Only SEEK_SET allowed");

    file_offset_of_buffer_end = offset;
    size_t size = getTotalSizeToRead();
    initialize(offset, size);

    return offset;
}

size_t CacheableReadBufferFromRemoteFS::getTotalSizeToRead()
{
    /// Last position should be guaranteed to be set, as at least we always know file size.
    if (!read_until_position)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Last position was not set");

    /// On this level should be guaranteed that read size is non-zero.
    if (file_offset_of_buffer_end >= read_until_position)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Read boundaries mismatch. Expected {} < {}",
                        file_offset_of_buffer_end, read_until_position);

    return read_until_position - file_offset_of_buffer_end;
}

off_t CacheableReadBufferFromRemoteFS::getPosition()
{
    return file_offset_of_buffer_end - available();
}

CacheableReadBufferFromRemoteFS::~CacheableReadBufferFromRemoteFS()
{
    std::optional<FileSegment::Range> range;
    if (download_current_segment
        && current_file_segment_it != file_segments_holder->file_segments.end())
        range = (*current_file_segment_it)->range();
    LOG_TEST(log, "Buffer reset. Current offset: {}, last download range: {}, state: {}",
             file_offset_of_buffer_end, range ? range->toString() : "None", (*current_file_segment_it)->state());
}

}
