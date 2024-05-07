#include <Interpreters/Cache/WriteBufferToFileSegment.h>
#include <Interpreters/Cache/FileSegment.h>
#include <Interpreters/Cache/FileCache.h>
#include <Interpreters/Context.h>
#include <IO/SwapHelper.h>
#include <IO/ReadBufferFromFile.h>

#include <base/scope_guard.h>

#include <Common/CurrentThread.h>
#include <Common/formatReadable.h>
#include <Common/logger_useful.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int NOT_ENOUGH_SPACE;
}

namespace
{
    size_t getCacheLockWaitTimeout()
    {
        auto query_context = CurrentThread::getQueryContext();
        if (query_context)
            return query_context->getSettingsRef().temporary_data_in_cache_reserve_space_wait_lock_timeout_milliseconds;
        else
            return Context::getGlobalContextInstance()->getSettingsRef().temporary_data_in_cache_reserve_space_wait_lock_timeout_milliseconds;
    }
}

WriteBufferToFileSegment::WriteBufferToFileSegment(FileSegment * file_segment_)
    : WriteBufferFromFileDecorator(std::make_unique<WriteBufferFromFile>(file_segment_->getPath()))
    , file_segment(file_segment_)
    , reserve_space_lock_wait_timeout_milliseconds(getCacheLockWaitTimeout())
{
}

WriteBufferToFileSegment::WriteBufferToFileSegment(FileSegmentsHolderPtr segment_holder_)
    : WriteBufferFromFileDecorator(
        segment_holder_->size() == 1
        ? std::make_unique<WriteBufferFromFile>(segment_holder_->front().getPath())
        : throw Exception(ErrorCodes::LOGICAL_ERROR, "WriteBufferToFileSegment can be created only from single segment"))
    , file_segment(&segment_holder_->front())
    , segment_holder(std::move(segment_holder_))
    , reserve_space_lock_wait_timeout_milliseconds(getCacheLockWaitTimeout())
{
}

/// If it throws an exception, the file segment will be incomplete, so you should not use it in the future.
void WriteBufferToFileSegment::nextImpl()
{
    auto downloader [[maybe_unused]] = file_segment->getOrSetDownloader();
    if (downloader != FileSegment::getCallerId())
    {
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Failed to set a downloader (current downloader: {}, file segment info: {})",
                        downloader, file_segment->getInfoForLog());
    }

    SCOPE_EXIT({
        if (file_segment->isDownloader())
            file_segment->completePartAndResetDownloader();
        else
            chassert(false);
    });

    size_t bytes_to_write = offset();

    FileCacheReserveStat reserve_stat;
    /// In case of an error, we don't need to finalize the file segment
    /// because it will be deleted soon and completed in the holder's destructor.
    bool ok = file_segment->reserve(bytes_to_write, reserve_space_lock_wait_timeout_milliseconds, &reserve_stat);

    if (!ok)
    {
        String reserve_stat_msg;
        for (const auto & [kind, stat] : reserve_stat.stat_by_kind)
            reserve_stat_msg += fmt::format("{} hold {}, can release {}; ",
                toString(kind), ReadableSize(stat.non_releasable_size), ReadableSize(stat.releasable_size));

        if (std::filesystem::exists(file_segment->getPath()))
            std::filesystem::remove(file_segment->getPath());

        throw Exception(ErrorCodes::NOT_ENOUGH_SPACE, "Failed to reserve {} bytes for {}: {}(segment info: {})",
            bytes_to_write,
            file_segment->getKind() == FileSegmentKind::Temporary ? "temporary file" : "the file in cache",
            reserve_stat_msg,
            file_segment->getInfoForLog()
        );
    }

    try
    {
        SwapHelper swap(*this, *impl);
        /// Write data to the underlying buffer.
        impl->next();
    }
    catch (...)
    {
        LOG_WARNING(getLogger("WriteBufferToFileSegment"), "Failed to write to the underlying buffer ({})", file_segment->getInfoForLog());
        throw;
    }

    file_segment->setDownloadedSize(bytes_to_write);
}

std::unique_ptr<ReadBuffer> WriteBufferToFileSegment::getReadBufferImpl()
{
    /** Finalize here and we don't need to finalize in the destructor,
      * because in case destructor called without `getReadBufferImpl` called, data won't be read.
      */
    finalize();
    return std::make_unique<ReadBufferFromFile>(file_segment->getPath());
}

}
