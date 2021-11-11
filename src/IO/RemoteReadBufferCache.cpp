#include <functional>
#include <IO/RemoteReadBufferCache.h>
#include <Poco/Logger.h>
#include <Poco/JSON/JSON.h>
#include <Poco/JSON/Parser.h>
#include <Poco/Dynamic/Var.h>
#include <Poco/DigestStream.h>
#include <Poco/MD5Engine.h>
#include <base/logger_useful.h>
#include "Common/Exception.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int BAD_GET;
    extern const int LOGICAL_ERROR;
}

std::shared_ptr<RemoteCacheController>
RemoteCacheController::recover(const std::string & local_path_, std::function<void(RemoteCacheController *)> const & finish_callback)
{
    std::filesystem::path dir_handle(local_path_);
    std::filesystem::path data_file(local_path_ + "/data.bin");
    std::filesystem::path meta_file(local_path_ + "/meta.txt");
    if (!std::filesystem::exists(dir_handle) || !std::filesystem::exists(data_file) || !std::filesystem::exists(meta_file))
    {
        LOG_ERROR(&Poco::Logger::get("RemoteCacheController"), "not exists directory:" + local_path_);
        return nullptr;
    }

    std::ifstream meta_fs(local_path_ + "/meta.txt");
    Poco::JSON::Parser meta_parser;
    auto meta_jobj = meta_parser.parse(meta_fs).extract<Poco::JSON::Object::Ptr>();
    auto remote_path = meta_jobj->get("remote_path").convert<std::string>();
    auto schema = meta_jobj->get("schema").convert<std::string>();
    auto cluster = meta_jobj->get("cluster").convert<std::string>();
    auto downloaded = meta_jobj->get("downloaded").convert<std::string>();
    auto mod_ts = meta_jobj->get("last_mod_ts").convert<UInt64>();
    if (downloaded == "false")
    {
        LOG_ERROR(&Poco::Logger::get("RemoteCacheController"), "not a downloaded file: " + local_path_);
        return nullptr;
    }
    auto size = std::filesystem::file_size(data_file);

    auto cntrl = std::make_shared<RemoteCacheController>(schema, cluster, remote_path, mod_ts, local_path_, nullptr, finish_callback);
    cntrl->download_finished = true;
    cntrl->current_offset = size;
    meta_fs.close();

    finish_callback(cntrl.get());
    return cntrl;
}

RemoteCacheController::RemoteCacheController(
    const std::string & schema_,
    const std::string & cluster_,
    const std::string & path_,
    UInt64 mod_ts_,
    const std::string & local_path_,
    std::shared_ptr<ReadBuffer> readbuffer_,
    std::function<void(RemoteCacheController *)> const & finish_callback)
{
    download_thread = nullptr;
    schema = schema_;
    cluster = cluster_;
    local_path = local_path_;
    remote_path = path_;
    last_mod_ts = mod_ts_;
    valid = true;
    if (readbuffer_ != nullptr)
    {
        download_finished = false;
        current_offset = 0;
        remote_readbuffer = readbuffer_;
        // setup local files
        out_file = new std::ofstream(local_path_ + "/data.bin", std::ios::out | std::ios::binary);
        out_file->flush();

        Poco::JSON::Object jobj;
        jobj.set("schema", schema_);
        jobj.set("cluster", cluster_);
        jobj.set("remote_path", path_);
        jobj.set("downloaded", "false");
        jobj.set("last_mod_ts", mod_ts_);
        std::stringstream buf; // STYLE_CHECK_ALLOW_STD_STRING_STREAM
        jobj.stringify(buf);
        Poco::FileOutputStream meta_file(local_path_ + "/meta.txt", std::ios::out);
        meta_file.write(buf.str().c_str(), buf.str().size());
        meta_file.close();

        backgroupDownload(finish_callback);
    }
}

RemoteReadBufferCacheError RemoteCacheController::waitMoreData(size_t start_offset_, size_t end_offset_)
{
    std::unique_lock lock{mutex};
    if (download_finished)
    {
        if (download_thread != nullptr)
        {
            download_thread->wait();
            LOG_TRACE(&Poco::Logger::get("RemoteCacheController"), "try to release down thread");
            delete download_thread;
            download_thread = nullptr;
        }
        // finish reading
        if (start_offset_ >= current_offset)
        {
            lock.unlock();
            return RemoteReadBufferCacheError::END_OF_FILE;
        }
    }
    else // block until more data is ready
    {
        if (current_offset >= end_offset_)
        {
            lock.unlock();
            return RemoteReadBufferCacheError::OK;
        }
        else
            more_data_signal.wait(lock, [this, end_offset_] { return this->download_finished || this->current_offset >= end_offset_; });
    }
    lock.unlock();
    return RemoteReadBufferCacheError::OK;
}

void RemoteCacheController::backgroupDownload(std::function<void(RemoteCacheController *)> const & finish_callback)
{
    download_thread = new ThreadPool(1);
    auto task = [this, finish_callback]()
    {
        size_t n = 0;
        size_t total_bytes = 0;
        while (!remote_readbuffer->eof())
        {
            size_t bytes = remote_readbuffer->buffer().end() - remote_readbuffer->position();
            out_file->write(remote_readbuffer->position(), bytes);
            remote_readbuffer->position() += bytes;
            total_bytes += bytes;
            if (n++ % 10 == 0)
            {
                std::unique_lock lock(mutex);
                current_offset += total_bytes;
                total_bytes = 0;
                flush();
                lock.unlock();
                more_data_signal.notify_all();
            }
        }
        std::unique_lock lock(mutex);
        current_offset += total_bytes;
        download_finished = true;
        flush();
        out_file->close();
        delete out_file;
        out_file = nullptr;
        remote_readbuffer = nullptr;
        lock.unlock();
        more_data_signal.notify_all();
        finish_callback(this);
        LOG_TRACE(
            &Poco::Logger::get("RemoteCacheController"), "finish download.{} into {}. size:{} ", remote_path, local_path, current_offset);
    };
    download_thread->scheduleOrThrow(task);
}

void RemoteCacheController::flush()
{
    if (out_file != nullptr)
    {
        out_file->flush();
    }
    Poco::JSON::Object jobj;
    jobj.set("schema", schema);
    jobj.set("cluster", cluster);
    jobj.set("remote_path", remote_path);
    jobj.set("downloaded", download_finished ? "true" : "false");
    jobj.set("last_mod_ts", last_mod_ts);
    std::stringstream buf; // STYLE_CHECK_ALLOW_STD_STRING_STREAM
    jobj.stringify(buf);

    std::ofstream meta_file(local_path + "/meta.txt", std::ios::out);
    meta_file << buf.str();
    meta_file.close();
}

RemoteCacheController::~RemoteCacheController()
{
    delete out_file;

    if (download_thread != nullptr)
    {
        download_thread->wait();
        delete download_thread;
    }
}

void RemoteCacheController::close()
{
    // delete the directory
    LOG_TRACE(&Poco::Logger::get("RemoteCacheController"), "release local resource: " + remote_path + ", " + local_path);
    std::filesystem::remove_all(local_path);
}

std::tuple<FILE *, std::string> RemoteCacheController::allocFile()
{
    std::string result_local_path;
    if (download_finished)
        result_local_path = local_path + "/data.bin";
    FILE * fs = fopen((local_path + "/data.bin").c_str(), "r");
    if (fs == nullptr)
        return {fs, result_local_path};
    std::lock_guard lock{mutex};
    opened_file_streams.insert(fs);
    return {fs, result_local_path};
}

void RemoteCacheController::deallocFile(FILE * fs)
{
    {
        std::lock_guard lock{mutex};
        auto it = opened_file_streams.find(fs);
        if (it == opened_file_streams.end())
        {
            std::string err = "try to close an invalid file " + remote_path;
            throw Exception(err, ErrorCodes::BAD_ARGUMENTS);
        }
        opened_file_streams.erase(it);
    }
    fclose(fs);
}

LocalCachedFileReader::LocalCachedFileReader(RemoteCacheController * cntrl_, size_t size_)
    : offset(0), file_size(size_), fs(nullptr), controller(cntrl_)
{
    std::tie(fs, local_path) = controller->allocFile();
    if (fs == nullptr)
        throw Exception("alloc file failed.", ErrorCodes::BAD_GET);
}
LocalCachedFileReader::~LocalCachedFileReader()
{
    controller->deallocFile(fs);
}

size_t LocalCachedFileReader::read(char * buf, size_t size)
{
    auto wret = controller->waitMoreData(offset, offset + size);
    if (wret != RemoteReadBufferCacheError::OK)
        return 0;
    std::lock_guard lock(mutex);
    auto ret_size = fread(buf, 1, size, fs);
    offset += ret_size;
    return ret_size;
}

off_t LocalCachedFileReader::seek(off_t off)
{
    controller->waitMoreData(off, 1);
    std::lock_guard lock(mutex);
    auto ret = fseek(fs, off, SEEK_SET);
    offset = off;
    if (ret != 0)
    {
        return -1;
    }
    return off;
}
size_t LocalCachedFileReader::size()
{
    if (file_size != 0)
        return file_size;
    if (local_path.empty())
    {
        LOG_TRACE(&Poco::Logger::get("LocalCachedFileReader"), "empty local_path");
        return 0;
    }

    auto ret = std::filesystem::file_size(local_path);
    file_size = ret;
    return ret;
}

// the size need be equal to the original buffer
RemoteReadBuffer::RemoteReadBuffer(size_t buff_size) : BufferWithOwnMemory<SeekableReadBuffer>(buff_size)
{
}

RemoteReadBuffer::~RemoteReadBuffer() = default;

std::unique_ptr<RemoteReadBuffer> RemoteReadBuffer::create(
    const std::string & schema_,
    const std::string & cluster_,
    const std::string & remote_path_,
    UInt64 mod_ts_,
    size_t file_size_,
    std::unique_ptr<ReadBuffer> readbuffer)
{
    size_t buff_size = DBMS_DEFAULT_BUFFER_SIZE;
    if (readbuffer != nullptr)
        buff_size = readbuffer->internalBuffer().size();
    /*
     * in the new implement of ReadBufferFromHDFS, buffer size is 0.
     *
     * in the common case, we don't read bytes from readbuffer directly, so set buff_size = DBMS_DEFAULT_BUFFER_SIZE
     * is OK.
     *
     * we need be careful with the case  without local file reader.
     */
    if (buff_size == 0)
        buff_size = DBMS_DEFAULT_BUFFER_SIZE;

    auto rrb = std::make_unique<RemoteReadBuffer>(buff_size);
    auto * raw_rbp = readbuffer.release();
    std::shared_ptr<ReadBuffer> srb(raw_rbp);
    RemoteReadBufferCacheError error;
    int retry = 0;
    do
    {
        if (retry > 0)
            usleep(20 * retry);

        std::tie(rrb->file_reader, error)
            = RemoteReadBufferCache::instance().createReader(schema_, cluster_, remote_path_, mod_ts_, file_size_, srb);
        retry++;
    } while (error == RemoteReadBufferCacheError::FILE_INVALID && retry < 10);
    if (rrb->file_reader == nullptr)
    {
        LOG_ERROR(&Poco::Logger::get("RemoteReadBuffer"), "allocate local file failed for " + remote_path_ + "{}", error);
        rrb->original_readbuffer = srb;
    }
    return rrb;
}

bool RemoteReadBuffer::nextImpl()
{
    if (file_reader != nullptr)
    {
        int bytes_read = file_reader->read(internal_buffer.begin(), internal_buffer.size());
        if (bytes_read)
            working_buffer.resize(bytes_read);
        else
        {
            return false;
        }
    }
    else // in the case we cannot use local cache, read from the original readbuffer directly
    {
        if (original_readbuffer == nullptr)
            throw Exception("original readbuffer should not be null", ErrorCodes::LOGICAL_ERROR);
        auto status = original_readbuffer->next();
        // we don't need to worry about the memory buffer allocated in RemoteReadBuffer, since it is owned by
        // BufferWithOwnMemory, BufferWithOwnMemory would release it.
        //LOG_TRACE(&Poco::Logger::get("RemoteReadBuffer"), "from original rb {} {}", original_readbuffer->buffer().size(), original_readbuffer->offset());
        if (status)
            BufferBase::set(original_readbuffer->buffer().begin(), original_readbuffer->buffer().size(), original_readbuffer->offset());
        return status;
    }
    return true;
}

off_t RemoteReadBuffer::seek(off_t offset, int whence)
{
    off_t pos_in_file = file_reader->getOffset();
    off_t new_pos;
    if (whence == SEEK_SET)
        new_pos = offset;
    else if (whence == SEEK_CUR)
        new_pos = pos_in_file - (working_buffer.end() - pos) + offset;
    else
        throw Exception("expects SEEK_SET or SEEK_CUR as whence", ErrorCodes::BAD_ARGUMENTS);

    /// Position is unchanged.
    if (new_pos + (working_buffer.end() - pos) == pos_in_file)
        return new_pos;

    if (new_pos <= pos_in_file && new_pos >= pos_in_file - static_cast<off_t>(working_buffer.size()))
    {
        /// Position is still inside buffer.
        pos = working_buffer.begin() + (new_pos - (pos_in_file - working_buffer.size()));
        return new_pos;
    }

    pos = working_buffer.end();
    auto ret_off = file_reader->seek(new_pos);
    if (ret_off == -1)
        throw Exception(
            "seek file failed. " + std::to_string(pos_in_file) + "->" + std::to_string(new_pos) + "@" + std::to_string(file_reader->size())
                + "," + std::to_string(whence) + "," + file_reader->getPath(),
            ErrorCodes::BAD_ARGUMENTS);
    return ret_off;
}

off_t RemoteReadBuffer::getPosition()
{
    return file_reader->getOffset() - available();
}

RemoteReadBufferCache::RemoteReadBufferCache() = default;

RemoteReadBufferCache::~RemoteReadBufferCache() = default;

RemoteReadBufferCache & RemoteReadBufferCache::instance()
{
    static RemoteReadBufferCache instance;
    return instance;
}

void RemoteReadBufferCache::initOnce(const std::string & dir, size_t limit_size_)
{
    LOG_TRACE(log, "init local cache. path: {}, limit {}", dir, limit_size_);
    std::lock_guard lock(mutex);
    local_path_prefix = dir;
    limit_size = limit_size_;

    // scan local disk dir and recover the cache metas
    std::filesystem::path root_dir(local_path_prefix);
    if (!std::filesystem::exists(root_dir))
    {
        LOG_INFO(log, "{} not exists. this cache will be disable", local_path_prefix);
        return;
    }
    auto callback = [this](RemoteCacheController * cntrl) { this->total_size += cntrl->size(); };

    // four level dir. <schema>/<cluster>/<first 3 chars of path hash code>/<path hash code>``
    for (auto const &schema_dir : std::filesystem::directory_iterator{root_dir})
    {
        for (auto const &cluster_dir : std::filesystem::directory_iterator{schema_dir.path()})
        {
            for (auto const &first_hash_dir : std::filesystem::directory_iterator{cluster_dir.path()})
            {
                for (auto const &second_hash_dir : std::filesystem::directory_iterator{first_hash_dir.path()})
                {
                    std::string path = second_hash_dir.path().string();
                    if (caches.count(path))
                    {
                        LOG_ERROR(log, "duplicated file:{}", path);
                        continue;
                    }
                    auto cache_cntrl = RemoteCacheController::recover(path, callback);
                    if (cache_cntrl == nullptr)
                        continue;
                    CacheCell cell;
                    cell.cache_controller = cache_cntrl;
                    cell.key_iterator = keys.insert(keys.end(), path);
                    caches[path] = cell;
                }
            }
        }
    }
    inited = true;
}

std::string
RemoteReadBufferCache::calculateLocalPath(const std::string & schema_, const std::string & cluster_, const std::string & remote_path_)
{
    std::string local_path = local_path_prefix + "/" + schema_ + "/" + cluster_;

    Poco::MD5Engine md5;
    Poco::DigestOutputStream outstr(md5);
    outstr << remote_path_;
    outstr.flush(); //to pass everything to the digest engine
    const Poco::DigestEngine::Digest & digest = md5.digest();
    std::string md5string = Poco::DigestEngine::digestToHex(digest);

    local_path += "/" + md5string.substr(0, 3) + "/" + md5string;

    return local_path;
}

std::tuple<std::shared_ptr<LocalCachedFileReader>, RemoteReadBufferCacheError> RemoteReadBufferCache::createReader(
    const std::string & schema,
    const std::string & cluster,
    const std::string & remote_path,
    UInt64 mod_ts,
    size_t file_size,
    std::shared_ptr<ReadBuffer> & readbuffer)
{
    if (!hasInitialized())
    {
        LOG_ERROR(log, "RemoteReadBufferCache not init");
        return {nullptr, RemoteReadBufferCacheError::NOT_INIT};
    }
    auto local_path = calculateLocalPath(schema, cluster, remote_path);
    std::lock_guard lock(mutex);
    auto cache_iter = caches.find(local_path);
    if (cache_iter != caches.end())
    {
        // if the file has been update on remote side, we need to redownload it
        if (cache_iter->second.cache_controller->getLastModTS() != mod_ts)
        {
            LOG_TRACE(log,
                "remote file has been updated. " + remote_path + ":" + std::to_string(cache_iter->second.cache_controller->getLastModTS()) + "->"
                    + std::to_string(mod_ts));
            cache_iter->second.cache_controller->markInvalid();
        }
        else
        {
            // move the key to the list end
            keys.splice(keys.end(), keys, cache_iter->second.key_iterator);
            return {std::make_shared<LocalCachedFileReader>(cache_iter->second.cache_controller.get(), file_size), RemoteReadBufferCacheError::OK};
        }
    }

    auto clear_ret = clearLocalCache();
    cache_iter = caches.find(local_path);
    if (cache_iter != caches.end())
    {
        if (cache_iter->second.cache_controller->isValid())
        {
            // move the key to the list end, this case should not happen?
            keys.splice(keys.end(), keys, cache_iter->second.key_iterator);
            return {std::make_shared<LocalCachedFileReader>(cache_iter->second.cache_controller.get(), file_size), RemoteReadBufferCacheError::OK};
        }
        else
        {
            // maybe someone is holding this file
            return {nullptr, RemoteReadBufferCacheError::FILE_INVALID};
        }
    }

    // reach the disk capacity limit
    if (!clear_ret)
    {
        LOG_ERROR(log, "local cache is full, return nullptr");
        return {nullptr, RemoteReadBufferCacheError::DISK_FULL};
    }

    std::filesystem::create_directory(local_path);

    auto callback = [this](RemoteCacheController * cntrl) { this->total_size += cntrl->size(); };
    auto cache_cntrl = std::make_shared<RemoteCacheController>(schema, cluster, remote_path, mod_ts, local_path, readbuffer, callback);
    CacheCell cc;
    cc.cache_controller = cache_cntrl;
    cc.key_iterator = keys.insert(keys.end(), local_path);
    caches[local_path] = cc;

    return {std::make_shared<LocalCachedFileReader>(cache_cntrl.get(), file_size), RemoteReadBufferCacheError::OK};
}

bool RemoteReadBufferCache::clearLocalCache()
{
    for (auto it = keys.begin(); it != keys.end();)
    {
        auto cit = caches.find(*it);
        auto cntrl = cit->second.cache_controller;
        if (!cntrl->isValid() && cntrl->closable())
        {
            LOG_TRACE(log, "clear invalid cache: " + *it);
            total_size = total_size > cit->second.cache_controller->size() ? total_size - cit->second.cache_controller->size() : 0;
            cntrl->close();
            it = keys.erase(it);
            caches.erase(cit);
        }
        else
            it++;
    }
    // clear closable cache from the list head
    for (auto it = keys.begin(); it != keys.end();)
    {
        if (total_size < limit_size)
            break;
        auto cit = caches.find(*it);
        if (cit == caches.end())
        {
            throw Exception("file not found in cache?" + *it, ErrorCodes::LOGICAL_ERROR);
        }
        if (cit->second.cache_controller->closable())
        {
            total_size = total_size > cit->second.cache_controller->size() ? total_size - cit->second.cache_controller->size() : 0;
            cit->second.cache_controller->close();
            caches.erase(cit);
            it = keys.erase(it);
            LOG_TRACE(log, "clear local file {} for {}. key size:{}. next{}", cit->second.cache_controller->getLocalPath(),
                cit->second.cache_controller->getRemotePath(), keys.size(), *it);
        }
        else
            break;
    }
    LOG_TRACE(log, "keys size:{}, total_size:{}, limit size:{}", keys.size(), total_size, limit_size);
    return total_size < limit_size * 1.5;
}

}
