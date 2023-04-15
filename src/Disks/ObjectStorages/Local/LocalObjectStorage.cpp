#include <Disks/ObjectStorages/Local/LocalObjectStorage.h>

#include <Disks/ObjectStorages/DiskObjectStorageCommon.h>
#include <Common/filesystemHelpers.h>
#include <Common/logger_useful.h>
#include <Disks/IO/ReadIndirectBufferFromRemoteFS.h>
#include <Disks/IO/ReadBufferFromRemoteFSGather.h>
#include <Disks/IO/createReadBufferFromFileBase.h>
#include <Disks/IO/WriteIndirectBufferFromRemoteFS.h>
#include <IO/SeekAvoidingReadBuffer.h>
#include <IO/WriteBufferFromFile.h>
#include <Common/getRandomASCIIString.h>
#include <IO/BoundedReadBuffer.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
}

namespace ErrorCodes
{
    extern const int CANNOT_UNLINK;
}

LocalObjectStorage::LocalObjectStorage()
    : log(&Poco::Logger::get("LocalObjectStorage"))
{
    data_source_description.type = DataSourceType::Local;
    if (auto block_device_id = tryGetBlockDeviceId("/"); block_device_id.has_value())
        data_source_description.description = *block_device_id;
    else
        data_source_description.description = "/";

    data_source_description.is_cached = false;
    data_source_description.is_encrypted = false;
}

bool LocalObjectStorage::exists(const StoredObject & object) const
{
    return fs::exists(object.absolute_path);
}

std::unique_ptr<ReadBufferFromFileBase> LocalObjectStorage::readObjects( /// NOLINT
    const StoredObjects & objects,
    const ReadSettings & read_settings,
    std::optional<size_t> read_hint,
    std::optional<size_t> file_size) const
{
    auto modified_settings = patchSettings(read_settings);
    auto read_buffer_creator =
        [=] (const std::string & file_path, size_t /* read_until_position */)
        -> std::shared_ptr<ReadBufferFromFileBase>
    {
        auto impl = createReadBufferFromFileBase(file_path, modified_settings, read_hint, file_size);
        if (modified_settings.enable_filesystem_cache)
        {
            return std::make_unique<BoundedReadBuffer>(std::move(impl));
        }
        return impl;
    };

    auto impl = std::make_unique<ReadBufferFromRemoteFSGather>(
        std::move(read_buffer_creator), objects, modified_settings);

    if (read_settings.remote_fs_method == RemoteFSReadMethod::threadpool)
    {
        auto & reader = getThreadPoolReader();
        return std::make_unique<AsynchronousReadIndirectBufferFromRemoteFS>(reader, modified_settings, std::move(impl));
    }
    else
    {
        auto buf = std::make_unique<ReadIndirectBufferFromRemoteFS>(std::move(impl), modified_settings);
        return std::make_unique<SeekAvoidingReadBuffer>(std::move(buf), modified_settings.remote_read_min_bytes_for_seek);
    }
}

std::string LocalObjectStorage::getUniqueId(const std::string & path) const
{
    return toString(getINodeNumberFromPath(path));
}

ReadSettings LocalObjectStorage::patchSettings(const ReadSettings & read_settings) const
{
    if (!read_settings.enable_filesystem_cache)
        return IObjectStorage::patchSettings(read_settings);

    auto modified_settings{read_settings};
    /// For now we cannot allow asynchronous reader from local filesystem when CachedObjectStorage is used.
    switch (modified_settings.local_fs_method)
    {
        case LocalFSReadMethod::pread_threadpool:
        case LocalFSReadMethod::pread_fake_async:
        {
            modified_settings.local_fs_method = LocalFSReadMethod::pread;
            LOG_INFO(log, "Changing local filesystem read method to `pread`");
            break;
        }
        default:
        {
            break;
        }
    }
    return IObjectStorage::patchSettings(modified_settings);
}

std::unique_ptr<ReadBufferFromFileBase> LocalObjectStorage::readObject( /// NOLINT
    const StoredObject & object,
    const ReadSettings & read_settings,
    std::optional<size_t> read_hint,
    std::optional<size_t> file_size) const
{
    const auto & path = object.absolute_path;

    if (!file_size)
        file_size = tryGetSizeFromFilePath(path);

    LOG_TEST(log, "Read object: {}", path);
    return createReadBufferFromFileBase(path, patchSettings(read_settings), read_hint, file_size);
}

std::unique_ptr<WriteBufferFromFileBase> LocalObjectStorage::writeObject( /// NOLINT
    const StoredObject & object,
    WriteMode mode,
    std::optional<ObjectAttributes> /* attributes */,
    FinalizeCallback && finalize_callback,
    size_t buf_size,
    const WriteSettings & /* write_settings */)
{
    int flags = (mode == WriteMode::Append) ? (O_APPEND | O_CREAT | O_WRONLY) : -1;
    LOG_TEST(log, "Write object: {}", object.absolute_path);
    auto impl = std::make_unique<WriteBufferFromFile>(object.absolute_path, buf_size, flags);
    return std::make_unique<WriteIndirectBufferFromRemoteFS>(std::move(impl), std::move(finalize_callback), object.absolute_path);
}

void LocalObjectStorage::removeObject(const StoredObject & object)
{
    /// For local object storage files are actually removed when "metadata" is removed.
    if (!exists(object))
        return;

    if (0 != unlink(object.absolute_path.data()))
        throwFromErrnoWithPath("Cannot unlink file " + object.absolute_path, object.absolute_path, ErrorCodes::CANNOT_UNLINK);
}

void LocalObjectStorage::removeObjects(const StoredObjects & objects)
{
    for (const auto & object : objects)
        removeObject(object);
}

void LocalObjectStorage::removeObjectIfExists(const StoredObject & object)
{
    if (exists(object))
        removeObject(object);
}

void LocalObjectStorage::removeObjectsIfExist(const StoredObjects & objects)
{
    for (const auto & object : objects)
        removeObjectIfExists(object);
}

ObjectMetadata LocalObjectStorage::getObjectMetadata(const std::string & /* path */) const
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Metadata is not supported for LocalObjectStorage");
}

void LocalObjectStorage::copyObject( // NOLINT
    const StoredObject & object_from, const StoredObject & object_to, std::optional<ObjectAttributes> /* object_to_attributes */)
{
    fs::path to = object_to.absolute_path;
    fs::path from = object_from.absolute_path;

    /// Same logic as in DiskLocal.
    if (object_from.absolute_path.ends_with('/'))
        from = from.parent_path();
    if (fs::is_directory(from))
        to /= from.filename();

    fs::copy(from, to, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
}

void LocalObjectStorage::shutdown()
{
}

void LocalObjectStorage::startup()
{
}

std::unique_ptr<IObjectStorage> LocalObjectStorage::cloneObjectStorage(
    const std::string & /* new_namespace */,
    const Poco::Util::AbstractConfiguration & /* config */,
    const std::string & /* config_prefix */, ContextPtr /* context */)
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "cloneObjectStorage() is not implemented for LocalObjectStorage");
}

void LocalObjectStorage::applyNewSettings(
    const Poco::Util::AbstractConfiguration & /* config */, const std::string & /* config_prefix */, ContextPtr /* context */)
{
}

std::string LocalObjectStorage::generateBlobNameForPath(const std::string & /* path */)
{
    constexpr size_t key_name_total_size = 32;
    return getRandomASCIIString(key_name_total_size);
}

}
