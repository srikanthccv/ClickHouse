#pragma once

#include <filesystem>
#include <string>
#include <map>
#include <optional>

#include <Poco/Timestamp.h>
#include <Core/Defines.h>
#include <Common/Exception.h>
#include <IO/ReadSettings.h>
#include <IO/WriteSettings.h>

#include <Disks/IO/AsynchronousReadIndirectBufferFromRemoteFS.h>
#include <Common/ThreadPool.h>
#include <Disks/WriteMode.h>


namespace DB
{

class ReadBufferFromFileBase;
class WriteBufferFromFileBase;

using ObjectAttributes = std::map<std::string, std::string>;

struct RelativePathWithSize
{
    String relative_path;
    size_t bytes_size;

    RelativePathWithSize() = default;

    RelativePathWithSize(const String & relative_path_, size_t bytes_size_)
        : relative_path(relative_path_), bytes_size(bytes_size_) {}
};
using RelativePathsWithSize = std::vector<RelativePathWithSize>;


/// Object metadata: path, size. cache_hint.
struct StoredObject
{
    std::string path; /// absolute
    uint64_t bytes_size;

    /// Optional cache hint for cache. Use delayed initialization
    /// because somecache hint implementation requires it.
    using CacheHintCreator = std::function<std::string(const std::string &)>;
    CacheHintCreator cache_hint_creator;

    StoredObject() = default;

    explicit StoredObject(
        const std::string & path_, uint64_t bytes_size_ = 0, CacheHintCreator && cache_hint_creator_ = {});

    std::string getCacheHint() const;
};

using StoredObjects = std::vector<StoredObject>;

struct ObjectMetadata
{
    uint64_t size_bytes;
    std::optional<Poco::Timestamp> last_modified;
    std::optional<ObjectAttributes> attributes;
};

using FinalizeCallback = std::function<void(size_t bytes_count)>;

/// Base class for all object storages which implement some subset of ordinary filesystem operations.
///
/// Examples of object storages are S3, Azure Blob Storage, HDFS.
class IObjectStorage
{
public:
    IObjectStorage() = default;

    /// Object exists or not
    virtual bool exists(const StoredObject & object) const = 0;

    /// List on prefix, return children (relative paths) with their sizes.
    virtual void listPrefix(const std::string & path, RelativePathsWithSize & children) const = 0;

    /// Get object metadata if supported. It should be possible to receive
    /// at least size of object
    virtual ObjectMetadata getObjectMetadata(const std::string & path) const = 0;

    /// Read single object
    virtual std::unique_ptr<ReadBufferFromFileBase> readObject( /// NOLINT
        const StoredObject & object,
        const ReadSettings & read_settings = ReadSettings{},
        std::optional<size_t> read_hint = {},
        std::optional<size_t> file_size = {}) const = 0;

    /// Read multiple objects with common prefix
    virtual std::unique_ptr<ReadBufferFromFileBase> readObjects( /// NOLINT
        const StoredObjects & objects,
        const ReadSettings & read_settings = ReadSettings{},
        std::optional<size_t> read_hint = {},
        std::optional<size_t> file_size = {}) const = 0;

    /// Open the file for write and return WriteBufferFromFileBase object.
    virtual std::unique_ptr<WriteBufferFromFileBase> writeObject( /// NOLINT
        const StoredObject & object,
        WriteMode mode,
        std::optional<ObjectAttributes> attributes = {},
        FinalizeCallback && finalize_callback = {},
        size_t buf_size = DBMS_DEFAULT_BUFFER_SIZE,
        const WriteSettings & write_settings = {}) = 0;

    virtual bool isRemote() const = 0;

    /// Remove object. Throws exception if object doesn't exists.
    virtual void removeObject(const StoredObject & object) = 0;

    /// Remove multiple objects. Some object storages can do batch remove in a more
    /// optimal way.
    virtual void removeObjects(const StoredObjects & objects) = 0;

    /// Remove object on path if exists
    virtual void removeObjectIfExists(const StoredObject & object) = 0;

    /// Remove objects on path if exists
    virtual void removeObjectsIfExist(const StoredObjects & object) = 0;

    /// Copy object with different attributes if required
    virtual void copyObject( /// NOLINT
        const StoredObject & object_from,
        const StoredObject & object_to,
        std::optional<ObjectAttributes> object_to_attributes = {}) = 0;

    /// Copy object to another instance of object storage
    /// by default just read the object from source object storage and write
    /// to destination through buffers.
    virtual void copyObjectToAnotherObjectStorage( /// NOLINT
        const StoredObject & object_from,
        const StoredObject & object_to,
        IObjectStorage & object_storage_to,
        std::optional<ObjectAttributes> object_to_attributes = {});

    virtual ~IObjectStorage() = default;

    /// Path to directory with objects cache
    virtual std::string getCacheBasePath() const;

    static AsynchronousReaderPtr getThreadPoolReader();

    static ThreadPool & getThreadPoolWriter();

    virtual void shutdown() = 0;

    virtual void startup() = 0;

    /// Apply new settings, in most cases reiniatilize client and some other staff
    virtual void applyNewSettings(
        const Poco::Util::AbstractConfiguration & config,
        const std::string & config_prefix,
        ContextPtr context) = 0;

    /// Sometimes object storages have something similar to chroot or namespace, for example
    /// buckets in S3. If object storage doesn't have any namepaces return empty string.
    virtual String getObjectsNamespace() const = 0;

    /// FIXME: confusing function required for a very specific case. Create new instance of object storage
    /// in different namespace.
    virtual std::unique_ptr<IObjectStorage> cloneObjectStorage(
        const std::string & new_namespace,
        const Poco::Util::AbstractConfiguration & config,
        const std::string & config_prefix, ContextPtr context) = 0;

    /// Generate object storage path.
    /// Path can be generated either independently or based on `path`.
    virtual std::string generateBlobNameForPath(const std::string & path) = 0;

    virtual bool supportsAppend() const { return false; }

    virtual void removeCacheIfExists(const std::string & /* path */) {}

    virtual bool isCached() const { return false; }
};

using ObjectStoragePtr = std::shared_ptr<IObjectStorage>;

}
