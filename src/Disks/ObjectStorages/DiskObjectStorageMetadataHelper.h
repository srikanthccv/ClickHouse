#pragma once

#include <Disks/ObjectStorages/IObjectStorage.h>

namespace DB
{

class DiskObjectStorage;

/// Class implements storage of ObjectStorage metadata inside object storage itself,
/// so it's possible to recover from this remote information in case of local disk loss.
///
/// This mechanism can be enabled with `<send_metadata>true</send_metadata>` option inside
/// disk configuration. Implemented only for S3 and Azure Block storage. Other object storages
/// doesn't support metadata for blobs.
///
/// FIXME: this class is very intrusive and use a lot of DiskObjectStorage internals.
/// FIXME: it's very complex and unreliable, need to implement something better.
class DiskObjectStorageMetadataHelper
{
public:
    static constexpr UInt64 LATEST_REVISION = std::numeric_limits<UInt64>::max();
    static constexpr UInt64 UNKNOWN_REVISION = 0;

    DiskObjectStorageMetadataHelper(DiskObjectStorage * disk_, ReadSettings read_settings_)
        : disk(disk_)
        , read_settings(std::move(read_settings_))
    {
    }

    struct RestoreInformation
    {
        UInt64 revision = LATEST_REVISION;
        String source_namespace;
        String source_path;
        bool detached = false;
    };

    using Futures = std::vector<std::future<void>>;

    void createFileOperationObject(const String & operation_name, UInt64 revision, const ObjectAttributes & metadata) const;

    void findLastRevision();

    static int readSchemaVersion(IObjectStorage * object_storage, const String & source_path);
    void saveSchemaVersion(const int & version) const;
    void updateObjectMetadata(const String & key, const ObjectAttributes & metadata) const;
    void migrateFileToRestorableSchema(const String & path) const;
    void migrateToRestorableSchemaRecursive(const String & path, Futures & results);
    void migrateToRestorableSchema();

    /// Most important method, called on DiskObjectStorage startup
    void restore(const Poco::Util::AbstractConfiguration & config, const std::string & config_prefix, ContextPtr context);

    void readRestoreInformation(RestoreInformation & restore_information);
    void restoreFiles(IObjectStorage * source_object_storage, const RestoreInformation & restore_information);
    void processRestoreFiles(IObjectStorage * source_object_storage, const String & source_path, const std::vector<String> & keys) const;
    void restoreFileOperations(IObjectStorage * source_object_storage, const RestoreInformation & restore_information);

    std::atomic<UInt64> revision_counter = 0;
    inline static const String RESTORE_FILE_NAME = "restore";

    /// Object contains information about schema version.
    inline static const String SCHEMA_VERSION_OBJECT = ".SCHEMA_VERSION";
    /// Version with possibility to backup-restore metadata.
    static constexpr int RESTORABLE_SCHEMA_VERSION = 1;
    /// Directories with data.
    const std::vector<String> data_roots {"data", "store"};

    DiskObjectStorage * disk;

    ObjectStoragePtr object_storage_from_another_namespace;

    ReadSettings read_settings;
};

}
