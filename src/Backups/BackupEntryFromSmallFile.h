#pragma once

#include <Backups/BackupEntryWithChecksumCalculation.h>


namespace DB
{
class IDisk;
using DiskPtr = std::shared_ptr<IDisk>;

/// Represents a file prepared to be included in a backup,
/// assuming that the file is small and can be easily loaded into memory.
class BackupEntryFromSmallFile : public BackupEntryWithChecksumCalculation<IBackupEntry>
{
public:
    explicit BackupEntryFromSmallFile(const String & file_path_);
    BackupEntryFromSmallFile(const DiskPtr & disk_, const String & file_path_);

    std::unique_ptr<SeekableReadBuffer> getReadBuffer() const override;
    UInt64 getSize() const override { return data.size(); }

    DataSourceDescription getDataSourceDescription() const override { return data_source_description; }
    bool isEncryptedByDisk() const override { return data_source_description.is_encrypted; }

    bool isFromFile() const override { return true; }
    DiskPtr getDisk() const override { return disk; }
    String getFilePath() const override { return file_path; }

private:
    const DiskPtr disk;
    const String file_path;
    const DataSourceDescription data_source_description;
    const String data;
};

}
