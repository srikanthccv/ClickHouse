#pragma once


namespace DB
{

/// Wraps another backup entry and a value of any type.
template <typename T>
class BackupEntryWrappedWith : public IBackupEntry
{
public:
    BackupEntryWrappedWith(BackupEntryPtr entry_, const T & custom_value_) : entry(entry_), custom_value(custom_value_) { }
    BackupEntryWrappedWith(BackupEntryPtr entry_, T && custom_value_) : entry(entry_), custom_value(std::move(custom_value_)) { }
    ~BackupEntryWrappedWith() override = default;

    UInt64 getSize() const override { return entry->getSize(); }
    std::optional<UInt128> getChecksum() const override { return entry->getChecksum(); }
    std::unique_ptr<SeekableReadBuffer> getReadBuffer() const override { return entry->getReadBuffer(); }
    String getFilePath() const override { return entry->getFilePath(); }
    DiskPtr tryGetDiskIfExists() const override { return entry->tryGetDiskIfExists(); }
    DataSourceDescription getDataSourceDescription() const override { return entry->getDataSourceDescription(); }

private:
    BackupEntryPtr entry;
    T custom_value;
};

}
