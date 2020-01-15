#pragma once

#include <Disks/IDisk.h>
#include <IO/ReadBuffer.h>
#include <IO/WriteBuffer.h>

#include <mutex>
#include <unordered_map>

namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

class DiskMemory : public IDisk
{
public:
    DiskMemory(const String & name_) : name(name_), disk_path("memory://" + name_ + '/') { }

    const String & getName() const override { return name; }

    const String & getPath() const override { return disk_path; }

    ReservationPtr reserve(UInt64 bytes) override;

    UInt64 getTotalSpace() const override;

    UInt64 getAvailableSpace() const override;

    UInt64 getUnreservedSpace() const override;

    bool exists(const String & path) const override;

    bool isFile(const String & path) const override;

    bool isDirectory(const String & path) const override;

    size_t getFileSize(const String & path) const override;

    void createDirectory(const String & path) override;

    void createDirectories(const String & path) override;

    void clearDirectory(const String & path) override;

    void moveDirectory(const String & from_path, const String & to_path) override;

    DiskDirectoryIteratorPtr iterateDirectory(const String & path) override;

    void moveFile(const String & from_path, const String & to_path) override;

    void replaceFile(const String & from_path, const String & to_path) override;

    void copyFile(const String & from_path, const String & to_path) override;

    std::unique_ptr<ReadBuffer> readFile(const String & path, size_t buf_size = DBMS_DEFAULT_BUFFER_SIZE) const override;

    std::unique_ptr<WriteBuffer> writeFile(
        const String & path,
        size_t buf_size = DBMS_DEFAULT_BUFFER_SIZE,
        WriteMode mode = WriteMode::Rewrite) override;

    void remove(const String & path, bool recursive) override;

private:
    void createDirectoriesImpl(const String & path);
    void replaceFileImpl(const String & from_path, const String & to_path);

private:
    enum class FileType
    {
        File,
        Directory
    };

    struct FileData
    {
        FileType type;
        String data;

        explicit FileData(FileType type_) : type(type_) { }
    };
    using Files = std::unordered_map<String, FileData>; /// file path -> file data

    const String name;
    const String disk_path;
    Files files;
    mutable std::mutex mutex;
};

using DiskMemoryPtr = std::shared_ptr<DiskMemory>;


class DiskMemoryDirectoryIterator : public IDiskDirectoryIterator
{
public:
    explicit DiskMemoryDirectoryIterator(std::vector<String> && dir_file_paths_)
        : dir_file_paths(std::move(dir_file_paths_)), iter(dir_file_paths.begin())
    {
    }

    void next() override { ++iter; }

    bool isValid() const override { return iter != dir_file_paths.end(); }

    String path() const override { return *iter; }

private:
    std::vector<String> dir_file_paths;
    std::vector<String>::iterator iter;
};


class DiskFactory;
void registerDiskMemory(DiskFactory & factory);

}
