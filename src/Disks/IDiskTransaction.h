#pragma once

#include <string>
#include <vector>
#include <boost/noncopyable.hpp>
#include <Disks/IDisk.h>

namespace DB
{

struct RemoveRequest
{
    std::string path;
    bool if_exists = false;

    explicit RemoveRequest(std::string path_, bool if_exists_ = false)
        : path(std::move(path_)), if_exists(std::move(if_exists_))
    {
    }
};

using RemoveBatchRequest = std::vector<RemoveRequest>;

struct IDiskTransaction : private boost::noncopyable
{
public:
    virtual void commit() = 0;

    virtual ~IDiskTransaction() = default;

    /// Create directory.
    virtual void createDirectory(const std::string & path) = 0;

    /// Create directory and all parent directories if necessary.
    virtual void createDirectories(const std::string & path) = 0;

    /// Remove all files from the directory. Directories are not removed.
    virtual void clearDirectory(const std::string & path) = 0;

    /// Move directory from `from_path` to `to_path`.
    virtual void moveDirectory(const std::string & from_path, const std::string & to_path) = 0;

    /// Move the file from `from_path` to `to_path`.
    /// If a file with `to_path` path already exists, it will be replaced.
    virtual void replaceFile(const std::string & from_path, const std::string & to_path) = 0;

    /// Recursively copy data containing at `from_path` to `to_path` located at `to_disk`.
    virtual void copy(const std::string & from_path, const std::string & to_path) = 0;

    /// Recursively copy files from from_dir to to_dir. Create to_dir if not exists.
    virtual void copyDirectoryContent(const std::string & from_dir, const std::string & to_dir) = 0;

    /// Copy file `from_file_path` to `to_file_path` located at `to_disk`.
    virtual void copyFile(const std::string & from_file_path, const std::string & to_file_path) = 0;

    /// Open the file for write and return WriteBufferFromFileBase object.
    virtual std::unique_ptr<WriteBufferFromFileBase> writeFile( /// NOLINT
        const std::string & path,
        size_t buf_size = DBMS_DEFAULT_BUFFER_SIZE,
        WriteMode mode = WriteMode::Rewrite,
        const WriteSettings & settings = {}) = 0;

    /// Remove file. Throws exception if file doesn't exists or it's a directory.
    virtual void removeFile(const std::string & path) = 0;

    /// Remove file if it exists.
    virtual void removeFileIfExists(const std::string & path) = 0;

    /// Remove directory. Throws exception if it's not a directory or if directory is not empty.
    virtual void removeDirectory(const std::string & path) = 0;

    /// Remove file or directory with all children. Use with extra caution. Throws exception if file doesn't exists.
    virtual void removeRecursive(const std::string & path) = 0;

    /// Remove file. Throws exception if file doesn't exists or if directory is not empty.
    /// Differs from removeFile for S3/HDFS disks
    /// Second bool param is a flag to remove (true) or keep (false) shared data on S3
    virtual void removeSharedFile(const std::string & path, bool /* keep_shared_data */) = 0;

    /// Remove file or directory with all children. Use with extra caution. Throws exception if file doesn't exists.
    /// Differs from removeRecursive for S3/HDFS disks
    /// Second bool param is a flag to remove (false) or keep (true) shared data on S3.
    /// Third param determines which files cannot be removed even if second is true.
    virtual void removeSharedRecursive(const std::string & path, bool /* keep_all_shared_data */, const NameSet & /* file_names_remove_metadata_only */) = 0;

    /// Remove file or directory if it exists.
    /// Differs from removeFileIfExists for S3/HDFS disks
    /// Second bool param is a flag to remove (true) or keep (false) shared data on S3
    virtual void removeSharedFileIfExists(const std::string & path, bool /* keep_shared_data */) = 0;

    /// Batch request to remove multiple files.
    /// May be much faster for blob storage.
    /// Second bool param is a flag to remove (true) or keep (false) shared data on S3.
    /// Third param determines which files cannot be removed even if second is true.
    virtual void removeSharedFiles(const RemoveBatchRequest & files, bool keep_all_batch_data, const NameSet & file_names_remove_metadata_only) = 0;

    /// Set last modified time to file or directory at `path`.
    virtual void setLastModified(const std::string & path, const Poco::Timestamp & timestamp) = 0;

    /// Set file at `path` as read-only.
    virtual void setReadOnly(const std::string & path) = 0;

    /// Create hardlink from `src_path` to `dst_path`.
    virtual void createHardLink(const std::string & src_path, const std::string & dst_path) = 0;

};

using DiskTransactionPtr = std::shared_ptr<IDiskTransaction>;

}
