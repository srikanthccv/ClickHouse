#pragma once

#include <Storages/IStorage.h>

#include <Poco/File.h>
#include <Poco/Path.h>

#include <common/logger_useful.h>

#include <atomic>
#include <shared_mutex>
#include <ext/shared_ptr_helper.h>


namespace DB
{

class StorageFileBlockInputStream;
class StorageFileBlockOutputStream;

class StorageFile final : public ext::shared_ptr_helper<StorageFile>, public IStorage
{
    friend struct ext::shared_ptr_helper<StorageFile>;
public:
    std::string getName() const override { return "File"; }

    Pipe read(
        const Names & column_names,
        const StorageMetadataPtr & /*metadata_snapshot*/,
        SelectQueryInfo & query_info,
        const Context & context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

    BlockOutputStreamPtr write(
        const ASTPtr & query,
        const StorageMetadataPtr & /*metadata_snapshot*/,
        const Context & context) override;

    void truncate(
        const ASTPtr & /*query*/,
        const StorageMetadataPtr & /* metadata_snapshot */,
        const Context & /* context */,
        TableExclusiveLockHolder &) override;

    void rename(const String & new_path_to_table_data, const StorageID & new_table_id) override;

    bool storesDataOnDisk() const override;
    Strings getDataPaths() const override;

    struct CommonArguments
    {
        StorageID table_id;
        std::string format_name;
        std::optional<FormatSettings> format_settings;
        std::string compression_method;
        const ColumnsDescription & columns;
        const ConstraintsDescription & constraints;
        const Context & context;
    };

    NamesAndTypesList getVirtuals() const override;

    static Strings getPathsList(const String & table_path, const String & user_files_path, const Context & context);

    /// Check if the format is column-oriented. Maybe it's more appropriate to put one flag in this class's private fields
    /// to identify this format type. We can set it in new formats' register code which will be automatically set at initialization.
    /// This will avoid updating this function frequently.
    bool isColumnOriented() const;

protected:
    friend class StorageFileSource;
    friend class StorageFileBlockOutputStream;

    /// From file descriptor
    StorageFile(int table_fd_, CommonArguments args);

    /// From user's file
    StorageFile(const std::string & table_path_, const std::string & user_files_path, CommonArguments args);

    /// From table in database
    StorageFile(const std::string & relative_table_dir_path, CommonArguments args);

private:
    explicit StorageFile(CommonArguments args);

    std::string format_name;
    // We use format settings from global context + CREATE query for File table
    // function -- in this case, format_settings is set.
    // For `file` table function, we use format settings from current user context,
    // in this case, format_settings is not set.
    std::optional<FormatSettings> format_settings;

    int table_fd = -1;
    String compression_method;

    std::string base_path;
    std::vector<std::string> paths;

    bool is_db_table = true;                     /// Table is stored in real database, not user's file
    bool use_table_fd = false;                    /// Use table_fd instead of path
    std::atomic<bool> table_fd_was_used{false}; /// To detect repeating reads from stdin
    off_t table_fd_init_offset = -1;            /// Initial position of fd, used for repeating reads

    mutable std::shared_timed_mutex rwlock;

    Poco::Logger * log = &Poco::Logger::get("StorageFile");
};

}
