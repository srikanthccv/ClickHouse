#pragma once

#include <Storages/IStorage.h>
#include <Poco/URI.h>
#include <common/logger_useful.h>
#include <ext/shared_ptr_helper.h>


namespace DB
{
/**
 * This class represents table engine for external S3 urls.
 * It sends HTTP GET to server when select is called and
 * HTTP PUT when insert is called.
 */
class StorageS3 : public ext::shared_ptr_helper<StorageS3>, public IStorage
{
public:
    StorageS3(
        const Poco::URI & uri_,
        const String & access_key_id,
        const String & secret_access_key,
        const StorageID & table_id_,
        const String & format_name_,
        UInt64 min_upload_part_size_,
        const ColumnsDescription & columns_,
        const ConstraintsDescription & constraints_,
        Context & context_,
        const String & compression_method_);

    String getName() const override
    {
        return "S3";
    }

    Block getHeaderBlock(const Names & /*column_names*/) const
    {
        return getSampleBlock();
    }

    BlockInputStreams read(
        const Names & column_names,
        const SelectQueryInfo & query_info,
        const Context & context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

    BlockOutputStreamPtr write(const ASTPtr & query, const Context & context) override;

private:
    Poco::URI uri;
    String access_key_id;
    String secret_access_key;
    const Context & context_global;

    String format_name;
    UInt64 min_upload_part_size;
    String compression_method;
};

}
