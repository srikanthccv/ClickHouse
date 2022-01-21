#pragma once

#include <memory>
#include <shared_mutex>
#include <Storages/IStorage.h>
#include <Storages/IKVStorage.h>
#include <rocksdb/status.h>


namespace rocksdb
{
    class DB;
    class Statistics;
}


namespace DB
{

class Context;

class StorageEmbeddedRocksDB final : public IKeyValueStorage, WithContext
{
    friend class EmbeddedRocksDBSink;
public:
    StorageEmbeddedRocksDB(const StorageID & table_id_,
        const String & relative_data_path_,
        const StorageInMemoryMetadata & metadata,
        bool attach,
        ContextPtr context_,
        const String & primary_key_);

    std::string getName() const override { return "EmbeddedRocksDB"; }

    Pipe read(
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

    SinkToStoragePtr write(const ASTPtr & query, const StorageMetadataPtr & /*metadata_snapshot*/, ContextPtr context) override;
    void truncate(const ASTPtr &, const StorageMetadataPtr & metadata_snapshot, ContextPtr, TableExclusiveLockHolder &) override;

    bool supportsParallelInsert() const override { return true; }
    bool supportsIndexForIn() const override { return true; }
    bool mayBenefitFromIndexForIn(
        const ASTPtr & node, ContextPtr /*query_context*/, const StorageMetadataPtr & /*metadata_snapshot*/) const override
    {
        return node->getColumnName() == primary_key;
    }

    bool storesDataOnDisk() const override { return true; }
    Strings getDataPaths() const override { return {rocksdb_dir}; }

    std::shared_ptr<rocksdb::Statistics> getRocksDBStatistics() const;
    std::vector<rocksdb::Status> multiGet(const std::vector<rocksdb::Slice> & slices_keys, std::vector<String> & values) const;
    const String & getPrimaryKey() const override { return primary_key; }

    FieldVector::const_iterator getByKeys(
        FieldVector::const_iterator begin,
        FieldVector::const_iterator end,
        const Block & sample_block,
        Chunk & result,
        PaddedPODArray<UInt8> * null_map,
        size_t max_block_size) const override;

protected:
    StorageEmbeddedRocksDB(const StorageID & table_id_,
        const String & relative_data_path_,
        const StorageInMemoryMetadata & metadata,
        bool attach,
        ContextPtr context_,
        const String & primary_key_);

private:
    const String primary_key;
    using RocksDBPtr = std::unique_ptr<rocksdb::DB>;
    RocksDBPtr rocksdb_ptr;
    mutable std::shared_mutex rocksdb_ptr_mx;
    String rocksdb_dir;

    void initDB();
};
}
