#pragma once

#include <ext/shared_ptr_helper.h>
#include <Storages/IStorage.h>


namespace DB
{




/** Implements `replicas` system table, which provides information about the status of the replicated tables.
  */
class StorageSystemReplicas final : public ext::shared_ptr_helper<StorageSystemReplicas>, public IStorage
{
    friend struct ext::shared_ptr_helper<StorageSystemReplicas>;
public:
    std::string getName() const override { return "SystemReplicas"; }

    Pipe read(
        const Names & column_names,
        const StorageMetadataPtr & /*metadata_snapshot*/,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

protected:
    StorageSystemReplicas(const StorageID & table_id_);
};

}
