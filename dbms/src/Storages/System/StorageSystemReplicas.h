#pragma once

#include <ext/shared_ptr_helper.h>
#include <Storages/IStorage.h>


namespace DB
{

class Context;


/** Implements `replicas` system table, which provides information about the status of the replicated tables.
  */
class StorageSystemReplicas : public ext::shared_ptr_helper<StorageSystemReplicas>, public IStorage
{
public:
    std::string getName() const override { return "SystemReplicas"; }
    std::string getTableName() const override { return name; }

    BlockInputStreams read(
        const Names & column_names,
        const SelectQueryInfo & query_info,
        const Context & context,
        QueryProcessingStage::Enum processed_stage,
        UInt64 max_block_size,
        unsigned num_streams) override;

private:
    const std::string name;

protected:
    StorageSystemReplicas(const std::string & name_);
};

}
