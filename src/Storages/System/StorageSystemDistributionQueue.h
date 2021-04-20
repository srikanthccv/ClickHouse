#pragma once

#include <ext/shared_ptr_helper.h>
#include <Storages/System/IStorageSystemOneBlock.h>


namespace DB
{




/** Implements the `distribution_queue` system table, which allows you to view the INSERT queues for the Distributed tables.
  */
class StorageSystemDistributionQueue final : public ext::shared_ptr_helper<StorageSystemDistributionQueue>, public IStorageSystemOneBlock<StorageSystemDistributionQueue>
{
    friend struct ext::shared_ptr_helper<StorageSystemDistributionQueue>;
public:
    std::string getName() const override { return "SystemDistributionQueue"; }

    static NamesAndTypesList getNamesAndTypes();

protected:
    using IStorageSystemOneBlock::IStorageSystemOneBlock;

    void fillData(MutableColumns & res_columns, ContextPtr context, const SelectQueryInfo & query_info) const override;
};

}
