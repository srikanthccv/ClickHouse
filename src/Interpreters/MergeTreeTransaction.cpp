#include <Interpreters/MergeTreeTransaction.h>
#include <Storages/MergeTree/IMergeTreeDataPart.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Interpreters/TransactionLog.h>

namespace DB
{

MergeTreeTransaction::MergeTreeTransaction(Snapshot snapshot_, LocalTID local_tid_, UUID host_id)
    : tid({snapshot_, local_tid_, host_id})
    , snapshot(snapshot_)
    , csn(Tx::UnknownCSN)
{
}

MergeTreeTransaction::State MergeTreeTransaction::getState() const
{
    if (csn == Tx::UnknownCSN)
        return RUNNING;
    if (csn == Tx::RolledBackCSN)
        return ROLLED_BACK;
    return COMMITTED;
}

void MergeTreeTransaction::addNewPart(const StoragePtr & storage, const DataPartPtr & new_part, MergeTreeTransaction * txn)
{
    TransactionID tid = txn ? txn->tid : Tx::PrehistoricTID;

    new_part->versions.setMinTID(tid);
    if (txn)
        txn->addNewPart(storage, new_part);
}

void MergeTreeTransaction::removeOldPart(const StoragePtr & storage, const DataPartPtr & part_to_remove, MergeTreeTransaction * txn)
{
    TransactionID tid = txn ? txn->tid : Tx::PrehistoricTID;
    String error_context = fmt::format("Table: {}, part name: {}",
                                       part_to_remove->storage.getStorageID().getNameForLogs(),
                                       part_to_remove->name);
    part_to_remove->versions.lockMaxTID(tid, error_context);
    if (txn)
        txn->removeOldPart(storage, part_to_remove);
}

void MergeTreeTransaction::addNewPartAndRemoveCovered(const StoragePtr & storage, const DataPartPtr & new_part, const DataPartsVector & covered_parts, MergeTreeTransaction * txn)
{
    TransactionID tid = txn ? txn->tid : Tx::PrehistoricTID;

    new_part->versions.setMinTID(tid);
    if (txn)
        txn->addNewPart(storage, new_part);

    String error_context = fmt::format("Table: {}, covering part name: {}",
                                       new_part->storage.getStorageID().getNameForLogs(),
                                       new_part->name);
    error_context += ", part_name: {}";
    for (const auto & covered : covered_parts)
    {
        covered->versions.lockMaxTID(tid, fmt::format(error_context, covered->name));
        if (txn)
            txn->removeOldPart(storage, covered);
    }
}

void MergeTreeTransaction::addNewPart(const StoragePtr & storage, const DataPartPtr & new_part)
{
    assert(csn == Tx::UnknownCSN);
    storages.insert(storage);
    creating_parts.push_back(new_part);
}

void MergeTreeTransaction::removeOldPart(const StoragePtr & storage, const DataPartPtr & part_to_remove)
{
    assert(csn == Tx::UnknownCSN);
    storages.insert(storage);
    removing_parts.push_back(part_to_remove);
}

bool MergeTreeTransaction::isReadOnly() const
{
    return creating_parts.empty() && removing_parts.empty();
}

void MergeTreeTransaction::beforeCommit() const
{
    assert(csn == Tx::UnknownCSN);
}

void MergeTreeTransaction::afterCommit(CSN assigned_csn) noexcept
{
    assert(csn == Tx::UnknownCSN);
    csn = assigned_csn;
    for (const auto & part : creating_parts)
        part->versions.mincsn.store(csn);
    for (const auto & part : removing_parts)
        part->versions.maxcsn.store(csn);
}

void MergeTreeTransaction::rollback() noexcept
{
    assert(csn == Tx::UnknownCSN);
    csn = Tx::RolledBackCSN;
    for (const auto & part : creating_parts)
        part->versions.mincsn.store(Tx::RolledBackCSN);

    for (const auto & part : removing_parts)
        part->versions.unlockMaxTID(tid);

    /// FIXME const_cast
    for (const auto & part : creating_parts)
        const_cast<MergeTreeData &>(part->storage).removePartsFromWorkingSet(nullptr, {part}, true);

    for (const auto & part : removing_parts)
        if (part->versions.getMinTID() != tid)
            const_cast<MergeTreeData &>(part->storage).restoreAndActivatePart(part);

    /// FIXME seems like session holds shared_ptr to Transaction and transaction holds shared_ptr to parts preventing cleanup
}

void MergeTreeTransaction::onException()
{
    if (csn)
        return;

    TransactionLog::instance().rollbackTransaction(shared_from_this());
}

String MergeTreeTransaction::dumpDescription() const
{
    String res = "\ncreating parts:\n";
    for (const auto & part : creating_parts)
    {
        res += part->name;
        res += "\n";
    }

    res += "removing parts:\n";
    for (const auto & part : removing_parts)
    {
        res += part->name;
        res += fmt::format(" (created by {}, {})\n", part->versions.getMinTID(), part->versions.mincsn);
        assert(!part->versions.mincsn || part->versions.mincsn <= snapshot);
        assert(!part->versions.maxcsn);
    }

    return res;
}

}
