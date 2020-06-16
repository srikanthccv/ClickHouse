#include <Storages/StorageReplicatedMergeTree.h>
#include <Storages/MergeTree/ReplicatedMergeTreeQuorumEntry.h>
#include <Storages/MergeTree/ReplicatedMergeTreeBlockOutputStream.h>
#include <Interpreters/PartLog.h>
#include <DataStreams/IBlockOutputStream.h>
#include <Common/SipHash.h>
#include <Common/ZooKeeper/KeeperException.h>
#include <IO/Operators.h>


namespace ProfileEvents
{
    extern const Event DuplicatedInsertedBlocks;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_FEW_LIVE_REPLICAS;
    extern const int UNSATISFIED_QUORUM_FOR_PREVIOUS_WRITE;
    extern const int UNEXPECTED_ZOOKEEPER_ERROR;
    extern const int NO_ZOOKEEPER;
    extern const int READONLY;
    extern const int UNKNOWN_STATUS_OF_INSERT;
    extern const int INSERT_WAS_DEDUPLICATED;
    extern const int TIMEOUT_EXCEEDED;
    extern const int NO_ACTIVE_REPLICAS;
    extern const int DUPLICATE_DATA_PART;
    extern const int LOGICAL_ERROR;
}


ReplicatedMergeTreeBlockOutputStream::ReplicatedMergeTreeBlockOutputStream(
    StorageReplicatedMergeTree & storage_, size_t quorum_, size_t quorum_timeout_ms_, size_t max_parts_per_block_, bool deduplicate_)
    : storage(storage_), quorum(quorum_), quorum_timeout_ms(quorum_timeout_ms_), max_parts_per_block(max_parts_per_block_), deduplicate(deduplicate_),
    log(&Poco::Logger::get(storage.getLogName() + " (Replicated OutputStream)"))
{
    /// The quorum value `1` has the same meaning as if it is disabled.
    if (quorum == 1)
        quorum = 0;
}


Block ReplicatedMergeTreeBlockOutputStream::getHeader() const
{
    return storage.getSampleBlock();
}


/// Allow to verify that the session in ZooKeeper is still alive.
static void assertSessionIsNotExpired(zkutil::ZooKeeperPtr & zookeeper)
{
    if (!zookeeper)
        throw Exception("No ZooKeeper session.", ErrorCodes::NO_ZOOKEEPER);

    if (zookeeper->expired())
        throw Exception("ZooKeeper session has been expired.", ErrorCodes::NO_ZOOKEEPER);
}


void ReplicatedMergeTreeBlockOutputStream::checkQuorumPrecondition(zkutil::ZooKeeperPtr & zookeeper)
{
    quorum_info.status_path = storage.zookeeper_path + "/quorum/status";

    std::future<Coordination::GetResponse> quorum_status_future = zookeeper->asyncTryGet(quorum_info.status_path);
    std::future<Coordination::GetResponse> is_active_future = zookeeper->asyncTryGet(storage.replica_path + "/is_active");
    std::future<Coordination::GetResponse> host_future = zookeeper->asyncTryGet(storage.replica_path + "/host");

    /// List of live replicas. All of them register an ephemeral node for leader_election.

    Coordination::Stat leader_election_stat;
    zookeeper->get(storage.zookeeper_path + "/leader_election", &leader_election_stat);

    if (leader_election_stat.numChildren < static_cast<int32_t>(quorum))
        throw Exception("Number of alive replicas ("
            + toString(leader_election_stat.numChildren) + ") is less than requested quorum (" + toString(quorum) + ").",
            ErrorCodes::TOO_FEW_LIVE_REPLICAS);

    /** Is there a quorum for the last part for which a quorum is needed?
        * Write of all the parts with the included quorum is linearly ordered.
        * This means that at any time there can be only one part,
        *  for which you need, but not yet reach the quorum.
        * Information about this part will be located in `/quorum/status` node.
        * If the quorum is reached, then the node is deleted.
        */

    auto quorum_status = quorum_status_future.get();
    if (quorum_status.error != Coordination::Error::ZNONODE)
        throw Exception("Quorum for previous write has not been satisfied yet. Status: " + quorum_status.data, ErrorCodes::UNSATISFIED_QUORUM_FOR_PREVIOUS_WRITE);

    /// Both checks are implicitly made also later (otherwise there would be a race condition).

    auto is_active = is_active_future.get();
    auto host = host_future.get();

    if (is_active.error == Coordination::Error::ZNONODE || host.error == Coordination::Error::ZNONODE)
        throw Exception("Replica is not active right now", ErrorCodes::READONLY);

    quorum_info.is_active_node_value = is_active.data;
    quorum_info.is_active_node_version = is_active.stat.version;
    quorum_info.host_node_version = host.stat.version;
}


void ReplicatedMergeTreeBlockOutputStream::write(const Block & block)
{
    last_block_is_duplicate = false;

    /// TODO Is it possible to not lock the table structure here?
    storage.delayInsertOrThrowIfNeeded(&storage.partial_shutdown_event);

    auto zookeeper = storage.getZooKeeper();
    assertSessionIsNotExpired(zookeeper);

    /** If write is with quorum, then we check that the required number of replicas is now live,
      *  and also that for all previous parts for which quorum is required, this quorum is reached.
      * And also check that during the insertion, the replica was not reinitialized or disabled (by the value of `is_active` node).
      * TODO Too complex logic, you can do better.
      */
    if (quorum)
        checkQuorumPrecondition(zookeeper);

    auto part_blocks = storage.writer.splitBlockIntoParts(block, max_parts_per_block);

    for (auto & current_block : part_blocks)
    {
        Stopwatch watch;

        /// Write part to the filesystem under temporary name. Calculate a checksum.

        MergeTreeData::MutableDataPartPtr part = storage.writer.writeTempPart(current_block);

        String block_id;

        if (deduplicate)
        {
            SipHash hash;
            part->checksums.computeTotalChecksumDataOnly(hash);
            union
            {
                char bytes[16];
                UInt64 words[2];
            } hash_value;
            hash.get128(hash_value.bytes);

            /// We add the hash from the data and partition identifier to deduplication ID.
            /// That is, do not insert the same data to the same partition twice.
            block_id = part->info.partition_id + "_" + toString(hash_value.words[0]) + "_" + toString(hash_value.words[1]);

            LOG_DEBUG(log, "Wrote block with ID '{}', {} rows", block_id, current_block.block.rows());
        }
        else
        {
            LOG_DEBUG(log, "Wrote block with {} rows", current_block.block.rows());
        }

        try
        {
            commitPart(zookeeper, part, block_id);

            /// Set a special error code if the block is duplicate
            int error = (deduplicate && last_block_is_duplicate) ? ErrorCodes::INSERT_WAS_DEDUPLICATED : 0;
            PartLog::addNewPart(storage.global_context, part, watch.elapsed(), ExecutionStatus(error));
        }
        catch (...)
        {
            PartLog::addNewPart(storage.global_context, part, watch.elapsed(), ExecutionStatus::fromCurrentException(__PRETTY_FUNCTION__));
            throw;
        }
    }
}


void ReplicatedMergeTreeBlockOutputStream::writeExistingPart(MergeTreeData::MutableDataPartPtr & part)
{
    last_block_is_duplicate = false;

    /// NOTE: No delay in this case. That's Ok.

    auto zookeeper = storage.getZooKeeper();
    assertSessionIsNotExpired(zookeeper);

    if (quorum)
        checkQuorumPrecondition(zookeeper);

    Stopwatch watch;

    try
    {
        commitPart(zookeeper, part, "");
        PartLog::addNewPart(storage.global_context, part, watch.elapsed());
    }
    catch (...)
    {
        PartLog::addNewPart(storage.global_context, part, watch.elapsed(), ExecutionStatus::fromCurrentException(__PRETTY_FUNCTION__));
        throw;
    }
}


void ReplicatedMergeTreeBlockOutputStream::commitPart(
    zkutil::ZooKeeperPtr & zookeeper, MergeTreeData::MutableDataPartPtr & part, const String & block_id)
{
    storage.check(part->getColumns());
    assertSessionIsNotExpired(zookeeper);

    String temporary_part_name = part->name;

    while (true)
    {
        /// Obtain incremental block number and lock it. The lock holds our intention to add the block to the filesystem.
        /// We remove the lock just after renaming the part. In case of exception, block number will be marked as abandoned.
        /// Also, make deduplication check. If a duplicate is detected, no nodes are created.

        /// Allocate new block number and check for duplicates
        bool deduplicate_block = !block_id.empty();
        String block_id_path = deduplicate_block ? storage.zookeeper_path + "/blocks/" + block_id : "";
        auto block_number_lock = storage.allocateBlockNumber(part->info.partition_id, zookeeper, block_id_path);

        Int64 block_number;
        String existing_part_name;
        if (block_number_lock)
        {
            block_number = block_number_lock->getNumber();

            /// Set part attributes according to part_number. Prepare an entry for log.

            part->info.min_block = block_number;
            part->info.max_block = block_number;
            part->info.level = 0;

            part->name = part->getNewName(part->info);
        }
        else
        {
            /// This block was already written to some replica. Get the part name for it.
            /// Note: race condition with DROP PARTITION operation is possible. User will get "No node" exception and it is Ok.
            existing_part_name = zookeeper->get(storage.zookeeper_path + "/blocks/" + block_id);

            /// If it exists on our replica, ignore it.
            if (storage.getActiveContainingPart(existing_part_name))
            {
                LOG_INFO(log, "Block with ID {} already exists locally as part {}; ignoring it.", block_id, existing_part_name);
                part->is_duplicate = true;
                last_block_is_duplicate = true;
                ProfileEvents::increment(ProfileEvents::DuplicatedInsertedBlocks);
                return;
            }

            LOG_INFO(log, "Block with ID {} already exists on other replicas as part {}; will write it locally with that name.",
                block_id, existing_part_name);

            /// If it does not exist, we will write a new part with existing name.
            /// Note that it may also appear on filesystem right now in PreCommitted state due to concurrent inserts of the same data.
            /// It will be checked when we will try to rename directory.

            part->name = existing_part_name;
            part->info = MergeTreePartInfo::fromPartName(existing_part_name, storage.format_version);

            /// Don't do subsequent duplicate check.
            block_id_path.clear();
        }

        StorageReplicatedMergeTree::LogEntry log_entry;
        log_entry.type = StorageReplicatedMergeTree::LogEntry::GET_PART;
        log_entry.create_time = time(nullptr);
        log_entry.source_replica = storage.replica_name;
        log_entry.new_part_name = part->name;
        log_entry.quorum = quorum;
        log_entry.block_id = block_id;

        /// Simultaneously add information about the part to all the necessary places in ZooKeeper and remove block_number_lock.

        /// Information about the part.
        Coordination::Requests ops;

        storage.getCommitPartOps(ops, part, block_id_path);

        /// Replication log.
        ops.emplace_back(zkutil::makeCreateRequest(
            storage.zookeeper_path + "/log/log-",
            log_entry.toString(),
            zkutil::CreateMode::PersistentSequential));

        /// Deletes the information that the block number is used for writing.
        if (block_number_lock)
            block_number_lock->getUnlockOps(ops);

        /** If you need a quorum - create a node in which the quorum is monitored.
        * (If such a node already exists, then someone has managed to make another quorum record at the same time,
        *  but for it the quorum has not yet been reached.
        *  You can not do the next quorum record at this time.)
        */
        if (quorum) /// TODO Duplicate blocks.
        {
            ReplicatedMergeTreeQuorumEntry quorum_entry;
            quorum_entry.part_name = part->name;
            quorum_entry.required_number_of_replicas = quorum;
            quorum_entry.replicas.insert(storage.replica_name);

            /** At this point, this node will contain information that the current replica received a part.
                * When other replicas will receive this part (in the usual way, processing the replication log),
                *  they will add themselves to the contents of this node.
                * When it contains information about `quorum` number of replicas, this node is deleted,
                *  which indicates that the quorum has been reached.
                */

            ops.emplace_back(
                zkutil::makeCreateRequest(
                    quorum_info.status_path,
                    quorum_entry.toString(),
                    zkutil::CreateMode::Persistent));

            /// Make sure that during the insertion time, the replica was not reinitialized or disabled (when the server is finished).
            ops.emplace_back(
                zkutil::makeCheckRequest(
                    storage.replica_path + "/is_active",
                    quorum_info.is_active_node_version));

            /// Unfortunately, just checking the above is not enough, because `is_active` node can be deleted and reappear with the same version.
            /// But then the `host` value will change. We will check this.
            /// It's great that these two nodes change in the same transaction (see MergeTreeRestartingThread).
            ops.emplace_back(
                zkutil::makeCheckRequest(
                    storage.replica_path + "/host",
                    quorum_info.host_node_version));
        }

        MergeTreeData::Transaction transaction(storage); /// If you can not add a part to ZK, we'll remove it back from the working set.
        bool renamed = false;
        try
        {
            renamed = storage.renameTempPartAndAdd(part, nullptr, &transaction);
        }
        catch (const Exception & e)
        {
            if (e.code() != ErrorCodes::DUPLICATE_DATA_PART)
                throw;
        }
        if (!renamed)
        {
            if (!existing_part_name.empty())
            {
                LOG_INFO(log, "Part {} is duplicate and it is already written by concurrent request; ignoring it.", block_id, existing_part_name);
                return;
            }
            else
                throw Exception("Part with name {} is already written by concurrent request. It should not happen for non-duplicate data parts because unique names are assigned for them. It's a bug", ErrorCodes::LOGICAL_ERROR);
        }

        Coordination::Responses responses;
        Coordination::Error multi_code = zookeeper->tryMultiNoThrow(ops, responses); /// 1 RTT

        if (multi_code == Coordination::Error::ZOK)
        {
            transaction.commit();
            storage.merge_selecting_task->schedule();

            /// Lock nodes have been already deleted, do not delete them in destructor
            if (block_number_lock)
                block_number_lock->assumeUnlocked();
        }
        else if (multi_code == Coordination::Error::ZCONNECTIONLOSS
            || multi_code == Coordination::Error::ZOPERATIONTIMEOUT)
        {
            /** If the connection is lost, and we do not know if the changes were applied, we can not delete the local part
              *  if the changes were applied, the inserted block appeared in `/blocks/`, and it can not be inserted again.
              */
            transaction.commit();
            storage.enqueuePartForCheck(part->name, MAX_AGE_OF_LOCAL_PART_THAT_WASNT_ADDED_TO_ZOOKEEPER);

            /// We do not know whether or not data has been inserted.
            throw Exception("Unknown status, client must retry. Reason: " + String(Coordination::errorMessage(multi_code)),
                ErrorCodes::UNKNOWN_STATUS_OF_INSERT);
        }
        else if (Coordination::isUserError(multi_code))
        {
            String failed_op_path = zkutil::KeeperMultiException(multi_code, ops, responses).getPathForFirstFailedOp();

            if (multi_code == Coordination::Error::ZNODEEXISTS && deduplicate_block && failed_op_path == block_id_path)
            {
                /// Block with the same id have just appeared in table (or other replica), rollback thee insertion.
                LOG_INFO(log, "Block with ID {} already exists (it was just appeared). Renaming part {} back to {}. Will retry write.",
                    block_id, part->name, temporary_part_name);

                transaction.rollback();

                part->is_duplicate = true;
                part->is_temp = true;
                part->state = MergeTreeDataPartState::Temporary;
                part->renameTo(temporary_part_name);

                continue;
            }
            else if (multi_code == Coordination::Error::ZNODEEXISTS && failed_op_path == quorum_info.status_path)
            {
                transaction.rollback();

                throw Exception("Another quorum insert has been already started", ErrorCodes::UNSATISFIED_QUORUM_FOR_PREVIOUS_WRITE);
            }
            else
            {
                /// NOTE: We could be here if the node with the quorum existed, but was quickly removed.
                transaction.rollback();
                throw Exception("Unexpected logical error while adding block " + toString(block_number) + " with ID '" + block_id + "': "
                                + Coordination::errorMessage(multi_code) + ", path " + failed_op_path,
                                ErrorCodes::UNEXPECTED_ZOOKEEPER_ERROR);
            }
        }
        else if (Coordination::isHardwareError(multi_code))
        {
            transaction.rollback();
            throw Exception("Unrecoverable network error while adding block " + toString(block_number) + " with ID '" + block_id + "': "
                            + Coordination::errorMessage(multi_code), ErrorCodes::UNEXPECTED_ZOOKEEPER_ERROR);
        }
        else
        {
            transaction.rollback();
            throw Exception("Unexpected ZooKeeper error while adding block " + toString(block_number) + " with ID '" + block_id + "': "
                            + Coordination::errorMessage(multi_code), ErrorCodes::UNEXPECTED_ZOOKEEPER_ERROR);
        }

        break;
    }

    if (quorum)
    {
        /// We are waiting for quorum to be satisfied.
        LOG_TRACE(log, "Waiting for quorum");

        String quorum_status_path = storage.zookeeper_path + "/quorum/status";

        try
        {
            while (true)
            {
                zkutil::EventPtr event = std::make_shared<Poco::Event>();

                std::string value;
                /// `get` instead of `exists` so that `watch` does not leak if the node is no longer there.
                if (!zookeeper->tryGet(quorum_status_path, value, nullptr, event))
                    break;

                ReplicatedMergeTreeQuorumEntry quorum_entry(value);

                /// If the node has time to disappear, and then appear again for the next insert.
                if (quorum_entry.part_name != part->name)
                    break;

                if (!event->tryWait(quorum_timeout_ms))
                    throw Exception("Timeout while waiting for quorum", ErrorCodes::TIMEOUT_EXCEEDED);
            }

            /// And what if it is possible that the current replica at this time has ceased to be active and the quorum is marked as failed and deleted?
            String value;
            if (!zookeeper->tryGet(storage.replica_path + "/is_active", value, nullptr)
                || value != quorum_info.is_active_node_value)
                throw Exception("Replica become inactive while waiting for quorum", ErrorCodes::NO_ACTIVE_REPLICAS);
        }
        catch (...)
        {
            /// We do not know whether or not data has been inserted
            /// - whether other replicas have time to download the part and mark the quorum as done.
            throw Exception("Unknown status, client must retry. Reason: " + getCurrentExceptionMessage(false),
                ErrorCodes::UNKNOWN_STATUS_OF_INSERT);
        }

        LOG_TRACE(log, "Quorum satisfied");
    }
}

void ReplicatedMergeTreeBlockOutputStream::writePrefix()
{
    storage.throwInsertIfNeeded();
}


}
