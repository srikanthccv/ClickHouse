#pragma once

#include <Backups/IRestoreCoordination.h>
#include <Backups/BackupCoordinationHelpers.h>
#include <Common/ZooKeeper/Common.h>


namespace DB
{

/// Stores restore temporary information in Zookeeper, used to perform RESTORE ON CLUSTER.
class RestoreCoordinationDistributed : public IRestoreCoordination
{
public:
    RestoreCoordinationDistributed(const String & zookeeper_path, zkutil::GetZooKeeper get_zookeeper);
    ~RestoreCoordinationDistributed() override;

    /// Starts creating a table in a replicated database. Returns false if there is another host which is already creating this table.
    bool startCreatingTableInReplicatedDB(
        const String & host_id, const String & database_name, const String & database_zk_path, const String & table_name) override;

    /// Sets that either we have been created a table in a replicated database or failed doing that.
    /// In the latter case `error_message` should be set.
    /// Calling this function unblocks other hosts waiting for this table to be created (see waitForCreatingTableInReplicatedDB()).
    void finishCreatingTableInReplicatedDB(
        const String & host_id,
        const String & database_name,
        const String & database_zk_path,
        const String & table_name,
        const String & error_message) override;

    /// Wait for another host to create a table in a replicated database.
    void waitForTableCreatedInReplicatedDB(
        const String & database_name, const String & database_zk_path, const String & table_name, std::chrono::seconds timeout) override;

    /// Sets that a specified host has finished restoring metadata, successfully or with an error.
    /// In the latter case `error_message` should be set.
    void finishRestoringMetadata(const String & host_id, const String & error_message) override;

    /// Waits for all hosts to finish restoring their metadata (i.e. to finish creating databases and tables). Returns false if time is out.
    void waitForAllHostsRestoredMetadata(const Strings & host_ids, std::chrono::seconds timeout) const override;

    /// Sets that this replica is going to restore a partition in a replicated table.
    /// The function returns false if this partition is being already restored by another replica.
    bool startInsertingDataToPartitionInReplicatedTable(
        const String & host_id,
        const StorageID & table_id,
        const String & table_zk_path,
        const String & partition_name) override;

    /// Removes remotely stored information.
    void drop() override;

private:
    void createRootNodes();
    void removeAllNodes();

    class ReplicatedDatabasesMetadataSync;

    const String zookeeper_path;
    const zkutil::GetZooKeeper get_zookeeper;
    std::unique_ptr<ReplicatedDatabasesMetadataSync> replicated_databases_metadata_sync;
    BackupCoordinationDistributedBarrier all_metadata_barrier;
};

}
