#pragma once

#include <Interpreters/InterserverIOHandler.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/IStorage_fwd.h>
#include <IO/HashingWriteBuffer.h>
#include <IO/copyData.h>
#include <IO/ConnectionTimeouts.h>
#include <IO/ReadWriteBufferFromHTTP.h>


namespace zkutil
{
    class ZooKeeper;
    using ZooKeeperPtr = std::shared_ptr<ZooKeeper>;
}

namespace DB
{

namespace DataPartsExchange
{

/** Service for sending parts from the table *MergeTree.
  */
class Service final : public InterserverIOEndpoint
{
public:
    Service(MergeTreeData & data_)
    : data(data_), log(&Poco::Logger::get(data.getLogName() + " (Replicated PartsService)")) {}

    Service(const Service &) = delete;
    Service & operator=(const Service &) = delete;

    std::string getId(const std::string & node_id) const override;
    void processQuery(const Poco::Net::HTMLForm & params, ReadBuffer & body, WriteBuffer & out, Poco::Net::HTTPServerResponse & response) override;

    void setZooKeeper(const zkutil::ZooKeeperPtr & zookeeper_, const String & zookeeper_path_, const String & replica_name_) override
    {
        zookeeper = zookeeper_;
        zookeeper_path = zookeeper_path_;
        replica_name = replica_name_;
    }

private:
    MergeTreeData::DataPartPtr findPart(const String & name);
    void sendPartFromMemory(const MergeTreeData::DataPartPtr & part, WriteBuffer & out);
    void sendPartFromDisk(const MergeTreeData::DataPartPtr & part, WriteBuffer & out, bool send_default_compression_file);
    void sendPartS3Metadata(const MergeTreeData::DataPartPtr & part, WriteBuffer & out);

private:
    /// StorageReplicatedMergeTree::shutdown() waits for all parts exchange handlers to finish,
    /// so Service will never access dangling reference to storage
    MergeTreeData & data;
    Poco::Logger * log;
    zkutil::ZooKeeperPtr zookeeper;
    String zookeeper_path;
    String replica_name;
};

/** Client for getting the parts from the table *MergeTree.
  */
class Fetcher final
{
public:
    Fetcher(MergeTreeData & data_) : data(data_), log(&Poco::Logger::get("Fetcher")) {}

    Fetcher(const Fetcher &) = delete;
    Fetcher & operator=(const Fetcher &) = delete;

    /// Downloads a part to tmp_directory. If to_detached - downloads to the `detached` directory.
    MergeTreeData::MutableDataPartPtr fetchPart(
        const StorageMetadataPtr & metadata_snapshot,
        const String & part_name,
        const String & replica_path,
        const String & host,
        int port,
        const ConnectionTimeouts & timeouts,
        const String & user,
        const String & password,
        const String & interserver_scheme,
        bool to_detached = false,
        const String & tmp_prefix_ = "",
        bool try_use_s3_copy = true);

    /// You need to stop the data transfer.
    ActionBlocker blocker;

    void setZooKeeper(const zkutil::ZooKeeperPtr & zookeeper_, const String & zookeeper_path_, const String & replica_name_)
    {
        zookeeper = zookeeper_;
        zookeeper_path = zookeeper_path_;
        replica_name = replica_name_;
    }

private:
    MergeTreeData::MutableDataPartPtr downloadPartToDisk(
            const String & part_name,
            const String & replica_path,
            bool to_detached,
            const String & tmp_prefix_,
            bool sync,
            const ReservationPtr reservation,
            PooledReadWriteBufferFromHTTP & in);

    MergeTreeData::MutableDataPartPtr downloadPartToMemory(
            const String & part_name,
            const StorageMetadataPtr & metadata_snapshot,
            ReservationPtr reservation,
            PooledReadWriteBufferFromHTTP & in);

    MergeTreeData::MutableDataPartPtr downloadPartToS3(
            const String & part_name,
            const String & replica_path,
            bool to_detached,
            const String & tmp_prefix_,
            const Disks & disks_s3,
            PooledReadWriteBufferFromHTTP & in);

    MergeTreeData & data;
    Poco::Logger * log;
    zkutil::ZooKeeperPtr zookeeper;
    String zookeeper_path;
    String replica_name;
};

}

}
