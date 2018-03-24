#pragma once
#include <common/Types.h>
#include <future>
#include <memory>
#include <vector>
#include <Common/ZooKeeper/ZooKeeperImpl.h>
#include <Poco/Event.h>


namespace zkutil
{

using Stat = ZooKeeperImpl::ZooKeeper::Stat;
using Strings = std::vector<std::string>;


namespace CreateMode
{
    extern const int Persistent;
    extern const int Ephemeral;
    extern const int EphemeralSequential;
    extern const int PersistentSequential;
}

using EventPtr = std::shared_ptr<Poco::Event>;

/// Callback to call when the watch fires.
/// Because callbacks are called in the single "completion" thread internal to libzookeeper,
/// they must execute as quickly as possible (preferably just set some notification).
using WatchCallback = ZooKeeperImpl::ZooKeeper::WatchCallback;

using RequestPtr = ZooKeeperImpl::ZooKeeper::RequestPtr;
using ResponsePtr = ZooKeeperImpl::ZooKeeper::ResponsePtr;

using Requests = ZooKeeperImpl::ZooKeeper::Requests;
using Responses = ZooKeeperImpl::ZooKeeper::Responses;

using CreateRequest = ZooKeeperImpl::ZooKeeper::CreateRequest;
using RemoveRequest = ZooKeeperImpl::ZooKeeper::RemoveRequest;
using ExistsRequest = ZooKeeperImpl::ZooKeeper::ExistsRequest;
using GetRequest = ZooKeeperImpl::ZooKeeper::GetRequest;
using SetRequest = ZooKeeperImpl::ZooKeeper::SetRequest;
using ListRequest = ZooKeeperImpl::ZooKeeper::ListRequest;
using CheckRequest = ZooKeeperImpl::ZooKeeper::CheckRequest;

using CreateResponse = ZooKeeperImpl::ZooKeeper::CreateResponse;
using RemoveResponse = ZooKeeperImpl::ZooKeeper::RemoveResponse;
using ExistsResponse = ZooKeeperImpl::ZooKeeper::ExistsResponse;
using GetResponse = ZooKeeperImpl::ZooKeeper::GetResponse;
using SetResponse = ZooKeeperImpl::ZooKeeper::SetResponse;
using ListResponse = ZooKeeperImpl::ZooKeeper::ListResponse;
using CheckResponse = ZooKeeperImpl::ZooKeeper::CheckResponse;

RequestPtr makeCreateRequest(const std::string & path, const std::string & data, int create_mode);
RequestPtr makeRemoveRequest(const std::string & path, int version);
RequestPtr makeSetRequest(const std::string & path, const std::string & data);
RequestPtr makeCheckRequest(const std::string & path, int version);

}
