#pragma once

#include "Types.h"
#include <Poco/Util/LayeredConfiguration.h>
#include <unordered_set>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <Common/ProfileEvents.h>
#include <Common/CurrentMetrics.h>
#include <Common/Stopwatch.h>
#include <Common/ZooKeeper/IKeeper.h>
#include <Common/ZooKeeper/KeeperException.h>
#include <Common/ZooKeeper/ZooKeeperConstants.h>
#include <Common/ZooKeeper/ZooKeeperArgs.h>
#include <Common/thread_local_rng.h>
#include <Coordination/KeeperFeatureFlags.h>
#include <unistd.h>
#include <random>


namespace ProfileEvents
{
    extern const Event CannotRemoveEphemeralNode;
}

namespace CurrentMetrics
{
    extern const Metric EphemeralNode;
}

namespace DB
{
    class ZooKeeperLog;

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

}

namespace zkutil
{

/// Preferred size of multi() command (in number of ops)
constexpr size_t MULTI_BATCH_SIZE = 100;

struct ShuffleHost
{
    String host;
    Priority priority;
    UInt64 random = 0;

    void randomize()
    {
        random = thread_local_rng();
    }

    static bool compare(const ShuffleHost & lhs, const ShuffleHost & rhs)
    {
        return std::forward_as_tuple(lhs.priority, lhs.random)
               < std::forward_as_tuple(rhs.priority, rhs.random);
    }
};

struct RemoveException
{
    explicit RemoveException(std::string_view path_ = "", bool remove_subtree_ = true)
        : path(path_)
        , remove_subtree(remove_subtree_)
    {}

    std::string_view path;
    // whether we should keep the child node and its subtree or just the child node
    bool remove_subtree;
};

using GetPriorityForLoadBalancing = DB::GetPriorityForLoadBalancing;

template <typename T>
concept ZooKeeperResponse = std::derived_from<T, Coordination::Response>;

template <ZooKeeperResponse ResponseType, bool try_multi>
struct MultiReadResponses
{
    MultiReadResponses() = default;

    template <typename TResponses>
    explicit MultiReadResponses(TResponses responses_) : responses(std::move(responses_))
    {}

    size_t size() const
    {
        return std::visit(
            [&]<typename TResponses>(const TResponses & resp) -> size_t
            {
                if constexpr (std::same_as<TResponses, std::monostate>)
                    throw DB::Exception(DB::ErrorCodes::LOGICAL_ERROR, "No responses set for MultiRead");
                else
                    return resp.size();
            },
            responses);
    }

    ResponseType & operator[](size_t index)
    {
        return std::visit(
            [&]<typename TResponses>(TResponses & resp) -> ResponseType &
            {
                if constexpr (std::same_as<TResponses, RegularResponses>)
                {
                    return dynamic_cast<ResponseType &>(*resp[index]);
                }
                else if constexpr (std::same_as<TResponses, ResponsesWithFutures>)
                {
                    if constexpr (try_multi)
                    {
                        /// We should not ignore errors except ZNONODE
                        /// for consistency with exists, tryGet and tryGetChildren
                        const auto & error = resp[index].error;
                        if (error != Coordination::Error::ZOK && error != Coordination::Error::ZNONODE)
                            throw KeeperException(error);
                    }
                    return resp[index];
                }
                else
                {
                    throw DB::Exception(DB::ErrorCodes::LOGICAL_ERROR, "No responses set for MultiRead");
                }
            },
            responses);
    }

private:
    using RegularResponses = std::vector<Coordination::ResponsePtr>;
    using FutureResponses = std::vector<std::future<ResponseType>>;

    struct ResponsesWithFutures
    {
        ResponsesWithFutures(FutureResponses future_responses_) : future_responses(std::move(future_responses_))
        {
            cached_responses.resize(future_responses.size());
        }

        FutureResponses future_responses;
        std::vector<std::optional<ResponseType>> cached_responses;

        ResponseType & operator[](size_t index)
        {
            if (cached_responses[index].has_value())
                return *cached_responses[index];

            cached_responses[index] = future_responses[index].get();
            return *cached_responses[index];
        }

        size_t size() const { return future_responses.size(); }
    };

    std::variant<std::monostate, RegularResponses, ResponsesWithFutures> responses;
};

/// ZooKeeper session. The interface is substantially different from the usual libzookeeper API.
///
/// Poco::Event objects are used for watches. The event is set only once on the first
/// watch notification.
/// Callback-based watch interface is also provided.
///
/// Modifying methods do not retry, because it leads to problems of the double-delete type.
///
/// Methods with names not starting at try- raise KeeperException on any error.
class ZooKeeper
{
public:

    using Ptr = std::shared_ptr<ZooKeeper>;
    using ErrorsList = std::initializer_list<Coordination::Error>;

    explicit ZooKeeper(const ZooKeeperArgs & args_, std::shared_ptr<DB::ZooKeeperLog> zk_log_ = nullptr);

    /** Config of the form:
        <zookeeper>
            <node>
                <host>example1</host>
                <port>2181</port>
                <!-- Optional. Enables communication over SSL . -->
                <secure>1</secure>
            </node>
            <node>
                <host>example2</host>
                <port>2181</port>
                <!-- Optional. Enables communication over SSL . -->
                <secure>1</secure>
            </node>
            <session_timeout_ms>30000</session_timeout_ms>
            <operation_timeout_ms>10000</operation_timeout_ms>
            <!-- Optional. Chroot suffix. Should exist. -->
            <root>/path/to/zookeeper/node</root>
            <!-- Optional. Zookeeper digest ACL string. -->
            <identity>user:password</identity>
        </zookeeper>
    */
    ZooKeeper(const Poco::Util::AbstractConfiguration & config, const std::string & config_name, std::shared_ptr<DB::ZooKeeperLog> zk_log_);

    std::vector<ShuffleHost> shuffleHosts() const;

    /// Creates a new session with the same parameters. This method can be used for reconnecting
    /// after the session has expired.
    /// This object remains unchanged, and the new session is returned.
    Ptr startNewSession() const;

    bool configChanged(const Poco::Util::AbstractConfiguration & config, const std::string & config_name) const;

    /// Returns true, if the session has expired.
    bool expired();

    bool isFeatureEnabled(DB::KeeperFeatureFlag feature_flag) const;

    /// Create a znode.
    /// Throw an exception if something went wrong.
    std::string create(const std::string & path, const std::string & data, int32_t mode);

    /// Does not throw in the following cases:
    /// * The parent for the created node does not exist
    /// * The parent is ephemeral.
    /// * The node already exists.
    /// In case of other errors throws an exception.
    Coordination::Error tryCreate(const std::string & path, const std::string & data, int32_t mode, std::string & path_created);
    Coordination::Error tryCreate(const std::string & path, const std::string & data, int32_t mode);

    /// Create a Persistent node.
    /// Does nothing if the node already exists.
    void createIfNotExists(const std::string & path, const std::string & data);

    /// Creates all non-existent ancestors of the given path with empty contents.
    /// Does not create the node itself.
    void createAncestors(const std::string & path);

    void checkExistsAndGetCreateAncestorsOps(const std::string & path, Coordination::Requests & requests);

    /// Remove the node if the version matches. (if version == -1, remove any version).
    void remove(const std::string & path, int32_t version = -1);

    /// Doesn't throw in the following cases:
    /// * The node doesn't exist
    /// * Versions don't match
    /// * The node has children.
    Coordination::Error tryRemove(const std::string & path, int32_t version = -1);

    bool exists(const std::string & path, Coordination::Stat * stat = nullptr, const EventPtr & watch = nullptr);
    bool existsWatch(const std::string & path, Coordination::Stat * stat, Coordination::WatchCallback watch_callback);

    using MultiExistsResponse = MultiReadResponses<Coordination::ExistsResponse, true>;
    template <typename TIter>
    MultiExistsResponse exists(TIter start, TIter end)
    {
        return multiRead<Coordination::ExistsResponse, true>(
            start, end, zkutil::makeExistsRequest, [&](const auto & path) { return asyncExists(path); });
    }

    MultiExistsResponse exists(const std::vector<std::string> & paths)
    {
        return exists(paths.begin(), paths.end());
    }

    std::string get(const std::string & path, Coordination::Stat * stat = nullptr, const EventPtr & watch = nullptr);
    std::string getWatch(const std::string & path, Coordination::Stat * stat, Coordination::WatchCallback watch_callback);

    using MultiGetResponse = MultiReadResponses<Coordination::GetResponse, false>;
    using MultiTryGetResponse = MultiReadResponses<Coordination::GetResponse, true>;

    template <typename TIter>
    MultiGetResponse get(TIter start, TIter end)
    {
        return multiRead<Coordination::GetResponse, false>(
            start, end, zkutil::makeGetRequest, [&](const auto & path) { return asyncGet(path); });
    }

    MultiGetResponse get(const std::vector<std::string> & paths)
    {
        return get(paths.begin(), paths.end());
    }

    /// Doesn't not throw in the following cases:
    /// * The node doesn't exist. Returns false in this case.
    bool tryGet(
        const std::string & path,
        std::string & res,
        Coordination::Stat * stat = nullptr,
        const EventPtr & watch = nullptr,
        Coordination::Error * code = nullptr);

    bool tryGetWatch(
        const std::string & path,
        std::string & res,
        Coordination::Stat * stat,
        Coordination::WatchCallback watch_callback,
        Coordination::Error * code = nullptr);

    template <typename TIter>
    MultiTryGetResponse tryGet(TIter start, TIter end)
    {
        return multiRead<Coordination::GetResponse, true>(
            start, end, zkutil::makeGetRequest, [&](const auto & path) { return asyncTryGet(path); });
    }

    MultiTryGetResponse tryGet(const std::vector<std::string> & paths)
    {
        return tryGet(paths.begin(), paths.end());
    }

    void set(const std::string & path, const std::string & data,
             int32_t version = -1, Coordination::Stat * stat = nullptr);

    /// Creates the node if it doesn't exist. Updates its contents otherwise.
    void createOrUpdate(const std::string & path, const std::string & data, int32_t mode);

    /// Doesn't not throw in the following cases:
    /// * The node doesn't exist.
    /// * Versions do not match.
    Coordination::Error trySet(const std::string & path, const std::string & data,
                   int32_t version = -1, Coordination::Stat * stat = nullptr);

    Strings getChildren(const std::string & path,
                        Coordination::Stat * stat = nullptr,
                        const EventPtr & watch = nullptr,
                        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    Strings getChildrenWatch(const std::string & path,
                             Coordination::Stat * stat,
                             Coordination::WatchCallback watch_callback,
                             Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    using MultiGetChildrenResponse = MultiReadResponses<Coordination::ListResponse, false>;
    using MultiTryGetChildrenResponse = MultiReadResponses<Coordination::ListResponse, true>;

    template <typename TIter>
    MultiGetChildrenResponse
    getChildren(TIter start, TIter end, Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL)
    {
        return multiRead<Coordination::ListResponse, false>(
            start,
            end,
            [list_request_type](const auto & path) { return zkutil::makeListRequest(path, list_request_type); },
            [&](const auto & path) { return asyncGetChildren(path, {}, list_request_type); });
    }

    MultiGetChildrenResponse
    getChildren(const std::vector<std::string> & paths, Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL)
    {
        return getChildren(paths.begin(), paths.end(), list_request_type);
    }

    /// Doesn't not throw in the following cases:
    /// * The node doesn't exist.
    Coordination::Error tryGetChildren(
        const std::string & path,
        Strings & res,
        Coordination::Stat * stat = nullptr,
        const EventPtr & watch = nullptr,
        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    Coordination::Error tryGetChildrenWatch(
        const std::string & path,
        Strings & res,
        Coordination::Stat * stat,
        Coordination::WatchCallback watch_callback,
        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    template <typename TIter>
    MultiTryGetChildrenResponse
    tryGetChildren(TIter start, TIter end, Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL)
    {
        return multiRead<Coordination::ListResponse, true>(
            start,
            end,
            [list_request_type](const auto & path) { return zkutil::makeListRequest(path, list_request_type); },
            [&](const auto & path) { return asyncTryGetChildren(path, list_request_type); });
    }

    MultiTryGetChildrenResponse
    tryGetChildren(const std::vector<std::string> & paths, Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL)
    {
        return tryGetChildren(paths.begin(), paths.end(), list_request_type);
    }

    /// Performs several operations in a transaction.
    /// Throws on every error.
    Coordination::Responses multi(const Coordination::Requests & requests);
    /// Throws only if some operation has returned an "unexpected" error
    /// - an error that would cause the corresponding try- method to throw.
    Coordination::Error tryMulti(const Coordination::Requests & requests, Coordination::Responses & responses);
    /// Throws nothing (even session expired errors)
    Coordination::Error tryMultiNoThrow(const Coordination::Requests & requests, Coordination::Responses & responses);

    std::string sync(const std::string & path);

    Coordination::Error trySync(const std::string & path, std::string & returned_path);

    Int64 getClientID();

    /// Remove the node with the subtree. If someone concurrently adds or removes a node
    /// in the subtree, the result is undefined.
    void removeRecursive(const std::string & path);

    /// Remove the node with the subtree. If someone concurrently removes a node in the subtree,
    /// this will not cause errors.
    /// For instance, you can call this method twice concurrently for the same node and the end
    /// result would be the same as for the single call.
    void tryRemoveRecursive(const std::string & path);

    /// Similar to removeRecursive(...) and tryRemoveRecursive(...), but does not remove path itself.
    /// Node defined as RemoveException will not be deleted.
    void removeChildrenRecursive(const std::string & path, RemoveException keep_child = RemoveException{});
    /// If probably_flat is true, this method will optimistically try to remove children non-recursive
    /// and will fall back to recursive removal if it gets ZNOTEMPTY for some child.
    /// Returns true if no kind of fallback happened.
    /// Node defined as RemoveException will not be deleted.
    bool tryRemoveChildrenRecursive(const std::string & path, bool probably_flat = false, RemoveException keep_child= RemoveException{});

    /// Remove all children nodes (non recursive).
    void removeChildren(const std::string & path);

    using WaitCondition = std::function<bool()>;

    /// Wait for the node to disappear or return immediately if it doesn't exist.
    /// If condition is specified, it is used to return early (when condition returns false)
    /// The function returns true if waited and false if waiting was interrupted by condition.
    bool waitForDisappear(const std::string & path, const WaitCondition & condition = {});

    /// Checks if a the ephemeral node exists. These nodes are removed automatically by ZK when the session ends
    /// If the node exists and its value is equal to fast_delete_if_equal_value it will remove it
    /// If the node exists and its value is different, it will wait for it to disappear. It will throw a LOGICAL_ERROR if the node doesn't
    /// disappear automatically after 3x session_timeout.
    void handleEphemeralNodeExistence(const std::string & path, const std::string & fast_delete_if_equal_value);

    /// Async interface (a small subset of operations is implemented).
    ///
    /// Usage:
    ///
    /// // Non-blocking calls:
    /// auto future1 = zk.asyncGet("/path1");
    /// auto future2 = zk.asyncGet("/path2");
    /// ...
    ///
    /// // These calls can block until the operations are completed:
    /// auto result1 = future1.get();
    /// auto result2 = future2.get();
    ///
    /// NoThrow versions never throw any exception on future.get(), even on SessionExpired error.

    using FutureCreate = std::future<Coordination::CreateResponse>;
    FutureCreate asyncCreate(const std::string & path, const std::string & data, int32_t mode);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureCreate asyncTryCreateNoThrow(const std::string & path, const std::string & data, int32_t mode);

    using FutureGet = std::future<Coordination::GetResponse>;
    FutureGet asyncGet(const std::string & path, Coordination::WatchCallback watch_callback = {});
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureGet asyncTryGetNoThrow(const std::string & path, Coordination::WatchCallback watch_callback = {});

    using FutureExists = std::future<Coordination::ExistsResponse>;
    FutureExists asyncExists(const std::string & path, Coordination::WatchCallback watch_callback = {});
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureExists asyncTryExistsNoThrow(const std::string & path, Coordination::WatchCallback watch_callback = {});

    using FutureGetChildren = std::future<Coordination::ListResponse>;
    FutureGetChildren asyncGetChildren(
        const std::string & path,
        Coordination::WatchCallback watch_callback = {},
        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureGetChildren asyncTryGetChildrenNoThrow(
        const std::string & path,
        Coordination::WatchCallback watch_callback = {},
        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    using FutureSet = std::future<Coordination::SetResponse>;
    FutureSet asyncSet(const std::string & path, const std::string & data, int32_t version = -1);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureSet asyncTrySetNoThrow(const std::string & path, const std::string & data, int32_t version = -1);

    using FutureRemove = std::future<Coordination::RemoveResponse>;
    FutureRemove asyncRemove(const std::string & path, int32_t version = -1);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureRemove asyncTryRemoveNoThrow(const std::string & path, int32_t version = -1);

    using FutureMulti = std::future<Coordination::MultiResponse>;
    FutureMulti asyncMulti(const Coordination::Requests & ops);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureMulti asyncTryMultiNoThrow(const Coordination::Requests & ops);

    using FutureSync = std::future<Coordination::SyncResponse>;
    FutureSync asyncSync(const std::string & path);
    /// Like the previous one but don't throw any exceptions on future.get()
    FutureSync asyncTrySyncNoThrow(const std::string & path);

    /// Very specific methods introduced without following general style. Implements
    /// some custom throw/no throw logic on future.get().
    ///
    /// Doesn't throw in the following cases:
    /// * The node doesn't exist
    /// * The versions do not match
    /// * The node has children
    FutureRemove asyncTryRemove(const std::string & path, int32_t version = -1);

    /// Doesn't throw in the following cases:
    /// * The node doesn't exist
    FutureGet asyncTryGet(const std::string & path);

    /// Doesn't throw in the following cases:
    /// * The node doesn't exist
    FutureGetChildren asyncTryGetChildren(
        const std::string & path,
        Coordination::ListRequestType list_request_type = Coordination::ListRequestType::ALL);

    void finalize(const String & reason);

    void setZooKeeperLog(std::shared_ptr<DB::ZooKeeperLog> zk_log_);

    UInt32 getSessionUptime() const { return static_cast<UInt32>(session_uptime.elapsedSeconds()); }
    bool hasReachedDeadline() const { return impl->hasReachedDeadline(); }

    void setServerCompletelyStarted();

    String getConnectedZooKeeperHost() const { return connected_zk_host; }
    UInt16 getConnectedZooKeeperPort() const { return connected_zk_port; }
    size_t getConnectedZooKeeperIndex() const { return connected_zk_index; }
    UInt64 getConnectedTime() const { return connected_time; }

    const DB::KeeperFeatureFlags * getKeeperFeatureFlags() const { return impl->getKeeperFeatureFlags(); }

private:
    void init(ZooKeeperArgs args_);

    /// The following methods don't any throw exceptions but return error codes.
    Coordination::Error createImpl(const std::string & path, const std::string & data, int32_t mode, std::string & path_created);
    Coordination::Error removeImpl(const std::string & path, int32_t version);
    Coordination::Error getImpl(
        const std::string & path, std::string & res, Coordination::Stat * stat, Coordination::WatchCallback watch_callback);
    Coordination::Error setImpl(const std::string & path, const std::string & data, int32_t version, Coordination::Stat * stat);
    Coordination::Error getChildrenImpl(
        const std::string & path,
        Strings & res,
        Coordination::Stat * stat,
        Coordination::WatchCallback watch_callback,
        Coordination::ListRequestType list_request_type);
    Coordination::Error multiImpl(const Coordination::Requests & requests, Coordination::Responses & responses);
    Coordination::Error existsImpl(const std::string & path, Coordination::Stat * stat_, Coordination::WatchCallback watch_callback);
    Coordination::Error syncImpl(const std::string & path, std::string & returned_path);

    using RequestFactory = std::function<Coordination::RequestPtr(const std::string &)>;
    template <typename TResponse>
    using AsyncFunction = std::function<std::future<TResponse>(const std::string &)>;

    template <typename TResponse, bool try_multi, typename TIter>
    MultiReadResponses<TResponse, try_multi> multiRead(TIter start, TIter end, RequestFactory request_factory, AsyncFunction<TResponse> async_fun)
    {
        if (isFeatureEnabled(DB::KeeperFeatureFlag::MULTI_READ))
        {
            Coordination::Requests requests;
            for (auto it = start; it != end; ++it)
                requests.push_back(request_factory(*it));

            if constexpr (try_multi)
            {
                Coordination::Responses responses;
                tryMulti(requests, responses);
                return MultiReadResponses<TResponse, try_multi>{std::move(responses)};
            }
            else
            {
                auto responses = multi(requests);
                return MultiReadResponses<TResponse, try_multi>{std::move(responses)};
            }
        }

        auto responses_size = std::distance(start, end);
        std::vector<std::future<TResponse>> future_responses;

        if (responses_size == 0)
            return MultiReadResponses<TResponse, try_multi>(std::move(future_responses));

        future_responses.reserve(responses_size);

        for (auto it = start; it != end; ++it)
            future_responses.push_back(async_fun(*it));

        return MultiReadResponses<TResponse, try_multi>{std::move(future_responses)};
    }

    std::unique_ptr<Coordination::IKeeper> impl;

    ZooKeeperArgs args;

    String connected_zk_host;
    UInt16 connected_zk_port;
    size_t connected_zk_index;
    UInt64 connected_time = timeInSeconds(std::chrono::system_clock::now());

    std::mutex mutex;

    Poco::Logger * log = nullptr;
    std::shared_ptr<DB::ZooKeeperLog> zk_log;

    AtomicStopwatch session_uptime;
};


using ZooKeeperPtr = ZooKeeper::Ptr;


/// Creates an ephemeral node in the constructor, removes it in the destructor.
class EphemeralNodeHolder
{
public:
    using Ptr = std::shared_ptr<EphemeralNodeHolder>;

    EphemeralNodeHolder(const std::string & path_, ZooKeeper & zookeeper_, bool create, bool sequential, const std::string & data)
            : path(path_), zookeeper(zookeeper_)
    {
        if (create)
            path = zookeeper.create(path, data, sequential ? CreateMode::EphemeralSequential : CreateMode::Ephemeral);
    }

    std::string getPath() const
    {
        return path;
    }

    static Ptr create(const std::string & path, ZooKeeper & zookeeper, const std::string & data = "")
    {
        return std::make_shared<EphemeralNodeHolder>(path, zookeeper, true, false, data);
    }

    static Ptr createSequential(const std::string & path, ZooKeeper & zookeeper, const std::string & data = "")
    {
        return std::make_shared<EphemeralNodeHolder>(path, zookeeper, true, true, data);
    }

    static Ptr existing(const std::string & path, ZooKeeper & zookeeper)
    {
        return std::make_shared<EphemeralNodeHolder>(path, zookeeper, false, false, "");
    }

    void setAlreadyRemoved()
    {
        need_remove = false;
    }

    ~EphemeralNodeHolder()
    {
        if (!need_remove)
            return;
        try
        {
            zookeeper.tryRemove(path);
        }
        catch (...)
        {
            ProfileEvents::increment(ProfileEvents::CannotRemoveEphemeralNode);
            DB::tryLogCurrentException(__PRETTY_FUNCTION__, "Cannot remove " + path);
        }
    }

private:
    std::string path;
    ZooKeeper & zookeeper;
    CurrentMetrics::Increment metric_increment{CurrentMetrics::EphemeralNode};
    bool need_remove = true;
};

using EphemeralNodeHolderPtr = EphemeralNodeHolder::Ptr;

String normalizeZooKeeperPath(std::string zookeeper_path, bool check_starts_with_slash, Poco::Logger * log = nullptr);

String extractZooKeeperName(const String & path);

String extractZooKeeperPath(const String & path, bool check_starts_with_slash, Poco::Logger * log = nullptr);

String getSequentialNodeName(const String & prefix, UInt64 number);

void validateZooKeeperConfig(const Poco::Util::AbstractConfiguration & config);

bool hasZooKeeperConfig(const Poco::Util::AbstractConfiguration & config);

String getZooKeeperConfigName(const Poco::Util::AbstractConfiguration & config);

template <typename Client>
void addCheckNotExistsRequest(Coordination::Requests & requests, const Client & client, const std::string & path)
{
    if (client.isFeatureEnabled(DB::KeeperFeatureFlag::CHECK_NOT_EXISTS))
    {
        auto request = std::make_shared<Coordination::CheckRequest>();
        request->path = path;
        request->not_exists = true;
        requests.push_back(std::move(request));
        return;
    }

    requests.push_back(makeCreateRequest(path, "", zkutil::CreateMode::Persistent));
    requests.push_back(makeRemoveRequest(path, -1));
}

}
