#pragma once

#include <Common/TimerDescriptor.h>
#include <Client/ConnectionPoolWithFailover.h>
#include <Core/Settings.h>
#include <Common/Epoll.h>
#include <unordered_map>

namespace DB
{

/// Class for establishing hedged connections with replicas.
/// It works with multiple replicas simultaneously without blocking
/// (in current implementation only with 2 replicas) by using epoll.
class GetHedgedConnections
{
public:
    using ShuffledPool = ConnectionPoolWithFailover::Base::ShuffledPool;

    enum State
    {
        EMPTY = 0,
        READY = 1,
        NOT_READY = 2,
        CANNOT_CHOOSE = 3,
    };

    struct ReplicaState
    {
        Connection * connection = nullptr;
        State state = State::EMPTY;
        int index = -1;
        int fd = -1;
        std::unordered_map<int, std::unique_ptr<TimerDescriptor>> active_timeouts;

        void reset()
        {
            connection = nullptr;
            state = State::EMPTY;
            index = -1;
            fd = -1;
            active_timeouts.clear();
        }

        bool isReady() const { return state == State::READY; };
        bool isNotReady() const { return state == State::NOT_READY; };
        bool isEmpty() const { return state == State::EMPTY; };
        bool isCannotChoose() const { return state == State::CANNOT_CHOOSE; };
    };

    using ReplicaStatePtr = ReplicaState *;

    struct Replicas
    {
        ReplicaStatePtr first_replica;
        ReplicaStatePtr second_replica;
    };

    GetHedgedConnections(const ConnectionPoolWithFailoverPtr & pool_,
                        const Settings * settings_,
                        const ConnectionTimeouts & timeouts_,
                        std::shared_ptr<QualifiedTableName> table_to_check_ = nullptr);

    /// Establish connection with replicas. Return replicas as soon as connection with one of them is finished.
    /// The first replica is always has state FINISHED and ready for sending query, the second replica
    /// may have any state. To continue working with second replica call chooseSecondReplica().
    Replicas getConnections();

    /// Continue choosing second replica, this function is not blocking. Second replica will be ready
    /// for sending query when it has state FINISHED.
    void chooseSecondReplica();

    void stopChoosingSecondReplica();

    void swapReplicas() { std::swap(first_replica, second_replica); }

    /// Move ready replica to the first place.
    void swapReplicasIfNeeded();

    /// Check if the file descriptor is belong to one of replicas. If yes, return this replica, if no, return nullptr.
    ReplicaStatePtr isEventReplica(int event_fd);

    /// Check if the file descriptor is belong to timeout to any replica.
    /// If yes, return corresponding TimerDescriptor and set timeout owner to replica,
    /// if no, return nullptr.
    TimerDescriptorPtr isEventTimeout(int event_fd, ReplicaStatePtr & replica);

    /// Get file rescriptor that ready for reading.
    int getReadyFileDescriptor(Epoll & epoll_, AsyncCallback async_callback = {});

    int getFileDescriptor() const { return epoll.getFileDescriptor(); }

    const ConnectionTimeouts & getConnectionTimeouts() const { return timeouts; }

    ~GetHedgedConnections();

private:

    enum Action
    {
        FINISH = 0,
        PROCESS_EPOLL_EVENTS = 1,
        TRY_NEXT_REPLICA = 2,
    };

    Action startTryGetConnection(int index, ReplicaStatePtr replica);

    Action processTryGetConnectionStage(ReplicaStatePtr replica, bool remove_from_epoll = false);

    int getNextIndex(int cur_index = -1);

    void addTimeouts(ReplicaStatePtr replica);

    void processFailedConnection(ReplicaStatePtr replica);

    void processReceiveTimeout(ReplicaStatePtr replica);

    bool processReplicaEvent(ReplicaStatePtr replica, bool non_blocking);

    void processTimeoutEvent(ReplicaStatePtr & replica, TimerDescriptorPtr timeout_descriptor);

    ReplicaStatePtr processEpollEvents(bool non_blocking = false);

    void setBestUsableReplica(ReplicaState & replica, int skip_index = -1);

    const ConnectionPoolWithFailoverPtr pool;
    const Settings * settings;
    const ConnectionTimeouts timeouts;
    std::shared_ptr<QualifiedTableName> table_to_check;
    std::vector<TryGetConnection> try_get_connections;
    std::vector<ShuffledPool> shuffled_pools;
    ReplicaState first_replica;
    ReplicaState second_replica;
    bool fallback_to_stale_replicas;
    Epoll epoll;
    Poco::Logger * log;
    std::string fail_messages;
    size_t entries_count;
    size_t usable_count;
    size_t failed_pools_count;
    size_t max_tries;

};

/// Add timeout with particular type to replica and add it to epoll.
void addTimeoutToReplica(int type, GetHedgedConnections::ReplicaStatePtr replica, Epoll & epoll, const ConnectionTimeouts & timeouts);

/// Remove timeout with particular type from replica and epoll.
void removeTimeoutFromReplica(int type, GetHedgedConnections::ReplicaStatePtr replica, Epoll & epoll);

/// Remove all timeouts from replica and epoll.
void removeTimeoutsFromReplica(GetHedgedConnections::ReplicaStatePtr replica, Epoll & epoll);

}
