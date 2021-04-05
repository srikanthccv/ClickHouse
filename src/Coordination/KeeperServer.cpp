#include <Coordination/KeeperServer.h>
#include <Coordination/LoggerWrapper.h>
#include <Coordination/KeeperStateMachine.h>
#include <Coordination/KeeperStateManager.h>
#include <Coordination/WriteBufferFromNuraftBuffer.h>
#include <Coordination/ReadBufferFromNuraftBuffer.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <chrono>
#include <Common/ZooKeeper/ZooKeeperIO.h>
#include <string>

namespace DB
{

namespace ErrorCodes
{
    extern const int RAFT_ERROR;
}

KeeperServer::KeeperServer(
    int server_id_,
    const CoordinationSettingsPtr & coordination_settings_,
    const Poco::Util::AbstractConfiguration & config,
    ResponsesQueue & responses_queue_,
    SnapshotsQueue & snapshots_queue_)
    : server_id(server_id_)
    , coordination_settings(coordination_settings_)
    , state_machine(nuraft::cs_new<KeeperStateMachine>(
                        responses_queue_, snapshots_queue_,
                        config.getString("keeper_server.snapshot_storage_path", config.getString("path", DBMS_DEFAULT_PATH) + "coordination/snapshots"),
                        coordination_settings))
    , state_manager(nuraft::cs_new<KeeperStateManager>(server_id, "keeper_server", config, coordination_settings))
    , responses_queue(responses_queue_)
{
    if (coordination_settings->quorum_reads)
        LOG_WARNING(&Poco::Logger::get("KeeperServer"), "Quorum reads enabled, Keeper will work slower.");
}

void KeeperServer::startup()
{

    state_machine->init();

    state_manager->loadLogStore(state_machine->last_commit_index() + 1, coordination_settings->reserved_log_items);

    bool single_server = state_manager->getTotalServers() == 1;

    nuraft::raft_params params;
    if (single_server)
    {
        /// Don't make sense in single server mode
        params.heart_beat_interval_ = 0;
        params.election_timeout_lower_bound_ = 0;
        params.election_timeout_upper_bound_ = 0;
    }
    else
    {
        params.heart_beat_interval_ = coordination_settings->heart_beat_interval_ms.totalMilliseconds();
        params.election_timeout_lower_bound_ = coordination_settings->election_timeout_lower_bound_ms.totalMilliseconds();
        params.election_timeout_upper_bound_ = coordination_settings->election_timeout_upper_bound_ms.totalMilliseconds();
    }

    params.reserved_log_items_ = coordination_settings->reserved_log_items;
    params.snapshot_distance_ = coordination_settings->snapshot_distance;
    params.stale_log_gap_ = coordination_settings->stale_log_gap;
    params.fresh_log_gap_ = coordination_settings->fresh_log_gap;
    params.client_req_timeout_ = coordination_settings->operation_timeout_ms.totalMilliseconds();
    params.auto_forwarding_ = coordination_settings->auto_forwarding;
    params.auto_forwarding_req_timeout_ = coordination_settings->operation_timeout_ms.totalMilliseconds() * 2;

    params.return_method_ = nuraft::raft_params::blocking;

    nuraft::asio_service::options asio_opts{};
    nuraft::raft_server::init_options init_options;
    init_options.skip_initial_election_timeout_ = state_manager->shouldStartAsFollower();
    init_options.raft_callback_ = [this] (nuraft::cb_func::Type type, nuraft::cb_func::Param * param)
    {
        return callbackFunc(type, param);
    };

    raft_instance = launcher.init(
        state_machine, state_manager, nuraft::cs_new<LoggerWrapper>("RaftInstance", coordination_settings->raft_logs_level), state_manager->getPort(),
        asio_opts, params, init_options);

    if (!raft_instance)
        throw Exception(ErrorCodes::RAFT_ERROR, "Cannot allocate RAFT instance");
}

void KeeperServer::shutdown()
{
    state_machine->shutdownStorage();
    state_manager->flushLogStore();
    auto timeout = coordination_settings->shutdown_timeout.totalSeconds();
    if (!launcher.shutdown(timeout))
        LOG_WARNING(&Poco::Logger::get("KeeperServer"), "Failed to shutdown RAFT server in {} seconds", timeout);
}

namespace
{

nuraft::ptr<nuraft::buffer> getZooKeeperLogEntry(int64_t session_id, const Coordination::ZooKeeperRequestPtr & request)
{
    DB::WriteBufferFromNuraftBuffer buf;
    DB::writeIntBinary(session_id, buf);
    request->write(buf);
    return buf.getBuffer();
}

}

void KeeperServer::putRequest(const KeeperStorage::RequestForSession & request_for_session)
{
    auto [session_id, request] = request_for_session;
    if (!coordination_settings->quorum_reads && isLeaderAlive() && request->isReadRequest())
    {
        state_machine->processReadRequest(request_for_session);
    }
    else
    {
        std::vector<nuraft::ptr<nuraft::buffer>> entries;
        entries.push_back(getZooKeeperLogEntry(session_id, request));

        std::lock_guard lock(append_entries_mutex);

        auto result = raft_instance->append_entries(entries);
        if (!result->get_accepted())
        {
            KeeperStorage::ResponsesForSessions responses;
            auto response = request->makeResponse();
            response->xid = request->xid;
            response->zxid = 0;
            response->error = Coordination::Error::ZOPERATIONTIMEOUT;
            responses_queue.push(DB::KeeperStorage::ResponseForSession{session_id, response});
        }

        if (result->get_result_code() == nuraft::cmd_result_code::TIMEOUT)
        {
            KeeperStorage::ResponsesForSessions responses;
            auto response = request->makeResponse();
            response->xid = request->xid;
            response->zxid = 0;
            response->error = Coordination::Error::ZOPERATIONTIMEOUT;
            responses_queue.push(DB::KeeperStorage::ResponseForSession{session_id, response});
        }
        else if (result->get_result_code() != nuraft::cmd_result_code::OK)
            throw Exception(ErrorCodes::RAFT_ERROR, "Requests result failed with code {} and message: '{}'", result->get_result_code(), result->get_result_str());
    }
}

int64_t KeeperServer::getSessionID(int64_t session_timeout_ms)
{
    /// Just some sanity check. We don't want to make a lot of clients wait with lock.
    if (active_session_id_requests > 10)
        throw Exception(ErrorCodes::RAFT_ERROR, "Too many concurrent SessionID requests already in flight");

    ++active_session_id_requests;
    SCOPE_EXIT({ --active_session_id_requests; });

    auto entry = nuraft::buffer::alloc(sizeof(int64_t));
    /// Just special session request
    nuraft::buffer_serializer bs(entry);
    bs.put_i64(session_timeout_ms);

    std::lock_guard lock(append_entries_mutex);

    auto result = raft_instance->append_entries({entry});

    if (!result->get_accepted())
        throw Exception(ErrorCodes::RAFT_ERROR, "Cannot send session_id request to RAFT");

    if (result->get_result_code() != nuraft::cmd_result_code::OK)
        throw Exception(ErrorCodes::RAFT_ERROR, "session_id request failed to RAFT");

    auto resp = result->get();
    if (resp == nullptr)
        throw Exception(ErrorCodes::RAFT_ERROR, "Received nullptr as session_id");

    nuraft::buffer_serializer bs_resp(resp);
    return bs_resp.get_i64();
}

bool KeeperServer::isLeader() const
{
    return raft_instance->is_leader();
}

bool KeeperServer::isLeaderAlive() const
{
    return raft_instance->is_leader_alive();
}

nuraft::cb_func::ReturnCode KeeperServer::callbackFunc(nuraft::cb_func::Type type, nuraft::cb_func::Param * /* param */)
{
    size_t last_commited = state_machine->last_commit_index();
    size_t next_index = state_manager->getLogStore()->next_slot();
    bool commited_store = false;
    if (next_index < last_commited || next_index - last_commited <= 1)
        commited_store = true;

    if (initialized_flag)
        return nuraft::cb_func::ReturnCode::Ok;

    auto set_initialized = [this] ()
    {
        std::unique_lock lock(initialized_mutex);
        initialized_flag = true;
        initialized_cv.notify_all();
    };

    switch (type)
    {
        case nuraft::cb_func::BecomeLeader:
        {
            /// We become leader and store is empty or we already committed it
            if (commited_store || initial_batch_committed)
                set_initialized();
            return nuraft::cb_func::ReturnCode::Ok;
        }
        case nuraft::cb_func::BecomeFollower:
        case nuraft::cb_func::GotAppendEntryReqFromLeader:
        {
            if (isLeaderAlive())
            {
                auto leader_index = raft_instance->get_leader_committed_log_idx();
                auto our_index = raft_instance->get_committed_log_idx();
                /// This may happen when we start RAFT cluster from scratch.
                /// Node first became leader, and after that some other node became leader.
                /// BecameFresh for this node will not be called because it was already fresh
                /// when it was leader.
                if (leader_index < our_index + coordination_settings->fresh_log_gap)
                    set_initialized();
            }
            return nuraft::cb_func::ReturnCode::Ok;
        }
        case nuraft::cb_func::BecomeFresh:
        {
            set_initialized(); /// We are fresh follower, ready to serve requests.
            return nuraft::cb_func::ReturnCode::Ok;
        }
        case nuraft::cb_func::InitialBatchCommited:
        {
            if (isLeader()) /// We have committed our log store and we are leader, ready to serve requests.
                set_initialized();
            initial_batch_committed = true;
            return nuraft::cb_func::ReturnCode::Ok;
        }
        default: /// ignore other events
            return nuraft::cb_func::ReturnCode::Ok;
    }
}

void KeeperServer::waitInit()
{
    std::unique_lock lock(initialized_mutex);
    int64_t timeout = coordination_settings->startup_timeout.totalMilliseconds();
    if (!initialized_cv.wait_for(lock, std::chrono::milliseconds(timeout), [&] { return initialized_flag.load(); }))
        throw Exception(ErrorCodes::RAFT_ERROR, "Failed to wait RAFT initialization");
}

std::unordered_set<int64_t> KeeperServer::getDeadSessions()
{
    return state_machine->getDeadSessions();
}

}
