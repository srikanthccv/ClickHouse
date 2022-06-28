#include <Coordination/KeeperDispatcher.h>
#include <Common/setThreadName.h>
#include <Common/ZooKeeper/KeeperException.h>
#include <future>
#include <chrono>
#include <Poco/Path.h>
#include <Common/hex.h>
#include <filesystem>
#include <limits>
#include <Common/checkStackSize.h>

namespace fs = std::filesystem;

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int TIMEOUT_EXCEEDED;
    extern const int SYSTEM_ERROR;
}


KeeperDispatcher::KeeperDispatcher()
    : responses_queue(std::numeric_limits<size_t>::max())
    , read_requests_queue(std::numeric_limits<size_t>::max())
    , configuration_and_settings(std::make_shared<KeeperConfigurationAndSettings>())
    , log(&Poco::Logger::get("KeeperDispatcher"))
{
}


void KeeperDispatcher::requestThread()
{
    setThreadName("KeeperReqT");

    /// Result of requests batch from previous iteration
    RaftAppendResult prev_result = nullptr;

    const auto needs_quorum = [](const auto & coordination_settings, const auto & request)
    {
        return coordination_settings->quorum_reads || !request.request->isReadRequest();
    };

    KeeperStorage::RequestsForSessions write_requests;
    KeeperStorage::RequestsForSessions read_requests;

    while (!shutdown_called)
    {
        KeeperStorage::RequestForSession request;

        auto coordination_settings = configuration_and_settings->coordination_settings;
        uint64_t max_wait = coordination_settings->operation_timeout_ms.totalMilliseconds();
        uint64_t max_batch_size = coordination_settings->max_requests_batch_size;

        try
        {
            KeeperStorage::RequestsForSessions * current_batch{nullptr};
            bool is_current_read = false;

            if (requests_queue->tryPop(request, max_wait))
            {
                if (shutdown_called)
                    break;

                if (needs_quorum(coordination_settings, request))
                {
                    write_requests.emplace_back(request);

                    if (!current_batch)
                    {
                        current_batch = &write_requests;
                        is_current_read = false;
                    }
                }
                else
                {
                    read_requests.emplace_back(request);
                    if (!current_batch)
                    {
                        current_batch = &read_requests;
                        is_current_read = true;
                    }
                }

                /// Waiting until previous append will be successful, or batch is big enough
                /// has_result == false && get_result_code == OK means that our request still not processed.
                /// Sometimes NuRaft set errorcode without setting result, so we check both here.
                while (true)
                {
                    if (write_requests.size() > max_batch_size)
                    {
                        current_batch = &write_requests;
                        is_current_read = false;
                        break;
                    }

                    if (read_requests.size() > max_batch_size)
                    {
                        current_batch = &read_requests;
                        is_current_read = true;
                        break;
                    }

                    /// Trying to get batch requests as fast as possible
                    if (requests_queue->tryPop(request, 1))
                    {
                        if (needs_quorum(coordination_settings, request))
                            write_requests.emplace_back(request);
                        else
                            read_requests.emplace_back(request);
                    }
                    else if (is_current_read || !prev_result || prev_result->has_result() || prev_result->get_result_code() != nuraft::cmd_result_code::OK)
                        break;

                    if (shutdown_called)
                        break;
                }

                if (shutdown_called)
                    break;

                /// Forcefully process all previous pending requests
                if (!is_current_read && prev_result)
                    forceWaitAndProcessResult(prev_result);

                if (current_batch)
                {
                    LOG_INFO(&Poco::Logger::get("BATCHING"), "Processing {} batch of {}", is_current_read, current_batch->size());
                    if (!is_current_read)
                    {
                        prev_result = server->putRequestBatch(*current_batch);
                        prev_result->when_ready([&, requests_for_sessions = std::move(*current_batch)](nuraft::cmd_result<nuraft::ptr<nuraft::buffer>> & result, nuraft::ptr<std::exception> &) mutable
                        {
                            if (!result.get_accepted() || result.get_result_code() == nuraft::cmd_result_code::TIMEOUT)
                                addErrorResponses(requests_for_sessions, Coordination::Error::ZOPERATIONTIMEOUT);
                            else if (result.get_result_code() != nuraft::cmd_result_code::OK)
                                addErrorResponses(requests_for_sessions, Coordination::Error::ZCONNECTIONLOSS);
                        });

                        current_batch->clear();

                        if (!read_requests.empty())
                        {
                            current_batch = &read_requests;
                            is_current_read = true;
                        }
                        else
                            current_batch = nullptr;
                    }
                    else
                    {
                        server->getLeaderInfo()->when_ready([&, requests_for_sessions = std::move(*current_batch)](nuraft::cmd_result<nuraft::ptr<nuraft::buffer>> & result, nuraft::ptr<std::exception> &) mutable
                        {
                            if (!result.get_accepted() || result.get_result_code() == nuraft::cmd_result_code::TIMEOUT)
                                addErrorResponses(requests_for_sessions, Coordination::Error::ZOPERATIONTIMEOUT);
                            else if (result.get_result_code() != nuraft::cmd_result_code::OK)
                                addErrorResponses(requests_for_sessions, Coordination::Error::ZCONNECTIONLOSS);

                            auto & leader_info_ctx = result.get();

                            KeeperServer::NodeInfo leader_info;
                            leader_info.term = leader_info_ctx->get_ulong();
                            leader_info.last_committed_index = leader_info_ctx->get_ulong();

                            std::lock_guard lock(leader_waiter_mutex);
                            auto node_info = server->getNodeInfo();

                            if (node_info.term < leader_info.term || node_info.last_committed_index < leader_info.last_committed_index)
                            {
                                auto & leader_waiter = leader_waiters[leader_info];
                                leader_waiter.insert(leader_waiter.end(), requests_for_sessions.begin(), requests_for_sessions.end());
                                LOG_INFO(log, "waiting for {}, idx {}", leader_info.term, leader_info.last_committed_index);
                            }
                            else if (!read_requests_queue.push(std::move(requests_for_sessions)))
                                throw Exception(ErrorCodes::SYSTEM_ERROR, "Cannot push read requests to queue");
                        });

                        current_batch->clear();

                        if (!write_requests.empty())
                        {
                            current_batch = &write_requests;
                            is_current_read = false;
                        }
                        else
                            current_batch = nullptr;
                    }
                }
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}

void KeeperDispatcher::responseThread()
{
    setThreadName("KeeperRspT");
    while (!shutdown_called)
    {
        KeeperStorage::ResponseForSession response_for_session;

        uint64_t max_wait = configuration_and_settings->coordination_settings->operation_timeout_ms.totalMilliseconds();

        if (responses_queue.tryPop(response_for_session, max_wait))
        {
            if (shutdown_called)
                break;

            try
            {
                 setResponse(response_for_session.session_id, response_for_session.response);
            }
            catch (...)
            {
                tryLogCurrentException(__PRETTY_FUNCTION__);
            }
        }
    }
}

void KeeperDispatcher::snapshotThread()
{
    setThreadName("KeeperSnpT");
    while (!shutdown_called)
    {
        CreateSnapshotTask task;
        if (!snapshots_queue.pop(task))
            break;

        if (shutdown_called)
            break;

        try
        {
            task.create_snapshot(std::move(task.snapshot));
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}

void KeeperDispatcher::readRequestThread()
{
    setThreadName("KeeperReadT");
    while (!shutdown_called)
    {
        KeeperStorage::RequestsForSessions requests;
        if (!read_requests_queue.pop(requests))
            break;

        if (shutdown_called)
            break;

        try
        {
            for (const auto & request_info : requests)
            {
                if (server->isLeaderAlive())
                    server->putLocalReadRequest(request_info);
                else
                    addErrorResponses({request_info}, Coordination::Error::ZCONNECTIONLOSS);
            }

            finalizeRequests(requests);
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}

void KeeperDispatcher::setResponse(int64_t session_id, const Coordination::ZooKeeperResponsePtr & response)
{
    std::lock_guard lock(session_to_response_callback_mutex);

    /// Special new session response.
    if (response->xid != Coordination::WATCH_XID && response->getOpNum() == Coordination::OpNum::SessionID)
    {
        const Coordination::ZooKeeperSessionIDResponse & session_id_resp = dynamic_cast<const Coordination::ZooKeeperSessionIDResponse &>(*response);

        /// Nobody waits for this session id
        if (session_id_resp.server_id != server->getServerID() || !new_session_id_response_callback.contains(session_id_resp.internal_id))
            return;

        auto callback = new_session_id_response_callback[session_id_resp.internal_id];
        callback(response);
        new_session_id_response_callback.erase(session_id_resp.internal_id);
    }
    else /// Normal response, just write to client
    {
        auto session_response_callback = session_to_response_callback.find(session_id);

        /// Session was disconnected, just skip this response
        if (session_response_callback == session_to_response_callback.end())
        {
            LOG_TEST(log, "Cannot write response xid={}, op={}, session {} disconnected", response->xid, response->getOpNum(), session_id);
            return;
        }

        session_response_callback->second(response);

        /// Session closed, no more writes
        if (response->xid != Coordination::WATCH_XID && response->getOpNum() == Coordination::OpNum::Close)
        {
            session_to_response_callback.erase(session_response_callback);
        }
    }
}

bool KeeperDispatcher::putRequest(const Coordination::ZooKeeperRequestPtr & request, int64_t session_id)
{
    {
        /// If session was already disconnected than we will ignore requests
        std::lock_guard lock(session_to_response_callback_mutex);
        if (!session_to_response_callback.contains(session_id))
            return false;
    }

    LOG_INFO(&Poco::Logger::get("BATCHING"), "Got request {}", request->getOpNum());
    KeeperStorage::RequestForSession request_info;
    request_info.request = request;
    using namespace std::chrono;
    request_info.time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    request_info.session_id = session_id;

    {
        std::lock_guard lock{unprocessed_request_mutex};
        auto unprocessed_requests_it = unprocessed_requests_for_session.find(session_id);
        if (unprocessed_requests_it == unprocessed_requests_for_session.end())
        {
            auto & unprocessed_requests = unprocessed_requests_for_session[session_id];
            unprocessed_requests.unprocessed_num = 1;
            unprocessed_requests.is_read = request->isReadRequest();
        }
        else
        {
            auto & unprocessed_requests = unprocessed_requests_it->second;

            if (!unprocessed_requests.request_queue.empty() || unprocessed_requests.is_read != request->isReadRequest())
            {
                unprocessed_requests.request_queue.push_back(std::move(request_info));
                return true;
            }
            else
            {
                ++unprocessed_requests.unprocessed_num;
            }
        }
    }

    std::lock_guard lock(push_request_mutex);

    if (shutdown_called)
        return false;

    /// Put close requests without timeouts
    if (request->getOpNum() == Coordination::OpNum::Close)
    {
        if (!requests_queue->push(std::move(request_info)))
            throw Exception("Cannot push request to queue", ErrorCodes::SYSTEM_ERROR);
    }
    else if (!requests_queue->tryPush(std::move(request_info), configuration_and_settings->coordination_settings->operation_timeout_ms.totalMilliseconds()))
    {
        throw Exception("Cannot push request to queue within operation timeout", ErrorCodes::TIMEOUT_EXCEEDED);
    }
    return true;
}

void KeeperDispatcher::initialize(const Poco::Util::AbstractConfiguration & config, bool standalone_keeper, bool start_async)
{
    LOG_DEBUG(log, "Initializing storage dispatcher");

    configuration_and_settings = KeeperConfigurationAndSettings::loadFromConfig(config, standalone_keeper);
    requests_queue = std::make_unique<RequestsQueue>(configuration_and_settings->coordination_settings->max_requests_batch_size);

    request_thread = ThreadFromGlobalPool([this] { requestThread(); });
    responses_thread = ThreadFromGlobalPool([this] { responseThread(); });
    snapshot_thread = ThreadFromGlobalPool([this] { snapshotThread(); });
    read_request_thread = ThreadFromGlobalPool([this] { readRequestThread(); });

    server = std::make_unique<KeeperServer>(configuration_and_settings, config, responses_queue, snapshots_queue, [this](const KeeperStorage::RequestForSession & request_for_session, uint64_t log_term, uint64_t log_idx) { onRequestCommit(request_for_session, log_term, log_idx); });

    try
    {
        LOG_DEBUG(log, "Waiting server to initialize");
        server->startup(config, configuration_and_settings->enable_ipv6);
        LOG_DEBUG(log, "Server initialized, waiting for quorum");

        if (!start_async)
        {
            server->waitInit();
            LOG_DEBUG(log, "Quorum initialized");
        }
        else
        {
            LOG_INFO(log, "Starting Keeper asynchronously, server will accept connections to Keeper when it will be ready");
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
        throw;
    }

    /// Start it after keeper server start
    session_cleaner_thread = ThreadFromGlobalPool([this] { sessionCleanerTask(); });
    update_configuration_thread = ThreadFromGlobalPool([this] { updateConfigurationThread(); });
    updateConfiguration(config);

    LOG_DEBUG(log, "Dispatcher initialized");
}

void KeeperDispatcher::shutdown()
{
    try
    {
        {
            std::lock_guard lock(push_request_mutex);

            if (shutdown_called)
                return;

            LOG_DEBUG(log, "Shutting down storage dispatcher");
            shutdown_called = true;

            if (session_cleaner_thread.joinable())
                session_cleaner_thread.join();

            if (requests_queue)
            {
                requests_queue->finish();

                if (request_thread.joinable())
                    request_thread.join();
            }

            responses_queue.finish();
            if (responses_thread.joinable())
                responses_thread.join();

            snapshots_queue.finish();
            if (snapshot_thread.joinable())
                snapshot_thread.join();

            read_requests_queue.finish();
            if (read_request_thread.joinable())
                read_request_thread.join();

            update_configuration_queue.finish();
            if (update_configuration_thread.joinable())
                update_configuration_thread.join();
        }

        if (server)
            server->shutdown();

        KeeperStorage::RequestForSession request_for_session;

        /// Set session expired for all pending requests
        while (requests_queue && requests_queue->tryPop(request_for_session))
        {
            auto response = request_for_session.request->makeResponse();
            response->error = Coordination::Error::ZSESSIONEXPIRED;
            setResponse(request_for_session.session_id, response);
        }

        /// Clear all registered sessions
        std::lock_guard lock(session_to_response_callback_mutex);
        session_to_response_callback.clear();
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }

    LOG_DEBUG(log, "Dispatcher shut down");
}

void KeeperDispatcher::forceRecovery()
{
    server->forceRecovery();
}

KeeperDispatcher::~KeeperDispatcher()
{
    shutdown();
}

void KeeperDispatcher::registerSession(int64_t session_id, ZooKeeperResponseCallback callback)
{
    std::lock_guard lock(session_to_response_callback_mutex);
    if (!session_to_response_callback.try_emplace(session_id, callback).second)
        throw Exception(DB::ErrorCodes::LOGICAL_ERROR, "Session with id {} already registered in dispatcher", session_id);
}

void KeeperDispatcher::sessionCleanerTask()
{
    while (true)
    {
        if (shutdown_called)
            return;

        try
        {
            /// Only leader node must check dead sessions
            if (server->checkInit() && isLeader())
            {
                auto dead_sessions = server->getDeadSessions();

                for (int64_t dead_session : dead_sessions)
                {
                    LOG_INFO(log, "Found dead session {}, will try to close it", dead_session);

                    /// Close session == send close request to raft server
                    Coordination::ZooKeeperRequestPtr request = Coordination::ZooKeeperRequestFactory::instance().get(Coordination::OpNum::Close);
                    request->xid = Coordination::CLOSE_XID;
                    KeeperStorage::RequestForSession request_info;
                    request_info.request = request;
                    using namespace std::chrono;
                    request_info.time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                    request_info.session_id = dead_session;
                    {
                        std::lock_guard lock(push_request_mutex);
                        if (!requests_queue->push(std::move(request_info)))
                            LOG_INFO(log, "Cannot push close request to queue while cleaning outdated sessions");
                    }

                    /// Remove session from registered sessions
                    finishSession(dead_session);
                    LOG_INFO(log, "Dead session close request pushed");
                }
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }

        auto time_to_sleep = configuration_and_settings->coordination_settings->dead_session_check_period_ms.totalMilliseconds();
        std::this_thread::sleep_for(std::chrono::milliseconds(time_to_sleep));
    }
}

void KeeperDispatcher::finishSession(int64_t session_id)
{
    std::lock_guard lock(session_to_response_callback_mutex);
    auto session_it = session_to_response_callback.find(session_id);
    if (session_it != session_to_response_callback.end())
        session_to_response_callback.erase(session_it);
}

void KeeperDispatcher::addErrorResponses(const KeeperStorage::RequestsForSessions & requests_for_sessions, Coordination::Error error)
{
    for (const auto & request_for_session : requests_for_sessions)
    {
        KeeperStorage::ResponsesForSessions responses;
        auto response = request_for_session.request->makeResponse();
        response->xid = request_for_session.request->xid;
        response->zxid = 0;
        response->error = error;
        if (!responses_queue.push(DB::KeeperStorage::ResponseForSession{request_for_session.session_id, response}))
            throw Exception(ErrorCodes::SYSTEM_ERROR,
                "Could not push error response xid {} zxid {} error message {} to responses queue",
                response->xid,
                response->zxid,
                errorMessage(error));
    }
}

void KeeperDispatcher::forceWaitAndProcessResult(RaftAppendResult & result)
{
    LOG_INFO(&Poco::Logger::get("TESTER"), "Waiting for result");
    if (!result->has_result())
        result->get();

    LOG_INFO(&Poco::Logger::get("TESTER"), "Got result");
    result = nullptr;
}

int64_t KeeperDispatcher::getSessionID(int64_t session_timeout_ms)
{
    /// New session id allocation is a special request, because we cannot process it in normal
    /// way: get request -> put to raft -> set response for registered callback.
    KeeperStorage::RequestForSession request_info;
    std::shared_ptr<Coordination::ZooKeeperSessionIDRequest> request = std::make_shared<Coordination::ZooKeeperSessionIDRequest>();
    /// Internal session id. It's a temporary number which is unique for each client on this server
    /// but can be same on different servers.
    request->internal_id = internal_session_id_counter.fetch_add(1);
    request->session_timeout_ms = session_timeout_ms;
    request->server_id = server->getServerID();

    request_info.request = request;
    using namespace std::chrono;
    request_info.time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    request_info.session_id = -1;

    auto promise = std::make_shared<std::promise<int64_t>>();
    auto future = promise->get_future();

    {
        std::lock_guard lock(session_to_response_callback_mutex);
        new_session_id_response_callback[request->internal_id] = [promise, internal_id = request->internal_id] (const Coordination::ZooKeeperResponsePtr & response)
        {
            if (response->getOpNum() != Coordination::OpNum::SessionID)
                promise->set_exception(std::make_exception_ptr(Exception(ErrorCodes::LOGICAL_ERROR,
                            "Incorrect response of type {} instead of SessionID response", Coordination::toString(response->getOpNum()))));

            auto session_id_response = dynamic_cast<const Coordination::ZooKeeperSessionIDResponse &>(*response);
            if (session_id_response.internal_id != internal_id)
            {
                promise->set_exception(std::make_exception_ptr(Exception(ErrorCodes::LOGICAL_ERROR,
                            "Incorrect response with internal id {} instead of {}", session_id_response.internal_id, internal_id)));
            }

            if (response->error != Coordination::Error::ZOK)
                promise->set_exception(std::make_exception_ptr(zkutil::KeeperException("SessionID request failed with error", response->error)));

            promise->set_value(session_id_response.session_id);
        };
    }

    /// Push new session request to queue
    {
        std::lock_guard lock(push_request_mutex);
        if (!requests_queue->tryPush(std::move(request_info), session_timeout_ms))
            throw Exception("Cannot push session id request to queue within session timeout", ErrorCodes::TIMEOUT_EXCEEDED);
    }

    if (future.wait_for(std::chrono::milliseconds(session_timeout_ms)) != std::future_status::ready)
        throw Exception("Cannot receive session id within session timeout", ErrorCodes::TIMEOUT_EXCEEDED);

    /// Forcefully wait for request execution because we cannot process any other
    /// requests for this client until it get new session id.
    return future.get();
}


void KeeperDispatcher::updateConfigurationThread()
{
    while (true)
    {
        if (shutdown_called)
            return;

        try
        {
            using namespace std::chrono_literals;
            if (!server->checkInit())
            {
                LOG_INFO(log, "Server still not initialized, will not apply configuration until initialization finished");
                std::this_thread::sleep_for(5000ms);
                continue;
            }

            if (server->isRecovering())
            {
                LOG_INFO(log, "Server is recovering, will not apply configuration until recovery is finished");
                std::this_thread::sleep_for(5000ms);
                continue;
            }

            ConfigUpdateAction action;
            if (!update_configuration_queue.pop(action))
                break;


            /// We must wait this update from leader or apply it ourself (if we are leader)
            bool done = false;
            while (!done)
            {
                if (server->isRecovering())
                    break;

                if (shutdown_called)
                    return;

                if (isLeader())
                {
                    server->applyConfigurationUpdate(action);
                    done = true;
                }
                else
                {
                    done = server->waitConfigurationUpdate(action);
                    if (!done)
                        LOG_INFO(log, "Cannot wait for configuration update, maybe we become leader, or maybe update is invalid, will try to wait one more time");
                }
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}

void KeeperDispatcher::finalizeRequests(const KeeperStorage::RequestsForSessions & requests_for_sessions)
{
    std::unordered_map<int64_t, size_t> counts_for_session;

    for (const auto & request_for_session : requests_for_sessions)
    {
        ++counts_for_session[request_for_session.session_id];
    }

    std::lock_guard lock{unprocessed_request_mutex};
    for (const auto [session_id, count] : counts_for_session)
    {
        auto unprocessed_requests_it = unprocessed_requests_for_session.find(session_id);
        if (unprocessed_requests_it == unprocessed_requests_for_session.end())
            continue;

        auto & unprocessed_requests = unprocessed_requests_it->second;
        unprocessed_requests.unprocessed_num -= count;

        if (unprocessed_requests.unprocessed_num == 0)
        {
            if (!unprocessed_requests.request_queue.empty())
            {
                auto & request_queue = unprocessed_requests.request_queue;
                unprocessed_requests.is_read = !unprocessed_requests.is_read;
                while (!request_queue.empty() && request_queue.front().request->isReadRequest() == unprocessed_requests.is_read)
                {
                    auto & front_request = request_queue.front();

                    /// Put close requests without timeouts
                    if (front_request.request->getOpNum() == Coordination::OpNum::Close)
                    {
                        if (!requests_queue->push(std::move(front_request)))
                            throw Exception("Cannot push request to queue", ErrorCodes::SYSTEM_ERROR);
                    }
                    else if (!requests_queue->tryPush(std::move(front_request), configuration_and_settings->coordination_settings->operation_timeout_ms.totalMilliseconds()))
                    {
                        throw Exception("Cannot push request to queue within operation timeout", ErrorCodes::TIMEOUT_EXCEEDED);
                    }

                    ++unprocessed_requests.unprocessed_num;
                    request_queue.pop_front();
                }
            }
            else
            {
                unprocessed_requests_for_session.erase(unprocessed_requests_it);
            }
        }
    }
}

void KeeperDispatcher::onRequestCommit(const KeeperStorage::RequestForSession & request_for_session, uint64_t log_term, uint64_t log_idx)
{
    finalizeRequests({request_for_session});

    KeeperStorage::RequestsForSessions requests;
    {
        std::lock_guard lock(leader_waiter_mutex);
        auto request_queue_it = leader_waiters.find(KeeperServer::NodeInfo{.term = log_term, .last_committed_index = log_idx});
        if (request_queue_it != leader_waiters.end())
        {
            requests = std::move(request_queue_it->second);
            leader_waiters.erase(request_queue_it);
        }
    }
    if (!read_requests_queue.push(std::move(requests)))
        throw Exception(ErrorCodes::SYSTEM_ERROR, "Cannot push read requests to queue");
}

bool KeeperDispatcher::isServerActive() const
{
    return checkInit() && hasLeader() && !server->isRecovering();
}

void KeeperDispatcher::updateConfiguration(const Poco::Util::AbstractConfiguration & config)
{
    auto diff = server->getConfigurationDiff(config);
    if (diff.empty())
        LOG_TRACE(log, "Configuration update triggered, but nothing changed for RAFT");
    else if (diff.size() > 1)
        LOG_WARNING(log, "Configuration changed for more than one server ({}) from cluster, it's strictly not recommended", diff.size());
    else
        LOG_DEBUG(log, "Configuration change size ({})", diff.size());

    for (auto & change : diff)
    {
        bool push_result = update_configuration_queue.push(change);
        if (!push_result)
            throw Exception(ErrorCodes::SYSTEM_ERROR, "Cannot push configuration update to queue");
    }
}

void KeeperDispatcher::updateKeeperStatLatency(uint64_t process_time_ms)
{
    keeper_stats.updateLatency(process_time_ms);
}

static uint64_t getDirSize(const fs::path & dir)
{
    checkStackSize();
    if (!fs::exists(dir))
        return 0;

    fs::directory_iterator it(dir);
    fs::directory_iterator end;

    uint64_t size{0};
    while (it != end)
    {
        if (it->is_regular_file())
            size += fs::file_size(*it);
        else
            size += getDirSize(it->path());
        ++it;
    }
    return size;
}

uint64_t KeeperDispatcher::getLogDirSize() const
{
    return getDirSize(configuration_and_settings->log_storage_path);
}

uint64_t KeeperDispatcher::getSnapDirSize() const
{
    return getDirSize(configuration_and_settings->snapshot_storage_path);
}

Keeper4LWInfo KeeperDispatcher::getKeeper4LWInfo() const
{
    Keeper4LWInfo result;
    result.is_follower = server->isFollower();
    result.is_standalone = !result.is_follower && server->getFollowerCount() == 0;
    result.is_leader = isLeader();
    result.is_observer = server->isObserver();
    result.has_leader = hasLeader();
    {
        std::lock_guard lock(push_request_mutex);
        result.outstanding_requests_count = requests_queue->size();
    }
    {
        std::lock_guard lock(session_to_response_callback_mutex);
        result.alive_connections_count = session_to_response_callback.size();
    }
    if (result.is_leader)
    {
        result.follower_count = server->getFollowerCount();
        result.synced_follower_count = server->getSyncedFollowerCount();
    }
    result.total_nodes_count = server->getKeeperStateMachine()->getNodesCount();
    result.last_zxid = server->getKeeperStateMachine()->getLastProcessedZxid();
    return result;
}

}
