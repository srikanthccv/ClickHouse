#include <Client/ConnectionPoolWithFailover.h>

#include <Poco/Net/NetException.h>
#include <Poco/Net/DNS.h>

#include <Common/BitHelpers.h>
#include <Common/quoteString.h>
#include <common/getFQDNOrHostName.h>
#include <Common/isLocalAddress.h>
#include <Common/ProfileEvents.h>
#include <Core/Settings.h>

#include <IO/ConnectionTimeouts.h>

namespace ProfileEvents
{
    extern const Event DistributedConnectionMissingTable;
    extern const Event DistributedConnectionStaleReplica;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int ATTEMPT_TO_READ_AFTER_EOF;
    extern const int NETWORK_ERROR;
    extern const int SOCKET_TIMEOUT;
    extern const int LOGICAL_ERROR;
}


ConnectionPoolWithFailover::ConnectionPoolWithFailover(
        ConnectionPoolPtrs nested_pools_,
        LoadBalancing load_balancing,
        time_t decrease_error_period_,
        size_t max_error_cap_)
    : Base(std::move(nested_pools_), decrease_error_period_, max_error_cap_, &Poco::Logger::get("ConnectionPoolWithFailover"))
    , default_load_balancing(load_balancing)
{
    const std::string & local_hostname = getFQDNOrHostName();

    hostname_differences.resize(nested_pools.size());
    for (size_t i = 0; i < nested_pools.size(); ++i)
    {
        ConnectionPool & connection_pool = dynamic_cast<ConnectionPool &>(*nested_pools[i]);
        hostname_differences[i] = getHostNameDifference(local_hostname, connection_pool.getHost());
    }
}

IConnectionPool::Entry ConnectionPoolWithFailover::get(const ConnectionTimeouts & timeouts,
                                                       const Settings * settings,
                                                       bool /*force_connected*/)
{
    TryGetEntryFunc try_get_entry = [&](NestedPool & pool, std::string & fail_message)
    {
        return tryGetEntry(pool, timeouts, fail_message, settings);
    };

    size_t offset = 0;
    if (settings)
        offset = settings->load_balancing_first_offset % nested_pools.size();
    GetPriorityFunc get_priority;
    switch (settings ? LoadBalancing(settings->load_balancing) : default_load_balancing)
    {
    case LoadBalancing::NEAREST_HOSTNAME:
        get_priority = [&](size_t i) { return hostname_differences[i]; };
        break;
    case LoadBalancing::IN_ORDER:
        get_priority = [](size_t i) { return i; };
        break;
    case LoadBalancing::RANDOM:
        break;
    case LoadBalancing::FIRST_OR_RANDOM:
        get_priority = [offset](size_t i) -> size_t { return i != offset; };
        break;
    case LoadBalancing::ROUND_ROBIN:
        if (last_used >= nested_pools.size())
            last_used = 0;
        ++last_used;
        /* Consider nested_pools.size() equals to 5
         * last_used = 1 -> get_priority: 0 1 2 3 4
         * last_used = 2 -> get_priority: 5 0 1 2 3
         * last_used = 3 -> get_priority: 5 4 0 1 2
         * ...
         * */
        get_priority = [&](size_t i) { ++i; return i < last_used ? nested_pools.size() - i : i - last_used; };
        break;
    }

    UInt64 max_ignored_errors = settings ? settings->distributed_replica_max_ignored_errors.value : 0;
    bool fallback_to_stale_replicas = settings ? settings->fallback_to_stale_replicas_for_distributed_queries.value : true;

    return Base::get(max_ignored_errors, fallback_to_stale_replicas, try_get_entry, get_priority);
}

Int64 ConnectionPoolWithFailover::getPriority() const
{
    return (*std::max_element(nested_pools.begin(), nested_pools.end(), [](const auto &a, const auto &b)
    {
        return a->getPriority() - b->getPriority();
    }))->getPriority();
}

ConnectionPoolWithFailover::Status ConnectionPoolWithFailover::getStatus() const
{
    const auto [states, pools, error_decrease_time] = getPoolExtendedStates();
    // NOTE: to avoid data races do not touch any data of ConnectionPoolWithFailover or PoolWithFailoverBase in the code below.

    assert(states.size() == pools.size());

    ConnectionPoolWithFailover::Status result;
    result.reserve(states.size());
    const time_t since_last_error_decrease = time(nullptr) - error_decrease_time;

    for (size_t i = 0; i < states.size(); ++i)
    {
        const auto rounds_to_zero_errors = states[i].error_count ? bitScanReverse(states[i].error_count) + 1 : 0;
        const auto seconds_to_zero_errors = std::max(static_cast<time_t>(0), rounds_to_zero_errors * decrease_error_period - since_last_error_decrease);

        result.emplace_back(NestedPoolStatus{
            pools[i],
            states[i].error_count,
            std::chrono::seconds{seconds_to_zero_errors}
        });
    }

    return result;
}

std::vector<IConnectionPool::Entry> ConnectionPoolWithFailover::getMany(const ConnectionTimeouts & timeouts,
                                                                        const Settings * settings,
                                                                        PoolMode pool_mode)
{
    TryGetEntryFunc try_get_entry = [&](NestedPool & pool, std::string & fail_message)
    {
        return tryGetEntry(pool, timeouts, fail_message, settings);
    };

    std::vector<TryResult> results = getManyImpl(settings, pool_mode, try_get_entry);

    std::vector<Entry> entries;
    entries.reserve(results.size());
    for (auto & result : results)
        entries.emplace_back(std::move(result.entry));
    return entries;
}

std::vector<ConnectionPoolWithFailover::TryResult> ConnectionPoolWithFailover::getManyForTableFunction(
    const ConnectionTimeouts & timeouts,
    const Settings * settings,
    PoolMode pool_mode)
{
    TryGetEntryFunc try_get_entry = [&](NestedPool & pool, std::string & fail_message)
    {
        return tryGetEntry(pool, timeouts, fail_message, settings);
    };

    return getManyImpl(settings, pool_mode, try_get_entry);
}

std::vector<ConnectionPoolWithFailover::TryResult> ConnectionPoolWithFailover::getManyChecked(
    const ConnectionTimeouts & timeouts,
    const Settings * settings, PoolMode pool_mode,
    const QualifiedTableName & table_to_check)
{
    TryGetEntryFunc try_get_entry = [&](NestedPool & pool, std::string & fail_message)
    {
        return tryGetEntry(pool, timeouts, fail_message, settings, &table_to_check);
    };

    return getManyImpl(settings, pool_mode, try_get_entry);
}

ConnectionPoolWithFailover::Base::GetPriorityFunc ConnectionPoolWithFailover::makeGetPriorityFunc(const Settings * settings)
{
    size_t offset = 0;
    if (settings)
        offset = settings->load_balancing_first_offset % nested_pools.size();

    GetPriorityFunc get_priority;
    switch (settings ? LoadBalancing(settings->load_balancing) : default_load_balancing)
    {
        case LoadBalancing::NEAREST_HOSTNAME:
            get_priority = [&](size_t i) { return hostname_differences[i]; };
            break;
        case LoadBalancing::IN_ORDER:
            get_priority = [](size_t i) { return i; };
            break;
        case LoadBalancing::RANDOM:
            break;
        case LoadBalancing::FIRST_OR_RANDOM:
            get_priority = [offset](size_t i) -> size_t { return i != offset; };
            break;
        case LoadBalancing::ROUND_ROBIN:
            if (last_used >= nested_pools.size())
                last_used = 0;
            ++last_used;
            /* Consider nested_pools.size() equals to 5
             * last_used = 1 -> get_priority: 0 1 2 3 4
             * last_used = 2 -> get_priority: 5 0 1 2 3
             * last_used = 3 -> get_priority: 5 4 0 1 2
             * ...
             * */
            get_priority = [&](size_t i) { ++i; return i < last_used ? nested_pools.size() - i : i - last_used; };
            break;
    }

    return get_priority;
}

std::vector<ConnectionPoolWithFailover::TryResult> ConnectionPoolWithFailover::getManyImpl(
        const Settings * settings,
        PoolMode pool_mode,
        const TryGetEntryFunc & try_get_entry)
{
    size_t min_entries = (settings && settings->skip_unavailable_shards) ? 0 : 1;
    size_t max_tries = (settings ?
        size_t{settings->connections_with_failover_max_tries} :
        size_t{DBMS_CONNECTION_POOL_WITH_FAILOVER_DEFAULT_MAX_TRIES});
    size_t max_entries;
    if (pool_mode == PoolMode::GET_ALL)
    {
        min_entries = nested_pools.size();
        max_entries = nested_pools.size();
    }
    else if (pool_mode == PoolMode::GET_ONE)
        max_entries = 1;
    else if (pool_mode == PoolMode::GET_MANY)
        max_entries = settings ? size_t(settings->max_parallel_replicas) : 1;
    else
        throw DB::Exception("Unknown pool allocation mode", DB::ErrorCodes::LOGICAL_ERROR);

    GetPriorityFunc get_priority = makeGetPriorityFunc(settings);

    UInt64 max_ignored_errors = settings ? settings->distributed_replica_max_ignored_errors.value : 0;
    bool fallback_to_stale_replicas = settings ? settings->fallback_to_stale_replicas_for_distributed_queries.value : true;

    return Base::getMany(min_entries, max_entries, max_tries,
        max_ignored_errors, fallback_to_stale_replicas,
        try_get_entry, get_priority);
}

ConnectionPoolWithFailover::TryResult
ConnectionPoolWithFailover::tryGetEntry(
        IConnectionPool & pool,
        const ConnectionTimeouts & timeouts,
        std::string & fail_message,
        const Settings * settings,
        const QualifiedTableName * table_to_check)
{
    TryGetConnection try_get_connection(&pool, &timeouts, settings, table_to_check, log, false);
    try_get_connection.run();
    fail_message = try_get_connection.fail_message;
    return try_get_connection.result;
}

std::vector<ConnectionPoolWithFailover::Base::ShuffledPool> ConnectionPoolWithFailover::getShuffledPools(const Settings * settings)
{
    GetPriorityFunc get_priority = makeGetPriorityFunc(settings);
    UInt64 max_ignored_errors = settings ? settings->distributed_replica_max_ignored_errors.value : 0;
    return Base::getShuffledPools(max_ignored_errors, get_priority);
}

TryGetConnection::TryGetConnection(
    IConnectionPool * pool_,
    const ConnectionTimeouts * timeouts_,
    const Settings * settings_,
    const QualifiedTableName * table_to_check_,
    Poco::Logger * log_,
    bool non_blocking_) :
        pool(pool_), timeouts(timeouts_), settings(settings_),
        table_to_check(table_to_check_), log(log_), stage(Stage::CONNECT), socket_fd(-1), non_blocking(non_blocking_)
{
}

void TryGetConnection::reset()
{
    resetResult();
    stage = Stage::CONNECT;
    action_before_disconnect = nullptr;
    socket_fd = -1;
    fail_message.clear();
}

void TryGetConnection::resetResult()
{
    if (!result.entry.isNull())
    {
        result.entry->disconnect();
        result.reset();
    }
}

void TryGetConnection::processFail(bool add_description)
{
    if (action_before_disconnect)
        action_before_disconnect(socket_fd);

    fail_message = getCurrentExceptionMessage(/* with_stacktrace = */ false);
    if (add_description)
        fail_message += " (" + result.entry->getDescription() + ")";
    resetResult();
    socket_fd = -1;
    stage = Stage::FAILED;
}

void TryGetConnection::run()
{
    try
    {
        if (stage == Stage::CONNECT)
        {
            result.entry = pool->get(*timeouts, settings, /* force_connected = */ false);

            if (!result.entry->isConnected())
            {
                result.entry->prepare(*timeouts);
                socket_fd = result.entry->getSocket()->impl()->sockfd();
                result.entry->sendHello();
                stage = Stage::RECEIVE_HELLO;
                /// We are waiting for hello from replica.
                if (non_blocking)
                    return;
            }

            socket_fd = result.entry->getSocket()->impl()->sockfd();
            stage = Stage::START_CHECK_TABLE;
        }

        if (stage == Stage::RECEIVE_HELLO)
        {
            result.entry->receiveHello();
            stage = Stage::START_CHECK_TABLE;
        }

        if (stage == Stage::START_CHECK_TABLE)
        {
            UInt64 server_revision = 0;
            if (table_to_check)
                server_revision = result.entry->getServerRevision(*timeouts);

            if (!table_to_check || server_revision < DBMS_MIN_REVISION_WITH_TABLES_STATUS)
            {
                result.entry->forceConnected(*timeouts);
                result.is_usable = true;
                result.is_up_to_date = true;
                stage = FINISHED;
                if (non_blocking)
                    return;
            }

            TablesStatusRequest status_request;
            status_request.tables.emplace(*table_to_check);

            result.entry->sendTablesStatusRequest(status_request);
            stage = Stage::RECEIVE_TABLES_STATUS;
            /// We are waiting for tables status response.
            if (non_blocking)
                return;
        }

        if (stage == Stage::RECEIVE_TABLES_STATUS)
        {
            TablesStatusResponse status_response = result.entry->receiveTablesStatusResponse();
            auto table_status_it = status_response.table_states_by_id.find(*table_to_check);
            if (table_status_it == status_response.table_states_by_id.end())
            {
                const char * message_pattern = "There is no table {}.{} on server: {}";
                fail_message = fmt::format(message_pattern, backQuote(table_to_check->database), backQuote(table_to_check->table), result.entry->getDescription());
                LOG_WARNING(log, fail_message);
                ProfileEvents::increment(ProfileEvents::DistributedConnectionMissingTable);
                stage = Stage::FINISHED;
                return;
            }

            result.is_usable = true;

            UInt64 max_allowed_delay = settings ? UInt64(settings->max_replica_delay_for_distributed_queries) : 0;
            if (!max_allowed_delay)
            {
                result.is_up_to_date = true;
                stage = Stage::FINISHED;
                return;
            }

            UInt32 delay = table_status_it->second.absolute_delay;

            if (delay < max_allowed_delay)
                result.is_up_to_date = true;
            else
            {
                result.is_up_to_date = false;
                result.staleness = delay;

                LOG_TRACE(log, "Server {} has unacceptable replica delay for table {}.{}: {}", result.entry->getDescription(), table_to_check->database, table_to_check->table, delay);
                ProfileEvents::increment(ProfileEvents::DistributedConnectionStaleReplica);
            }
        }

        stage = Stage::FINISHED;
    }
    catch (Poco::Net::NetException &)
    {
        processFail(true);
    }
    catch (Poco::TimeoutException &)
    {
        processFail(true);
    }
    catch (const Exception & e)
    {
        if (e.code() != ErrorCodes::ATTEMPT_TO_READ_AFTER_EOF)
            throw;

        processFail(false);
    }
}

}
