#include <Interpreters/ProcessList.h>
#include <Interpreters/Settings.h>
#include <Parsers/ASTKillQueryQuery.h>
#include <Common/Exception.h>
#include <IO/WriteHelpers.h>
#include <DataStreams/IProfilingBlockInputStream.h>
#include <Common/typeid_cast.h>
#include <common/logger_useful.h>
#include <pthread.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_MUCH_SIMULTANEOUS_QUERIES;
    extern const int QUERY_WITH_SAME_ID_IS_ALREADY_RUNNING;
}


ProcessList::EntryPtr ProcessList::insert(
    const String & query_, const IAST * ast, const ClientInfo & client_info, const Settings & settings)
{
    EntryPtr res;
    bool is_kill_query = ast && typeid_cast<const ASTKillQueryQuery *>(ast);

    {
        std::lock_guard<std::mutex> lock(mutex);

        if (!is_kill_query && max_size && processes.size() >= max_size
            && (!settings.queue_max_wait_ms.totalMilliseconds() || !have_space.tryWait(mutex, settings.queue_max_wait_ms.totalMilliseconds())))
            throw Exception("Too many simultaneous queries. Maximum: " + toString(max_size), ErrorCodes::TOO_MUCH_SIMULTANEOUS_QUERIES);

        /** Why we use current user?
          * Because initial one is passed by client and credentials for it is not verified,
          *  and using initial_user for limits will be insecure.
          *
          * Why we use current_query_id?
          * Because we want to allow distributed queries that will run multiple secondary queries on same server,
          *  like SELECT count() FROM remote('127.0.0.{1,2}', system.numbers)
          *  so they must have different query_ids.
          */

        {
            auto user_process_list = user_to_queries.find(client_info.current_user);

            if (user_process_list != user_to_queries.end())
            {
                if (!is_kill_query && settings.max_concurrent_queries_for_user
                    && user_process_list->second.queries.size() >= settings.max_concurrent_queries_for_user)
                    throw Exception("Too many simultaneous queries for user " + client_info.current_user
                        + ". Current: " + toString(user_process_list->second.queries.size())
                        + ", maximum: " + settings.max_concurrent_queries_for_user.toString(),
                        ErrorCodes::TOO_MUCH_SIMULTANEOUS_QUERIES);

                if (!client_info.current_query_id.empty())
                {
                    auto element = user_process_list->second.queries.find(client_info.current_query_id);
                    if (element != user_process_list->second.queries.end())
                    {
                        if (!settings.replace_running_query)
                            throw Exception("Query with id = " + client_info.current_query_id + " is already running.",
                                ErrorCodes::QUERY_WITH_SAME_ID_IS_ALREADY_RUNNING);

                        /// Kill query could be replaced since system.processes is continuously updated
                        element->second->is_cancelled = true;
                        /// If the request is canceled, the data about it is deleted from the map at the time of cancellation.
                        user_process_list->second.queries.erase(element);
                    }
                }
            }
        }

        auto process_it = processes.emplace(processes.end(),
            query_, client_info,
            settings.limits.max_memory_usage, settings.memory_tracker_fault_probability,
            priorities.insert(settings.priority));

        res = std::make_shared<Entry>(*this, process_it);

        if (!client_info.current_query_id.empty())
        {
            ProcessListForUser & user_process_list = user_to_queries[client_info.current_user];
            user_process_list.queries[client_info.current_query_id] = &*process_it;

            /// Limits are only raised (to be more relaxed) or set to something instead of zero,
            ///  because settings for different queries will interfere each other:
            ///  setting from one query effectively sets values for all other queries.

            /// Track memory usage for all simultaneously running queries.
            /// You should specify this value in configuration for default profile,
            ///  not for specific users, sessions or queries,
            ///  because this setting is effectively global.
            total_memory_tracker.setOrRaiseLimit(settings.limits.max_memory_usage_for_all_queries);
            total_memory_tracker.setDescription("(total)");

            /// Track memory usage for all simultaneously running queries from single user.
            user_process_list.user_memory_tracker.setParent(&total_memory_tracker);
            user_process_list.user_memory_tracker.setOrRaiseLimit(settings.limits.max_memory_usage_for_user);
            user_process_list.user_memory_tracker.setDescription("(for user)");

            /// Query-level memory tracker is already set in the QueryStatus constructor

            if (!current_thread)
                throw Exception("Thread is not initialized", ErrorCodes::LOGICAL_ERROR);

            if (current_thread)
            {
                current_thread->setCurrentThreadParentQuery(&*process_it);
                current_thread->memory_tracker.setOrRaiseLimit(settings.limits.max_memory_usage);
                current_thread->memory_tracker.setDescription("(for thread)");
            }

            if (settings.limits.max_network_bandwidth_for_user && !user_process_list.user_throttler)
            {
                user_process_list.user_throttler = std::make_shared<Throttler>(settings.limits.max_network_bandwidth_for_user, 0,
                    "Network bandwidth limit for a user exceeded.");
            }

            process_it->setUserProcessList(&user_process_list);
        }
    }

    return res;
}


ProcessListEntry::~ProcessListEntry()
{
    /// Destroy all streams to avoid long lock of ProcessList
    it->releaseQueryStreams();

    /// Finalize all threads statuses
    {
        std::lock_guard lock(it->threads_mutex);

        for (auto & elem : it->thread_statuses)
        {
            auto & thread_status = elem.second;
            thread_status->onExit();
            thread_status->reset();
            thread_status.reset();
        }

        it->thread_statuses.clear();
    }

    /// Also reset query master thread status
    /// NOTE: we can't destroy it, since master threads are selected from fixed thread pool
    if (current_thread)
        current_thread->reset();

    std::lock_guard<std::mutex> lock(parent.mutex);

    /// The order of removing memory_trackers is important.

    String user = it->client_info.current_user;
    String query_id = it->client_info.current_query_id;
    bool is_cancelled = it->is_cancelled;

    /// This removes the memory_tracker of one request.
    parent.processes.erase(it);

    auto user_process_list = parent.user_to_queries.find(user);
    if (user_process_list != parent.user_to_queries.end())
    {
        /// In case the request is canceled, the data about it is deleted from the map at the time of cancellation, and not here.
        if (!is_cancelled && !query_id.empty())
        {
            auto element = user_process_list->second.queries.find(query_id);
            if (element != user_process_list->second.queries.end())
                user_process_list->second.queries.erase(element);
        }

        /// This removes the memory_tracker from the user. At this time, the memory_tracker that references it does not live.

        /// If there are no more queries for the user, then we delete the entry.
        /// This also clears the MemoryTracker for the user, and a message about the memory consumption is output to the log.
        /// This also clears network bandwidth Throttler, so it will not count periods of inactivity.
        /// Sometimes it is important to reset the MemoryTracker, because it may accumulate skew
        ///  due to the fact that there are cases when memory can be allocated while processing the request, but released later.
        if (user_process_list->second.queries.empty())
            parent.user_to_queries.erase(user_process_list);
    }

    parent.have_space.signal();

    /// This removes memory_tracker for all requests. At this time, no other memory_trackers live.
    if (parent.processes.size() == 0)
    {
        /// Reset MemoryTracker, similarly (see above).
        parent.total_memory_tracker.logPeakMemoryUsage();
        parent.total_memory_tracker.reset();
    }
}


QueryStatus::QueryStatus(
    const String & query_,
    const ClientInfo & client_info_,
    size_t max_memory_usage,
    double memory_tracker_fault_probability,
    QueryPriorities::Handle && priority_handle_)
    :
    query(query_),
    client_info(client_info_),
    priority_handle(std::move(priority_handle_)),
    performance_counters(ProfileEvents::Level::Process),
    num_queries_increment{CurrentMetrics::Query}
{
    memory_tracker.setOrRaiseLimit(max_memory_usage);
    memory_tracker.setDescription("(for query)");

    if (memory_tracker_fault_probability)
        memory_tracker.setFaultProbability(memory_tracker_fault_probability);
}


void QueryStatus::setQueryStreams(const BlockIO & io)
{
    std::lock_guard<std::mutex> lock(query_streams_mutex);

    query_stream_in = io.in;
    query_stream_out = io.out;
    query_streams_initialized = true;
}

void QueryStatus::releaseQueryStreams()
{
    std::lock_guard<std::mutex> lock(query_streams_mutex);

    query_streams_initialized = false;
    query_streams_released = true;
    query_stream_in.reset();
    query_stream_out.reset();
}

bool QueryStatus::streamsAreReleased()
{
    std::lock_guard<std::mutex> lock(query_streams_mutex);

    return query_streams_released;
}

bool QueryStatus::tryGetQueryStreams(BlockInputStreamPtr & in, BlockOutputStreamPtr & out) const
{
    std::lock_guard<std::mutex> lock(query_streams_mutex);

    if (!query_streams_initialized)
        return false;

    in = query_stream_in;
    out = query_stream_out;
    return true;
}


void QueryStatus::setUserProcessList(ProcessListForUser * user_process_list_)
{
    user_process_list = user_process_list_;
    performance_counters.parent = &user_process_list->user_performance_counters;
    memory_tracker.setParent(&user_process_list->user_memory_tracker);
}


void ProcessList::addTemporaryTable(QueryStatus & elem, const String & table_name, const StoragePtr & storage)
{
    std::lock_guard<std::mutex> lock(mutex);

    elem.temporary_tables[table_name] = storage;
}


QueryStatus * ProcessList::tryGetProcessListElement(const String & current_query_id, const String & current_user)
{
    auto user_it = user_to_queries.find(current_user);
    if (user_it != user_to_queries.end())
    {
        const auto & user_queries = user_it->second.queries;
        auto query_it = user_queries.find(current_query_id);

        if (query_it != user_queries.end())
            return query_it->second;
    }

    return nullptr;
}


ProcessList::CancellationCode ProcessList::sendCancelToQuery(const String & current_query_id, const String & current_user, bool kill)
{
    std::lock_guard<std::mutex> lock(mutex);

    QueryStatus * elem = tryGetProcessListElement(current_query_id, current_user);

    if (!elem)
        return CancellationCode::NotFound;

    /// Streams are destroyed, and ProcessListElement will be deleted from ProcessList soon. We need wait a little bit
    if (elem->streamsAreReleased())
        return CancellationCode::CancelSent;

    BlockInputStreamPtr input_stream;
    BlockOutputStreamPtr output_stream;

    if (elem->tryGetQueryStreams(input_stream, output_stream))
    {
        IProfilingBlockInputStream * input_stream_casted;
        if (input_stream && (input_stream_casted = dynamic_cast<IProfilingBlockInputStream *>(input_stream.get())))
        {
            input_stream_casted->cancel(kill);
            return CancellationCode::CancelSent;
        }
        return CancellationCode::CancelCannotBeSent;
    }

    return CancellationCode::QueryIsNotInitializedYet;
}


ProcessListForUser::ProcessListForUser()
: user_performance_counters(ProfileEvents::Level::User, &ProfileEvents::global_counters)
{}


}
