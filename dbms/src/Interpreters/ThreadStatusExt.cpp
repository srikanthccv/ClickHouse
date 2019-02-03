#include <Common/ThreadStatus.h>
#include <Common/CurrentThread.h>
#include <Common/ThreadProfileEvents.h>
#include <Common/Exception.h>
#include <Interpreters/Context.h>
#include <Interpreters/QueryThreadLog.h>
#include <Interpreters/ProcessList.h>
#include <Interpreters/TraceCollector.h>
#include <IO/WriteBufferFromFileDescriptor.h>
#include <IO/WriteHelpers.h>
#include <common/logger_useful.h>

#include <csignal>
#include <time.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

/// Implement some methods of ThreadStatus and CurrentThread here to avoid extra linking dependencies in clickhouse_common_io
namespace DB
{

void ThreadStatus::attachQueryContext(Context & query_context_)
{
    query_context = &query_context_;
    if (!global_context)
        global_context = &query_context->getGlobalContext();

    if (!thread_group)
        return;

    std::unique_lock lock(thread_group->mutex);
    thread_group->query_context = query_context;
    if (!thread_group->global_context)
        thread_group->global_context = global_context;
}

String ThreadStatus::getQueryID()
{
    if (query_context)
        return query_context->getClientInfo().current_query_id;

    return {};
}

void CurrentThread::defaultThreadDeleter()
{
    ThreadStatus & thread = *CurrentThread::get();
    LOG_TRACE(thread.log, "Thread " << thread.thread_number << " exited");
    thread.detachQuery(true, true);
}

void ThreadStatus::initializeQuery()
{
    assertState({ThreadState::DetachedFromQuery}, __PRETTY_FUNCTION__);

    thread_group = std::make_shared<ThreadGroupStatus>();

    performance_counters.setParent(&thread_group->performance_counters);
    memory_tracker.setParent(&thread_group->memory_tracker);
    thread_group->memory_tracker.setDescription("(for query)");

    thread_group->master_thread = shared_from_this();
    thread_group->thread_statuses.emplace(thread_number, shared_from_this());

    initPerformanceCounters();
    thread_state = ThreadState::AttachedToQuery;
}

void ThreadStatus::attachQuery(const ThreadGroupStatusPtr & thread_group_, bool check_detached)
{
    if (thread_state == ThreadState::AttachedToQuery)
    {
        if (check_detached)
            throw Exception("Can't attach query to the thread, it is already attached", ErrorCodes::LOGICAL_ERROR);
        return;
    }

    assertState({ThreadState::DetachedFromQuery}, __PRETTY_FUNCTION__);

    if (!thread_group_)
        throw Exception("Attempt to attach to nullptr thread group", ErrorCodes::LOGICAL_ERROR);

    /// Attach current thread to thread group and copy useful information from it
    thread_group = thread_group_;

    performance_counters.setParent(&thread_group->performance_counters);
    memory_tracker.setParent(&thread_group->memory_tracker);

    {
        std::unique_lock lock(thread_group->mutex);

        logs_queue_ptr = thread_group->logs_queue_ptr;
        query_context = thread_group->query_context;

        if (!global_context)
            global_context = thread_group->global_context;

        if (!thread_group->thread_statuses.emplace(thread_number, shared_from_this()).second)
            throw Exception("Thread " + std::to_string(thread_number) + " is attached twice", ErrorCodes::LOGICAL_ERROR);
    }

    initPerformanceCounters();
    initQueryProfiler();

    thread_state = ThreadState::AttachedToQuery;
}

void ThreadStatus::finalizePerformanceCounters()
{
    if (performance_counters_finalized)
        return;

    performance_counters_finalized = true;
    updatePerformanceCounters();

    try
    {
        if (global_context && query_context)
        {
            auto & settings = query_context->getSettingsRef();
            if (settings.log_queries && settings.log_query_threads)
                if (auto thread_log = global_context->getQueryThreadLog())
                    logToQueryThreadLog(*thread_log);
        }
    }
    catch (...)
    {
        tryLogCurrentException(log);
    }
}

namespace {
    void queryProfilerTimerHandler(int sig, siginfo_t * /* info */, void * /* context */) {
        LOG_INFO(&Logger::get("laplab"), "Hello from handler!");

        char buffer[TraceCollector::buf_size];
        DB::WriteBufferFromFileDescriptor out(PipeSingleton::instance().write_fd, TraceCollector::buf_size, buffer);

        DB::writeBinary(sig, out);
        out.next();

        ::sleep(10);
    }
}

void ThreadStatus::initQueryProfiler() {
    if (!query_context) {
        LOG_INFO(log, "Query profiler disabled - no context");
        return;
    }

    struct sigevent sev;
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGALRM;
    sev._sigev_un._tid = os_thread_id;
    // TODO(laplab): get clock type from settings
    if (timer_create(CLOCK_REALTIME, &sev, &query_profiler_timer_id)) {
        throw Poco::Exception("Failed to create query profiler timer");
    }

    // TODO(laplab): get period from settings
    struct timespec period{.tv_sec = 0, .tv_nsec = 200000000};
    struct itimerspec timer_spec = {.it_interval = period, .it_value = period};
    if (timer_settime(query_profiler_timer_id, 0, &timer_spec, nullptr)) {
        throw Poco::Exception("Failed to set query profiler timer");
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = queryProfilerTimerHandler;
    sa.sa_flags = SA_SIGINFO;

    if (sigemptyset(&sa.sa_mask)) {
        throw Poco::Exception("Failed to clean signal mask for query profiler");
    }

    if (sigaddset(&sa.sa_mask, SIGALRM)) {
        throw Poco::Exception("Failed to add signal to mask for query profiler");
    }

    if (sigaction(SIGALRM, &sa, nullptr)) {
        throw Poco::Exception("Failed to setup signal handler for query profiler");
    }

    has_query_profiler = true;
}

void ThreadStatus::finalizeQueryProfiler() {
    if (!has_query_profiler) {
        return;
    }

    if (timer_delete(query_profiler_timer_id)) {
        throw Poco::Exception("Failed to delete query profiler timer");
    }

    has_query_profiler = false;
}

void ThreadStatus::detachQuery(bool exit_if_already_detached, bool thread_exits)
{
    if (exit_if_already_detached && thread_state == ThreadState::DetachedFromQuery)
    {
        thread_state = thread_exits ? ThreadState::Died : ThreadState::DetachedFromQuery;
        return;
    }

    assertState({ThreadState::AttachedToQuery}, __PRETTY_FUNCTION__);

    finalizeQueryProfiler();
    finalizePerformanceCounters();

    /// Detach from thread group
    performance_counters.setParent(&ProfileEvents::global_counters);
    memory_tracker.reset();

    /// Must reset pointer to thread_group's memory_tracker, because it will be destroyed two lines below.
    memory_tracker.setParent(nullptr);

    query_context = nullptr;
    thread_group.reset();

    thread_state = thread_exits ? ThreadState::Died : ThreadState::DetachedFromQuery;
}

void ThreadStatus::logToQueryThreadLog(QueryThreadLog & thread_log)
{
    QueryThreadLogElement elem;

    elem.event_time = time(nullptr);
    elem.query_start_time = query_start_time;
    elem.query_duration_ms = (getCurrentTimeNanoseconds() - query_start_time_nanoseconds) / 1000000U;

    elem.read_rows = progress_in.rows.load(std::memory_order_relaxed);
    elem.read_bytes = progress_in.bytes.load(std::memory_order_relaxed);
    elem.written_rows = progress_out.rows.load(std::memory_order_relaxed);
    elem.written_bytes = progress_out.bytes.load(std::memory_order_relaxed);
    elem.memory_usage = memory_tracker.get();
    elem.peak_memory_usage = memory_tracker.getPeak();

    elem.thread_name = getThreadName();
    elem.thread_number = thread_number;
    elem.os_thread_id = os_thread_id;

    if (thread_group)
    {
        {
            std::shared_lock lock(thread_group->mutex);

            if (thread_group->master_thread)
            {
                elem.master_thread_number = thread_group->master_thread->thread_number;
                elem.master_os_thread_id = thread_group->master_thread->os_thread_id;
            }

            elem.query = thread_group->query;
        }
    }

    if (query_context)
    {
        elem.client_info = query_context->getClientInfo();

        if (query_context->getSettingsRef().log_profile_events.value != 0)
        {
            /// NOTE: Here we are in the same thread, so we can make memcpy()
            elem.profile_counters = std::make_shared<ProfileEvents::Counters>(performance_counters.getPartiallyAtomicSnapshot());
        }
    }

    thread_log.add(elem);
}

void CurrentThread::initializeQuery()
{
    get()->initializeQuery();
    getScope()->deleter = CurrentThread::defaultThreadDeleter;
}

void CurrentThread::attachTo(const ThreadGroupStatusPtr & thread_group)
{
    get()->attachQuery(thread_group, true);
    getScope()->deleter = CurrentThread::defaultThreadDeleter;
}

void CurrentThread::attachToIfDetached(const ThreadGroupStatusPtr & thread_group)
{
    get()->attachQuery(thread_group, false);
    getScope()->deleter = CurrentThread::defaultThreadDeleter;
}

std::string CurrentThread::getCurrentQueryID()
{
    if (!get() || get().use_count() <= 0)
        return {};

    return get()->getQueryID();
}

void CurrentThread::attachQueryContext(Context & query_context)
{
    return get()->attachQueryContext(query_context);
}

void CurrentThread::finalizePerformanceCounters()
{
    get()->finalizePerformanceCounters();
}

void CurrentThread::detachQuery()
{
    get()->detachQuery(false);
}

void CurrentThread::detachQueryIfNotDetached()
{
    get()->detachQuery(true);
}


CurrentThread::QueryScope::QueryScope(Context & query_context)
{
    CurrentThread::initializeQuery();
    CurrentThread::attachQueryContext(query_context);
}

void CurrentThread::QueryScope::logPeakMemoryUsage()
{
    log_peak_memory_usage_in_destructor = false;
    CurrentThread::getGroup()->memory_tracker.logPeakMemoryUsage();
}

CurrentThread::QueryScope::~QueryScope()
{
    try
    {
        if (log_peak_memory_usage_in_destructor)
            logPeakMemoryUsage();

        CurrentThread::detachQueryIfNotDetached();
    }
    catch (...)
    {
        tryLogCurrentException("CurrentThread", __PRETTY_FUNCTION__);
    }
}

}
