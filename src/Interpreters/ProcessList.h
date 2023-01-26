#pragma once

#include <Core/Defines.h>
#include <IO/Progress.h>
#include <Interpreters/CancellationCode.h>
#include <Interpreters/ClientInfo.h>
#include <Interpreters/QueryPriorities.h>
#include <Interpreters/TemporaryDataOnDisk.h>
#include <Interpreters/Context.h>
#include <QueryPipeline/BlockIO.h>
#include <QueryPipeline/ExecutionSpeedLimits.h>
#include <Storages/IStorage_fwd.h>
#include <Poco/Condition.h>
#include <Parsers/IAST.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/MemoryTracker.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/Throttler.h>
#include <Common/OvercommitTracker.h>

#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


namespace DB
{

struct Settings;
class IAST;
class PipelineExecutor;

struct ProcessListForUser;
class QueryStatus;
class ThreadStatus;
class ProcessListEntry;


/** List of currently executing queries.
  * Also implements limit on their number.
  */

/** Information of process list element.
  * To output in SHOW PROCESSLIST query. Does not contain any complex objects, that do something on copy or destructor.
  */
struct QueryStatusInfo
{
    String query;
    UInt64 elapsed_microseconds;
    size_t read_rows;
    size_t read_bytes;
    size_t total_rows;
    size_t written_rows;
    size_t written_bytes;
    Int64 memory_usage;
    Int64 peak_memory_usage;
    ClientInfo client_info;
    bool is_cancelled;
    bool is_all_data_sent;

    /// Optional fields, filled by query
    std::vector<UInt64> thread_ids;
    std::shared_ptr<ProfileEvents::Counters::Snapshot> profile_counters;
    std::shared_ptr<Settings> query_settings;
    std::string current_database;
};

/// Query and information about its execution.
class QueryStatus : public WithContext
{
protected:
    friend class ProcessList;
    friend class ThreadStatus;
    friend class CurrentThread;
    friend class ProcessListEntry;
    friend struct ::GlobalOvercommitTracker;

    String query;
    ClientInfo client_info;

    /// Info about all threads involved in query execution
    ThreadGroupStatusPtr thread_group;

    Stopwatch watch;

    /// Progress of input stream
    Progress progress_in;
    /// Progress of output stream
    Progress progress_out;

    /// Used to externally check for the query time limits
    /// They are saved in the constructor to limit the overhead of each call to checkTimeLimit()
    ExecutionSpeedLimits limits;
    OverflowMode overflow_mode;

    QueryPriorities::Handle priority_handle = nullptr;

    /// True if query cancellation is in progress right now
    /// ProcessListEntry should not be destroyed if is_cancelling is true
    /// Flag changes is synced with ProcessListBase::mutex and notified with ProcessList::cancelled_cv
    bool is_cancelling { false };
    /// KILL was send to the query
    std::atomic<bool> is_killed { false };

    /// All data to the client already had been sent.
    /// Including EndOfStream or Exception.
    std::atomic<bool> is_all_data_sent { false };

    void setUserProcessList(ProcessListForUser * user_process_list_);
    /// Be careful using it. For example, queries field of ProcessListForUser could be modified concurrently.
    const ProcessListForUser * getUserProcessList() const { return user_process_list; }

    mutable std::mutex executors_mutex;

    /// Array of PipelineExecutors to be cancelled when a cancelQuery is received
    std::vector<PipelineExecutor *> executors;

    enum QueryStreamsStatus
    {
        NotInitialized,
        Initialized,
        Released
    };

    QueryStreamsStatus query_streams_status{NotInitialized};

    ProcessListForUser * user_process_list = nullptr;

    OvercommitTracker * global_overcommit_tracker = nullptr;

    IAST::QueryKind query_kind;

    /// This field is unused in this class, but it
    /// increments/decrements metric in constructor/destructor.
    CurrentMetrics::Increment num_queries_increment;

public:
    QueryStatus(
        ContextPtr context_,
        const String & query_,
        const ClientInfo & client_info_,
        QueryPriorities::Handle && priority_handle_,
        ThreadGroupStatusPtr && thread_group_,
        IAST::QueryKind query_kind_,
        UInt64 watch_start_nanoseconds);

    ~QueryStatus();

    const ClientInfo & getClientInfo() const
    {
        return client_info;
    }

    ProgressValues getProgressIn() const
    {
        return progress_in.getValues();
    }

    ProgressValues getProgressOut() const
    {
        return progress_out.getValues();
    }

    ThrottlerPtr getUserNetworkThrottler();

    MemoryTracker * getMemoryTracker() const
    {
        if (!thread_group)
            return nullptr;
        return &thread_group->memory_tracker;
    }

    IAST::QueryKind getQueryKind() const
    {
        return query_kind;
    }

    bool updateProgressIn(const Progress & value)
    {
        CurrentThread::updateProgressIn(value);
        progress_in.incrementPiecewiseAtomically(value);

        if (priority_handle)
            priority_handle->waitIfNeed(std::chrono::seconds(1));        /// NOTE Could make timeout customizable.

        return !is_killed.load(std::memory_order_relaxed);
    }

    bool updateProgressOut(const Progress & value)
    {
        CurrentThread::updateProgressOut(value);
        progress_out.incrementPiecewiseAtomically(value);

        return !is_killed.load(std::memory_order_relaxed);
    }

    QueryStatusInfo getInfo(bool get_thread_list = false, bool get_profile_events = false, bool get_settings = false) const;

    CancellationCode cancelQuery(bool kill);

    bool isKilled() const { return is_killed; }

    bool isAllDataSent() const { return is_all_data_sent; }
    void setAllDataSent() { is_all_data_sent = true; }

    /// Adds a pipeline to the QueryStatus
    void addPipelineExecutor(PipelineExecutor * e);

    /// Removes a pipeline to the QueryStatus
    void removePipelineExecutor(PipelineExecutor * e);

    /// Checks the query time limits (cancelled or timeout)
    bool checkTimeLimit();
    /// Same as checkTimeLimit but it never throws
    [[nodiscard]] bool checkTimeLimitSoft();

    /// Get the reference for the start of the query. Used to synchronize with other Stopwatches
    UInt64 getQueryCPUStartTime() { return watch.getStart(); }
};

using QueryStatusPtr = std::shared_ptr<QueryStatus>;


/// Information of process list for user.
struct ProcessListForUserInfo
{
    Int64 memory_usage;
    Int64 peak_memory_usage;

    // Optional field, filled by request.
    std::shared_ptr<ProfileEvents::Counters::Snapshot> profile_counters;
};


/// Data about queries for one user.
struct ProcessListForUser
{
    explicit ProcessListForUser(ProcessList * global_process_list);

    ProcessListForUser(ContextPtr global_context, ProcessList * global_process_list);

    /// query_id -> ProcessListElement(s). There can be multiple queries with the same query_id as long as all queries except one are cancelled.
    using QueryToElement = std::unordered_map<String, QueryStatusPtr>;
    QueryToElement queries;

    ProfileEvents::Counters user_performance_counters{VariableContext::User, &ProfileEvents::global_counters};
    /// Limit and counter for memory of all simultaneously running queries of single user.
    MemoryTracker user_memory_tracker{VariableContext::User};

    TemporaryDataOnDiskScopePtr user_temp_data_on_disk;

    UserOvercommitTracker user_overcommit_tracker;

    /// Count network usage for all simultaneously running queries of single user.
    ThrottlerPtr user_throttler;

    ProcessListForUserInfo getInfo(bool get_profile_events = false) const;

    /// Clears MemoryTracker for the user.
    /// Sometimes it is important to reset the MemoryTracker, because it may accumulate skew
    ///  due to the fact that there are cases when memory can be allocated while processing the query, but released later.
    /// Clears network bandwidth Throttler, so it will not count periods of inactivity.
    void resetTrackers()
    {
        /// TODO: should we drop user_temp_data_on_disk here?
        user_memory_tracker.reset();
        if (user_throttler)
            user_throttler.reset();
    }
};


class ProcessList;


/// Keeps iterator to process list and removes element in destructor.
class ProcessListEntry
{
private:
    using Container = std::list<QueryStatusPtr>;

    ProcessList & parent;
    Container::iterator it;

public:
    ProcessListEntry(ProcessList & parent_, Container::iterator it_)
        : parent(parent_), it(it_) {}

    ~ProcessListEntry();

    QueryStatusPtr getQueryStatus() { return *it; }
    const QueryStatusPtr getQueryStatus() const { return *it; }
};


class ProcessListBase
{
    mutable std::mutex mutex;

protected:
    using Lock = std::unique_lock<std::mutex>;
    struct LockAndBlocker
    {
        Lock lock;
        OvercommitTrackerBlockerInThread blocker;
    };

    // It is forbidden to do allocations/deallocations with acquired mutex and
    // enabled OvercommitTracker. This leads to deadlock in the case of OOM.
    LockAndBlocker safeLock() const noexcept { return { std::unique_lock{mutex}, {} }; }
    Lock unsafeLock() const noexcept { return std::unique_lock{mutex}; }
};


class ProcessList : public ProcessListBase
{
public:
    using Element = QueryStatusPtr;
    using Entry = ProcessListEntry;
    using QueryAmount = UInt64;

    /// list, for iterators not to invalidate. NOTE: could replace with cyclic buffer, but not worth.
    using Container = std::list<Element>;
    using Info = std::vector<QueryStatusInfo>;
    using UserInfo = std::unordered_map<String, ProcessListForUserInfo>;

    /// User -> queries
    using UserToQueries = std::unordered_map<String, ProcessListForUser>;

    using QueryKindAmounts = std::unordered_map<IAST::QueryKind, QueryAmount>;

protected:
    friend class ProcessListEntry;
    friend struct ::OvercommitTracker;
    friend struct ::UserOvercommitTracker;
    friend struct ::GlobalOvercommitTracker;

    mutable std::condition_variable have_space;        /// Number of currently running queries has become less than maximum.

    /// List of queries
    Container processes;
    /// Notify about cancelled queries (done with ProcessListBase::mutex acquired).
    mutable std::condition_variable cancelled_cv;

    size_t max_size = 0;        /// 0 means no limit. Otherwise, when limit exceeded, an exception is thrown.

    /// Stores per-user info: queries, statistics and limits
    UserToQueries user_to_queries;

    /// Stores info about queries grouped by their priority
    QueryPriorities priorities;

    /// Limit network bandwidth for all users
    ThrottlerPtr total_network_throttler;

    /// Call under lock. Finds process with specified current_user and current_query_id.
    QueryStatusPtr tryGetProcessListElement(const String & current_query_id, const String & current_user);

    /// limit for insert. 0 means no limit. Otherwise, when limit exceeded, an exception is thrown.
    size_t max_insert_queries_amount = 0;

    /// limit for select. 0 means no limit. Otherwise, when limit exceeded, an exception is thrown.
    size_t max_select_queries_amount = 0;

    /// amount of queries by query kind.
    QueryKindAmounts query_kind_amounts;

    void increaseQueryKindAmount(const IAST::QueryKind & query_kind);
    void decreaseQueryKindAmount(const IAST::QueryKind & query_kind);
    QueryAmount getQueryKindAmount(const IAST::QueryKind & query_kind) const;

public:
    using EntryPtr = std::shared_ptr<ProcessListEntry>;

    /** Register running query. Returns refcounted object, that will remove element from list in destructor.
      * If too many running queries - wait for not more than specified (see settings) amount of time.
      * If timeout is passed - throw an exception.
      * Don't count KILL QUERY queries.
      */
    EntryPtr insert(const String & query_, const IAST * ast, ContextMutablePtr query_context, UInt64 watch_start_nanoseconds);

    /// Number of currently executing queries.
    size_t size() const { return processes.size(); }

    /// Get current state of process list.
    Info getInfo(bool get_thread_list = false, bool get_profile_events = false, bool get_settings = false) const;

    /// Get current state of process list per user.
    UserInfo getUserInfo(bool get_profile_events = false) const;

    void setMaxSize(size_t max_size_)
    {
        auto lock = unsafeLock();
        max_size = max_size_;
    }

    void setMaxInsertQueriesAmount(size_t max_insert_queries_amount_)
    {
        auto lock = unsafeLock();
        max_insert_queries_amount = max_insert_queries_amount_;
    }

    void setMaxSelectQueriesAmount(size_t max_select_queries_amount_)
    {
        auto lock = unsafeLock();
        max_select_queries_amount = max_select_queries_amount_;
    }

    /// Try call cancel() for input and output streams of query with specified id and user
    CancellationCode sendCancelToQuery(const String & current_query_id, const String & current_user, bool kill = false);

    void killAllQueries();
};

}
