#pragma once

#include <deque>
#include <functional>
#include <atomic>
#include <mutex>
#include <future>
#include <condition_variable>
#include <set>

#include <boost/circular_buffer.hpp>

#include <common/shared_ptr_helper.h>
#include <common/logger_useful.h>
#include <Common/ThreadPool.h>
#include <Common/Stopwatch.h>
#include <Storages/MergeTree/IExecutableTask.h>


namespace DB
{

/**
 *  Executor for a background MergeTree related operations such as merges, mutations, fetches an so on.
 *  It can execute only successors of ExecutableTask interface.
 *  Which is a self-written coroutine. It suspends, when returns true from execute() method.
 *
 *  There are two queues of a tasks: pending (main queue for all the tasks) and active (currently executing).
 *  Pending queue is needed since the number of tasks will be more than thread to execute.
 *  Pending tasks are tasks that successfully scheduled to an executor or tasks that have some extra steps to execute.
 *  There is an invariant, that task may occur only in one of these queue. It can occur in both queues only in critical sections.
 *
 *  Pending:                                              Active:
 *
 *  |s| |s| |s| |s| |s| |s| |s| |s| |s| |s|               |s|
 *  |s| |s| |s| |s| |s| |s| |s| |s| |s|                   |s|
 *  |s| |s|     |s|     |s| |s|     |s|                   |s|
 *      |s|             |s| |s|                           |s|
 *      |s|                 |s|
 *                          |s|
 *
 *  Each task is simply a sequence of steps. Heavier tasks have longer sequences.
 *  When a step of a task is executed, we move tasks to pending queue. And take another from the queue's head.
 *  With these architecture all small merges / mutations will be executed faster, than bigger ones.
 *
 *  We use boost::circular_buffer as a container for queues not to do any allocations.
 *
 *  Another nuisance that we faces with is than background operations always interact with an associated Storage.
 *  So, when a Storage want to shutdown, it must wait until all its background operaions are finished.
 */
class MergeTreeBackgroundExecutor : public shared_ptr_helper<MergeTreeBackgroundExecutor>
{
public:

    using CountGetter = std::function<size_t()>;

    enum class Type
    {
        MERGE_MUTATE,
        FETCH,
        MOVE
    };

    MergeTreeBackgroundExecutor(
        Type type_,
        CountGetter && threads_count_getter_,
        CountGetter && max_task_count_getter_,
        CurrentMetrics::Metric metric_)
        : type(type_)
        , threads_count_getter(threads_count_getter_)
        , max_task_count_getter(max_task_count_getter_)
        , metric(metric_)
    {
        name = toString(type);

        updateConfiguration();
    }

    ~MergeTreeBackgroundExecutor()
    {
        wait();
    }

    bool trySchedule(ExecutableTaskPtr task);

    void removeTasksCorrespondingToStorage(StorageID id);

    void wait();

    size_t activeCount()
    {
        std::lock_guard lock(mutex);
        return active.size();
    }

    size_t pendingCount()
    {
        std::lock_guard lock(mutex);
        return pending.size();
    }

private:

    void updateConfiguration();

    static String toString(Type type);

    Type type;
    String name;
    CountGetter threads_count_getter;
    CountGetter max_task_count_getter;
    CurrentMetrics::Metric metric;

    size_t threads_count{0};
    size_t max_tasks_count{0};

    AtomicStopwatch update_timer;

    /**
     * Has RAII class to determine how many tasks are waiting for the execution and executing at the moment.
     * Also has some flags and primitives to wait for current task to be executed.
     */
    struct TaskRuntimeData
    {
        TaskRuntimeData(ExecutableTaskPtr && task_, CurrentMetrics::Metric metric_)
            : task(std::move(task_))
            , increment(std::move(metric_))
        {}

        ExecutableTaskPtr task;
        CurrentMetrics::Increment increment;
        std::atomic_bool is_currently_deleting{false};
        /// Actually autoreset=false is needed only for unit test
        /// where multiple threads could remove tasks corresponding to the same storage
        /// This scenario in not possible in reality.
        Poco::Event is_done{/*autoreset=*/false};
    };

    using TaskRuntimeDataPtr = std::shared_ptr<TaskRuntimeData>;

    void routine(TaskRuntimeDataPtr item);

    /// Number all the threads in ThreadPool. To be able to lower the number of threads in runtime.
    void threadFunction(size_t number);

    /// Initially it will be empty
    boost::circular_buffer<TaskRuntimeDataPtr> pending{0};
    boost::circular_buffer<TaskRuntimeDataPtr> active{0};

    std::mutex mutex;
    std::condition_variable has_tasks;

    std::atomic_bool shutdown{false};

    ThreadPool pool;
};

}
