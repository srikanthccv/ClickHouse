#pragma once

#include <Poco/Notification.h>
#include <Poco/NotificationQueue.h>
#include <Poco/Timestamp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <map>
#include <functional>
#include <boost/noncopyable.hpp>
#include <Common/ZooKeeper/Types.h>
#include <Common/CurrentThread.h>
#include <Common/ThreadPool.h>


namespace DB
{

class TaskNotification;


/** Executes functions scheduled at a specific point in time.
  * Basically all tasks are added in a queue and precessed by worker threads.
  *
  * The most important difference between this and BackgroundProcessingPool
  *  is that we have the guarantee that the same function is not executed from many workers in the same time.
  *
  * The usage scenario: instead starting a separate thread for each task,
  *  register a task in BackgroundSchedulePool and when you need to run the task,
  *  call schedule or scheduleAfter(duration) method.
  */
class BackgroundSchedulePool
{
public:
    class TaskInfo;
    using TaskInfoPtr = std::shared_ptr<TaskInfo>;
    using TaskFunc = std::function<void()>;
    using DelayedTasks = std::multimap<Poco::Timestamp, TaskInfoPtr>;

    class TaskInfo : public std::enable_shared_from_this<TaskInfo>, private boost::noncopyable
    {
    public:
        TaskInfo(BackgroundSchedulePool & pool_, const std::string & log_name_, const TaskFunc & function_);

        /// Schedule for execution as soon as possible (if not already scheduled).
        /// If the task was already scheduled with delay, the delay will be ignored.
        bool schedule();

        /// Schedule for execution after specified delay.
        bool scheduleAfter(size_t ms);

        /// Further attempts to schedule become no-op. Will wait till the end of the current execution of the task.
        void deactivate();

        void activate();

        /// Atomically activate task and schedule it for execution.
        bool activateAndSchedule();

        /// get Coordination::WatchCallback needed for notifications from ZooKeeper watches.
        Coordination::WatchCallback getWatchCallback();

    private:
        friend class TaskNotification;
        friend class BackgroundSchedulePool;

        void execute();

        void scheduleImpl(std::lock_guard<std::mutex> & schedule_mutex_lock);

        BackgroundSchedulePool & pool;
        std::string log_name;
        TaskFunc function;

        std::mutex exec_mutex;
        std::mutex schedule_mutex;

        /// Invariants:
        /// * If deactivated is true then scheduled, delayed and executing are all false.
        /// * scheduled and delayed cannot be true at the same time.
        bool deactivated = false;
        bool scheduled = false;
        bool delayed = false;
        bool executing = false;

        /// If the task is scheduled with delay, points to element of delayed_tasks.
        DelayedTasks::iterator iterator;
    };

    class TaskHolder
    {
    public:
        TaskHolder() = default;
        explicit TaskHolder(const TaskInfoPtr & task_info_) : task_info(task_info_) {}
        TaskHolder(const TaskHolder & other) = delete;
        TaskHolder(TaskHolder && other) noexcept = default;
        TaskHolder & operator=(const TaskHolder & other) noexcept = delete;
        TaskHolder & operator=(TaskHolder && other) noexcept = default;

        ~TaskHolder()
        {
            if (task_info)
                task_info->deactivate();
        }

        TaskInfo * operator->() { return task_info.get(); }
        const TaskInfo * operator->() const { return task_info.get(); }

    private:
        TaskInfoPtr task_info;
    };

    TaskHolder createTask(const std::string & log_name, const TaskFunc & function);

    size_t getNumberOfThreads() const { return size; }

    BackgroundSchedulePool(size_t size);
    ~BackgroundSchedulePool();

private:
    using Threads = std::vector<ThreadFromGlobalPool>;

    void threadFunction();
    void delayExecutionThreadFunction();

    /// Schedule task for execution after specified delay from now.
    void scheduleDelayedTask(const TaskInfoPtr & task_info, size_t ms, std::lock_guard<std::mutex> & task_schedule_mutex_lock);

    /// Remove task, that was scheduled with delay, from schedule.
    void cancelDelayedTask(const TaskInfoPtr & task_info, std::lock_guard<std::mutex> & task_schedule_mutex_lock);

    /// Number for worker threads.
    const size_t size;
    std::atomic<bool> shutdown {false};
    Threads threads;
    Poco::NotificationQueue queue;

    /// Delayed notifications.

    std::condition_variable wakeup_cond;
    std::mutex delayed_tasks_mutex;
    /// Thread waiting for next delayed task.
    ThreadFromGlobalPool delayed_thread;
    /// Tasks ordered by scheduled time.
    DelayedTasks delayed_tasks;

    /// Thread group used for profiling purposes
    ThreadGroupStatusPtr thread_group;

    void attachToThreadGroup();
};

}
