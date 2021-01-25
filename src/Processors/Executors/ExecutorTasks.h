#pragma once

#include <Processors/Executors/ExecutingGraph.h>
#include <Processors/Executors/PollingQueue.h>
#include <Processors/Executors/ThreadsQueue.h>
#include <Processors/Executors/TasksQueue.h>

#include <stack>
#include <queue>

namespace DB
{

/// Context for each thread.
class ExecutionThreadContext
{
public:

    /// Things to stop execution to expand pipeline.
    struct ExpandPipelineTask
    {
        std::function<bool()> callback;
        size_t num_waiting_processing_threads = 0;
        std::mutex mutex;
        std::condition_variable condvar;

        explicit ExpandPipelineTask(std::function<bool()> callback_) : callback(callback_) {}
    };

private:
    /// Will store context for all expand pipeline tasks (it's easy and we don't expect many).
    /// This can be solved by using atomic shard ptr.
    std::list<ExpandPipelineTask> task_list;

    std::queue<ExecutingGraph::Node *> async_tasks;
    std::atomic_bool has_async_tasks = false;

    std::condition_variable condvar;
    std::mutex mutex;
    bool wake_flag = false;

    /// Currently processing node.
    ExecutingGraph::Node * node = nullptr;

    /// Exception from executing thread itself.
    std::exception_ptr exception;

public:
#ifndef NDEBUG
    /// Time for different processing stages.
    UInt64 total_time_ns = 0;
    UInt64 execution_time_ns = 0;
    UInt64 processing_time_ns = 0;
    UInt64 wait_time_ns = 0;
#endif

    const size_t thread_number;

    void wait(std::atomic_bool & finished);
    void wakeUp();

    bool hasTask() const { return node != nullptr; }
    void setTask(ExecutingGraph::Node * task) { node = task; }
    bool executeTask();
    uint64_t getProcessorID() const { return node->processors_id; }

    ExecutingGraph::Node * tryPopAsyncTask();
    void pushAsyncTask(ExecutingGraph::Node * async_task);
    bool hasAsyncTasks() const { return has_async_tasks; }

    ExpandPipelineTask & addExpandPipelineTask(std::function<bool()> callback) { return task_list.emplace_back(std::move(callback)); }

    std::unique_lock<std::mutex> lockStatus() const { return std::unique_lock(node->status_mutex); }

    void setException(std::exception_ptr exception_) { exception = std::move(exception_); }
    void rethrowExceptionIfHas();

    explicit ExecutionThreadContext(size_t thread_number_) : thread_number(thread_number_) {}
};

class ExecutorTasks
{
    std::atomic_bool finished = false;

    /// Queue with pointers to tasks. Each thread will concurrently read from it until finished flag is set.
    /// Stores processors need to be prepared. Preparing status is already set for them.
    TaskQueue<ExecutingGraph::Node> task_queue;

    /// Queue which stores tasks where processors returned Async status after prepare.
    /// If multiple threads are using, main thread will wait for async tasks.
    /// For single thread, will wait for async tasks only when task_queue is empty.
    PollingQueue async_task_queue;

    size_t num_threads = 0;
    size_t num_waiting_async_tasks = 0;

    ThreadsQueue threads_queue;
    std::mutex task_queue_mutex;

    std::atomic<size_t> num_processing_executors = 0;
    std::atomic<ExecutionThreadContext::ExpandPipelineTask *> expand_pipeline_task = nullptr;

    std::vector<std::unique_ptr<ExecutionThreadContext>> executor_contexts;
    std::mutex executor_contexts_mutex;

public:
    using Stack = std::stack<UInt64>;
    using Queue = std::queue<ExecutingGraph::Node *>;

    bool doExpandPipeline(ExecutionThreadContext::ExpandPipelineTask * task, bool processing);
    bool runExpandPipeline(size_t thread_number, std::function<bool()> callback);

    void expandPipelineStart();
    void expandPipelineEnd();

    void finish();
    bool isFinished() const { return finished; }

    void rethrowFirstThreadException();

    void tryGetTask(ExecutionThreadContext & context);
    void pushTasks(Queue & queue, Queue & async_queue, ExecutionThreadContext & context);

    void init(size_t num_threads_);
    void fill(Queue & queue);

    void processAsyncTasks();

    ExecutionThreadContext & getThreadContext(size_t thread_num) { return *executor_contexts[thread_num]; }
};

}
