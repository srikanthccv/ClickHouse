#include <Processors/Executors/PushingAsyncPipelineExecutor.h>
#include <Processors/Executors/PipelineExecutor.h>
#include <Processors/ISource.h>
#include <Processors/Chain.h>
#include <Processors/Sinks/ExceptionHandlingSink.h>
#include <iostream>

#include <Common/setThreadName.h>
#include <common/scope_guard_safe.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

class PushingAsyncSource : public ISource
{
public:
    explicit PushingAsyncSource(const Block & header)
        : ISource(header)
    {}

    String getName() const override { return "PushingAsyncSource"; }

    bool setData(Chunk chunk)
    {
        std::unique_lock lock(mutex);
        condvar.wait(lock, [this] { return !has_data || is_finished; });

        if (is_finished)
            return false;

        data.swap(chunk);
        has_data = true;
        condvar.notify_one();

        return true;
    }

    void finish()
    {
        is_finished = true;
        condvar.notify_all();
    }

protected:

    Chunk generate() override
    {
        std::unique_lock lock(mutex);
        condvar.wait(lock, [this] { return has_data || is_finished; });

        Chunk res;

        res.swap(data);
        has_data = false;
        condvar.notify_one();

        return res;
    }

private:
    Chunk data;
    bool has_data = false;
    std::atomic_bool is_finished = false;
    std::mutex mutex;
    std::condition_variable condvar;
};

struct PushingAsyncPipelineExecutor::Data
{
    PipelineExecutorPtr executor;
    std::exception_ptr exception;
    PushingAsyncSource * source = nullptr;
    std::atomic_bool is_finished = false;
    std::atomic_bool has_exception = false;
    ThreadFromGlobalPool thread;
    Poco::Event finish_event;

    ~Data()
    {
        if (thread.joinable())
            thread.join();
    }

    void rethrowExceptionIfHas()
    {
        if (has_exception)
        {
            has_exception = false;
            std::rethrow_exception(std::move(exception));
        }
    }
};

static void threadFunction(PushingAsyncPipelineExecutor::Data & data, ThreadGroupStatusPtr thread_group, size_t num_threads)
{
    setThreadName("QueryPipelineEx");

    try
    {
        if (thread_group)
            CurrentThread::attachTo(thread_group);

        SCOPE_EXIT_SAFE(
            if (thread_group)
                CurrentThread::detachQueryIfNotDetached();
        );

        data.executor->execute(num_threads);
    }
    catch (...)
    {
        data.exception = std::current_exception();
        data.has_exception = true;

        /// Finish source in case of exception. Otherwise thread.join() may hung.
        if (data.source)
            data.source->finish();
    }

    data.is_finished = true;
    data.finish_event.set();
}


PushingAsyncPipelineExecutor::PushingAsyncPipelineExecutor(Chain & chain_) : chain(chain_)
{
    pushing_source = std::make_shared<PushingAsyncSource>(chain.getInputHeader());
    auto sink = std::make_shared<ExceptionHandlingSink>(chain.getOutputHeader());
    connect(pushing_source->getPort(), chain.getInputPort());
    connect(chain.getOutputPort(), sink->getPort());

    processors = std::make_unique<Processors>();
    processors->reserve(chain.getProcessors().size() + 2);
    for (const auto & processor : chain.getProcessors())
        processors->push_back(processor);
    processors->push_back(pushing_source);
    processors->push_back(std::move(sink));
}

PushingAsyncPipelineExecutor::~PushingAsyncPipelineExecutor()
{
    try
    {
        finish();
    }
    catch (...)
    {
        tryLogCurrentException("PushingAsyncPipelineExecutor");
    }
}

const Block & PushingAsyncPipelineExecutor::getHeader() const
{
    return pushing_source->getPort().getHeader();
}


void PushingAsyncPipelineExecutor::start()
{
    if (started)
        return;

    started = true;

    data = std::make_unique<Data>();
    data->executor = std::make_shared<PipelineExecutor>(*processors);
    data->source = pushing_source.get();

    auto func = [&, thread_group = CurrentThread::getGroup()]()
    {
        threadFunction(*data, thread_group, chain.getNumThreads());
    };

    data->thread = ThreadFromGlobalPool(std::move(func));
}

void PushingAsyncPipelineExecutor::push(Chunk chunk)
{
    if (!started)
        start();

    bool is_pushed = pushing_source->setData(std::move(chunk));
    data->rethrowExceptionIfHas();

    if (!is_pushed)
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Pipeline for PushingPipelineExecutor was finished before all data was inserted");
}

void PushingAsyncPipelineExecutor::push(Block block)
{
    push(Chunk(block.getColumns(), block.rows()));
}

void PushingAsyncPipelineExecutor::finish()
{
    if (finished)
        return;
    finished = true;

    pushing_source->finish();

    /// Join thread here to wait for possible exception.
    if (data && data->thread.joinable())
        data->thread.join();

    /// Rethrow exception to not swallow it in destructor.
    if (data)
        data->rethrowExceptionIfHas();
}

void PushingAsyncPipelineExecutor::cancel()
{
    /// Cancel execution if it wasn't finished.
    if (data && !data->is_finished && data->executor)
        data->executor->cancel();

    finish();
}

}
