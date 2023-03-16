#include <Processors/Executors/PullingAsyncPipelineExecutor.h>
#include <Processors/Executors/PipelineExecutor.h>
#include <Processors/Formats/LazyOutputFormat.h>
#include <Processors/Transforms/AggregatingTransform.h>
#include <Processors/Sources/NullSource.h>
#include <QueryPipeline/QueryPipeline.h>
#include <QueryPipeline/ReadProgressCallback.h>
#include <Common/setThreadName.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

struct PullingAsyncPipelineExecutor::Data
{
    PipelineExecutorPtr executor;
    std::exception_ptr exception;
    LazyOutputFormat * lazy_format = nullptr;
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
            std::rethrow_exception(exception);
        }
    }
};

PullingAsyncPipelineExecutor::PullingAsyncPipelineExecutor(QueryPipeline & pipeline_) : pipeline(pipeline_)
{
    if (!pipeline.pulling())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Pipeline for PullingAsyncPipelineExecutor must be pulling");

    lazy_format = std::make_shared<LazyOutputFormat>(pipeline.output->getHeader());
    pipeline.complete(lazy_format);
}

PullingAsyncPipelineExecutor::~PullingAsyncPipelineExecutor()
{
    try
    {
        cancel();
    }
    catch (...)
    {
        tryLogCurrentException("PullingAsyncPipelineExecutor");
    }
}

const Block & PullingAsyncPipelineExecutor::getHeader() const
{
    return lazy_format->getPort(IOutputFormat::PortKind::Main).getHeader();
}

static void threadFunction(PullingAsyncPipelineExecutor::Data & data, ThreadGroupStatusPtr thread_group, size_t num_threads)
{
    SCOPE_EXIT_SAFE(
        if (thread_group)
            CurrentThread::detachQueryIfNotDetached();
    );
    setThreadName("QueryPullPipeEx");

    try
    {
        if (thread_group)
            CurrentThread::attachTo(thread_group);

        data.executor->execute(num_threads);
    }
    catch (...)
    {
        data.exception = std::current_exception();
        data.has_exception = true;

        /// Finish lazy format in case of exception. Otherwise thread.join() may hung.
        if (data.lazy_format)
            data.lazy_format->finalize();
    }

    data.is_finished = true;
    data.finish_event.set();
}


bool PullingAsyncPipelineExecutor::pull(Chunk & chunk, uint64_t milliseconds)
{
    if (!data)
    {
        data = std::make_unique<Data>();
        data->executor = std::make_shared<PipelineExecutor>(pipeline.processors, pipeline.process_list_element);
        data->executor->setReadProgressCallback(pipeline.getReadProgressCallback());
        data->lazy_format = lazy_format.get();

        auto func = [&, thread_group = CurrentThread::getGroup()]()
        {
            threadFunction(*data, thread_group, pipeline.getNumThreads());
        };

        data->thread = ThreadFromGlobalPool(std::move(func));
    }

    data->rethrowExceptionIfHas();

    bool is_execution_finished
        = !data->executor->checkTimeLimitSoft() || (lazy_format ? lazy_format->isFinished() : data->is_finished.load());

    if (is_execution_finished)
    {
        /// If lazy format is finished, we don't cancel pipeline but wait for main thread to be finished.
        data->is_finished = true;
        /// Wait thread and rethrow exception if any.
        cancel();
        return false;
    }

    if (lazy_format)
    {
        chunk = lazy_format->getChunk(milliseconds);
        data->rethrowExceptionIfHas();
        return true;
    }

    chunk.clear();

    if (milliseconds)
        data->finish_event.tryWait(milliseconds);
    else
        data->finish_event.wait();

    return true;
}

bool PullingAsyncPipelineExecutor::pull(Block & block, uint64_t milliseconds)
{
    Chunk chunk;

    if (!pull(chunk, milliseconds))
        return false;

    if (!chunk)
    {
        /// In case if timeout exceeded.
        block.clear();
        return true;
    }

    block = lazy_format->getPort(IOutputFormat::PortKind::Main).getHeader().cloneWithColumns(chunk.detachColumns());

    if (auto chunk_info = chunk.getChunkInfo())
    {
        if (const auto * agg_info = typeid_cast<const AggregatedChunkInfo *>(chunk_info.get()))
        {
            block.info.bucket_num = agg_info->bucket_num;
            block.info.is_overflows = agg_info->is_overflows;
        }
    }

    return true;
}

void PullingAsyncPipelineExecutor::cancel()
{
    if (!data)
        return;

    /// Cancel execution if it wasn't finished.
    cancelWithExceptionHandling([&]()
    {
        data->executor->cancel();
    });

    /// The following code is needed to rethrow exception from PipelineExecutor.
    /// It could have been thrown from pull(), but we will not likely call it again.

    /// Join thread here to wait for possible exception.
    if (data->thread.joinable())
        data->thread.join();

    /// Rethrow exception to not swallow it in destructor.
    data->rethrowExceptionIfHas();
}

void PullingAsyncPipelineExecutor::cancelReading()
{
    if (!data)
        return;

    /// Stop reading from source if pipeline wasn't finished.
    cancelWithExceptionHandling([&]()
    {
        data->executor->cancelReading();
    });
}

void PullingAsyncPipelineExecutor::cancelWithExceptionHandling(CancelFunc && cancel_func)
{
    try
    {
        if (!data->is_finished && data->executor)
            cancel_func();
    }
    catch (...)
    {
        /// Store exception only of during query execution there was no
        /// exception, since only one exception can be re-thrown.
        if (!data->has_exception)
        {
            data->exception = std::current_exception();
            data->has_exception = true;
        }
    }
}

Chunk PullingAsyncPipelineExecutor::getTotals()
{
    return lazy_format ? lazy_format->getTotals()
                       : Chunk();
}

Chunk PullingAsyncPipelineExecutor::getExtremes()
{
    return lazy_format ? lazy_format->getExtremes()
                       : Chunk();
}

Block PullingAsyncPipelineExecutor::getTotalsBlock()
{
    auto totals = getTotals();

    if (totals.empty())
        return {};

    const auto & header = lazy_format->getPort(IOutputFormat::PortKind::Totals).getHeader();
    return header.cloneWithColumns(totals.detachColumns());
}

Block PullingAsyncPipelineExecutor::getExtremesBlock()
{
    auto extremes = getExtremes();

    if (extremes.empty())
        return {};

    const auto & header = lazy_format->getPort(IOutputFormat::PortKind::Extremes).getHeader();
    return header.cloneWithColumns(extremes.detachColumns());
}

ProfileInfo & PullingAsyncPipelineExecutor::getProfileInfo()
{
    if (lazy_format)
        return lazy_format->getProfileInfo();

    static ProfileInfo profile_info;
    static std::once_flag flag;
    /// Calculate rows before limit here to avoid race.
    std::call_once(flag, []() { profile_info.getRowsBeforeLimit(); });

    return profile_info;
}

}
