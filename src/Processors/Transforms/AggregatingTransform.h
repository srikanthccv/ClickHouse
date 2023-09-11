#pragma once
#include <Compression/CompressedReadBuffer.h>
#include <IO/ReadBufferFromFile.h>
#include <Interpreters/Aggregator.h>
#include <Processors/IAccumulatingTransform.h>
#include <Common/Stopwatch.h>
#include <Common/setThreadName.h>
#include <Common/scope_guard_safe.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>

namespace CurrentMetrics
{
    extern const Metric DestroyAggregatesThreads;
    extern const Metric DestroyAggregatesThreadsActive;
}

namespace DB
{

class AggregatedChunkInfo : public ChunkInfo
{
public:
    bool is_overflows = false;
    Int32 bucket_num = -1;
    UInt64 chunk_num = 0; // chunk number in order of generation, used during memory bound merging to restore chunks order
};

using AggregatorList = std::list<Aggregator>;
using AggregatorListPtr = std::shared_ptr<AggregatorList>;

using AggregatorList = std::list<Aggregator>;
using AggregatorListPtr = std::shared_ptr<AggregatorList>;

struct AggregatingTransformParams
{
    Aggregator::Params params;

    /// Each params holds a list of aggregators which are used in query. It's needed because we need
    /// to use a pointer of aggregator to proper destroy complex aggregation states on exception
    /// (See comments in AggregatedDataVariants). However, this pointer might not be valid because
    /// we can have two different aggregators at the same time due to mixed pipeline of aggregate
    /// projections, and one of them might gets destroyed before used.
    AggregatorListPtr aggregator_list_ptr;
    Aggregator & aggregator;
    bool final;

    AggregatingTransformParams(const Block & header, const Aggregator::Params & params_, bool final_)
        : params(params_)
        , aggregator_list_ptr(std::make_shared<AggregatorList>())
        , aggregator(*aggregator_list_ptr->emplace(aggregator_list_ptr->end(), header, params))
        , final(final_)
    {
    }

    AggregatingTransformParams(
        const Block & header, const Aggregator::Params & params_, const AggregatorListPtr & aggregator_list_ptr_, bool final_)
        : params(params_)
        , aggregator_list_ptr(aggregator_list_ptr_)
        , aggregator(*aggregator_list_ptr->emplace(aggregator_list_ptr->end(), header, params))
        , final(final_)
    {
    }

    Block getHeader() const { return aggregator.getHeader(final); }

    Block getCustomHeader(bool final_) const { return aggregator.getHeader(final_); }
};

struct ManyAggregatedData
{
    ManyAggregatedDataVariants variants;
    std::vector<std::unique_ptr<std::mutex>> mutexes;
    std::atomic<UInt32> num_finished = 0;

    explicit ManyAggregatedData(size_t num_threads = 0) : variants(num_threads), mutexes(num_threads)
    {
        for (auto & elem : variants)
            elem = std::make_shared<AggregatedDataVariants>();

        for (auto & mut : mutexes)
            mut = std::make_unique<std::mutex>();
    }

    ~ManyAggregatedData()
    {
        try
        {
            if (variants.size() <= 1)
                return;

            // Aggregation states destruction may be very time-consuming.
            // In the case of a query with LIMIT, most states won't be destroyed during conversion to blocks.
            // Without the following code, they would be destroyed in the destructor of AggregatedDataVariants in the current thread (i.e. sequentially).
            const auto pool = std::make_unique<ThreadPool>(
                CurrentMetrics::DestroyAggregatesThreads,
                CurrentMetrics::DestroyAggregatesThreadsActive,
                variants.size());

            for (auto && variant : variants)
            {
                if (variant->size() < 100'000) // some seemingly reasonable constant
                    continue;

                // It doesn't make sense to spawn a thread if the variant is not going to actually destroy anything.
                if (variant->aggregator)
                {
                    // variant is moved here and will be destroyed in the destructor of the lambda function.
                    pool->trySchedule(
                        [my_variant = std::move(variant), thread_group = CurrentThread::getGroup()]()
                        {
                            SCOPE_EXIT_SAFE(
                                if (thread_group)
                                    CurrentThread::detachFromGroupIfNotDetached();
                            );
                            if (thread_group)
                                CurrentThread::attachToGroupIfDetached(thread_group);

                            setThreadName("AggregDestruct");
                        });
                }
            }

            pool->wait();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
};

using AggregatingTransformParamsPtr = std::shared_ptr<AggregatingTransformParams>;
using ManyAggregatedDataPtr = std::shared_ptr<ManyAggregatedData>;

/** Aggregates the stream of blocks using the specified key columns and aggregate functions.
  * Columns with aggregate functions adds to the end of the block.
  * If final = false, the aggregate functions are not finalized, that is, they are not replaced by their value, but contain an intermediate state of calculations.
  * This is necessary so that aggregation can continue (for example, by combining streams of partially aggregated data).
  *
  * For every separate stream of data separate AggregatingTransform is created.
  * Every AggregatingTransform reads data from the first port till is is not run out, or max_rows_to_group_by reached.
  * When the last AggregatingTransform finish reading, the result of aggregation is needed to be merged together.
  * This task is performed by ConvertingAggregatedToChunksTransform.
  * Last AggregatingTransform expands pipeline and adds second input port, which reads from ConvertingAggregated.
  *
  * Aggregation data is passed by ManyAggregatedData structure, which is shared between all aggregating transforms.
  * At aggregation step, every transform uses it's own AggregatedDataVariants structure.
  * At merging step, all structures pass to ConvertingAggregatedToChunksTransform.
  */
class AggregatingTransform : public IProcessor
{
public:
    AggregatingTransform(Block header, AggregatingTransformParamsPtr params_);

    /// For Parallel aggregating.
    AggregatingTransform(
        Block header,
        AggregatingTransformParamsPtr params_,
        ManyAggregatedDataPtr many_data,
        size_t current_variant,
        size_t max_threads,
        size_t temporary_data_merge_threads,
        bool should_produce_results_in_order_of_bucket_number_ = true,
        bool skip_merging_ = false);
    ~AggregatingTransform() override;

    String getName() const override { return "AggregatingTransform"; }
    Status prepare() override;
    void work() override;
    Processors expandPipeline() override;

    PartialResultStatus getPartialResultProcessorSupportStatus() const override
    {
        /// Currently AggregatingPartialResultTransform support only single-thread aggregation without key.

        /// TODO: check that insert results from aggregator.prepareBlockAndFillWithoutKey return values without
        /// changing of the aggregator state when aggregation with keys will be supported in AggregatingPartialResultTransform.
        bool is_partial_result_supported = params->params.keys_size == 0 /// Aggregation without key.
                                    && many_data->variants.size() == 1; /// Use only one stream for aggregation.

        return is_partial_result_supported ? PartialResultStatus::FullSupported : PartialResultStatus::NotSupported;
    }

protected:
    void consume(Chunk chunk);

    ProcessorPtr getPartialResultProcessor(const ProcessorPtr & current_processor, UInt64 partial_result_limit, UInt64 partial_result_duration_ms) override;

private:
    /// To read the data that was flushed into the temporary data file.
    Processors processors;

    AggregatingTransformParamsPtr params;
    Poco::Logger * log = &Poco::Logger::get("AggregatingTransform");

    ColumnRawPtrs key_columns;
    Aggregator::AggregateColumns aggregate_columns;

    /** Used if there is a limit on the maximum number of rows in the aggregation,
     *   and if group_by_overflow_mode == ANY.
     *  In this case, new keys are not added to the set, but aggregation is performed only by
     *   keys that have already managed to get into the set.
     */
    bool no_more_keys = false;

    ManyAggregatedDataPtr many_data;
    AggregatedDataVariants & variants;
    size_t max_threads = 1;
    size_t temporary_data_merge_threads = 1;
    bool should_produce_results_in_order_of_bucket_number = true;
    bool skip_merging = false; /// If we aggregate partitioned data merging is not needed.

    /// TODO: calculate time only for aggregation.
    Stopwatch watch;

    UInt64 src_rows = 0;
    UInt64 src_bytes = 0;

    bool is_generate_initialized = false;
    bool is_consume_finished = false;
    bool is_pipeline_created = false;

    Chunk current_chunk;
    bool read_current_chunk = false;

    bool is_consume_started = false;

    friend class AggregatingPartialResultTransform;
    /// The mutex protects variables that are used for creating a snapshot of the current processor.
    /// The current implementation of AggregatingPartialResultTransform uses the 'is_generate_initialized' variable to check
    /// whether the processor has started sending data through the main pipeline, and the corresponding partial result processor should stop creating snapshots.
    /// Additionally, the mutex protects the 'params->aggregator' and 'many_data->variants' variables, which are used to get data from them for a snapshot.
    std::mutex snapshot_mutex;

    void initGenerate();
};

Chunk convertToChunk(const Block & block);

}
