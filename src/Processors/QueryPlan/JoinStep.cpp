#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPipeline.h>
#include <Processors/Transforms/JoiningTransform.h>
#include <Interpreters/IJoin.h>

namespace DB
{

JoinStep::JoinStep(
    const DataStream & left_stream_,
    const DataStream & right_stream_,
    JoinPtr join_,
    size_t max_block_size_)
    : IQueryPlanStep()
    , join(std::move(join_))
    , max_block_size(max_block_size_)
{
    input_streams = {left_stream_, right_stream_};
    output_stream = DataStream
    {
        .header = JoiningTransform::transformHeader(left_stream_.header, join),
    };
}

QueryPipelinePtr JoinStep::updatePipeline(QueryPipelines pipelines, const BuildQueryPipelineSettings &)
{
    if (pipelines.size() != 2)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "JoinStep expect two input steps");

    return QueryPipeline::joinPipelines(std::move(pipelines[0]), std::move(pipelines[1]), join, max_block_size, &processors);
}

void JoinStep::describePipeline(FormatSettings & settings) const
{
    IQueryPlanStep::describePipeline(processors, settings);
}

static ITransformingStep::Traits getStorageJoinTraits()
{
    return ITransformingStep::Traits
    {
        {
            .preserves_distinct_columns = false,
            .returns_single_stream = false,
            .preserves_number_of_streams = true,
            .preserves_sorting = false,
        },
        {
            .preserves_number_of_rows = false,
        }
    };
}

StorageJoinStep::StorageJoinStep(const DataStream & input_stream_, JoinPtr join_, size_t max_block_size_)
    : ITransformingStep(
        input_stream_,
        JoiningTransform::transformHeader(input_stream_.header, join_),
        getStorageJoinTraits())
    , join(std::move(join_))
    , max_block_size(max_block_size_)
{
    if (!join->isStorageJoin())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "StorageJoinStep expects StorageJoin");
}

void StorageJoinStep::transformPipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings &)
{
    bool default_totals = false;
    if (!pipeline.hasTotals() && join->hasTotals())
    {
        pipeline.addDefaultTotals();
        default_totals = true;
    }

    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type)
    {
        bool on_totals = stream_type == QueryPipeline::StreamType::Totals;
        return std::make_shared<JoiningTransform>(header, join, max_block_size, on_totals, default_totals);
    });
}

}
