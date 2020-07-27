#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPipeline.h>
#include <Processors/Transforms/PartialSortingTransform.h>
#include <Processors/Transforms/LimitsCheckingTransform.h>
#include <IO/Operators.h>

namespace DB
{

static ITransformingStep::DataStreamTraits getTraits(size_t limit)
{
    return ITransformingStep::DataStreamTraits
    {
            .preserves_distinct_columns = true,
            .returns_single_stream = false,
            .preserves_number_of_streams = true,
            .preserves_number_of_rows = limit == 0,
            .preserves_sorting = false,
    };
}

PartialSortingStep::PartialSortingStep(
    const DataStream & input_stream,
    SortDescription sort_description_,
    UInt64 limit_,
    SizeLimits size_limits_)
    : ITransformingStep(input_stream, input_stream.header, getTraits(limit_))
    , sort_description(std::move(sort_description_))
    , limit(limit_)
    , size_limits(size_limits_)
{
    output_stream->sort_description = sort_description;
    output_stream->sort_mode = DataStream::SortMode::Chunk;
}

void PartialSortingStep::transformPipeline(QueryPipeline & pipeline)
{
    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
    {
        if (stream_type != QueryPipeline::StreamType::Main)
            return nullptr;

        return std::make_shared<PartialSortingTransform>(header, sort_description, limit);
    });

    IBlockInputStream::LocalLimits limits;
    limits.mode = IBlockInputStream::LIMITS_CURRENT;
    limits.size_limits = size_limits;

    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
    {
        if (stream_type != QueryPipeline::StreamType::Main)
            return nullptr;

        auto transform = std::make_shared<LimitsCheckingTransform>(header, limits);
        return transform;
    });
}

void PartialSortingStep::describeActions(FormatSettings & settings) const
{
    String prefix(settings.offset, ' ');
    settings.out << prefix << "Sort description: ";
    dumpSortDescription(sort_description, input_streams.front().header, settings.out);
    settings.out << '\n';
}

}
