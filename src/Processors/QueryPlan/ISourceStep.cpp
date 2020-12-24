#include <Processors/QueryPlan/ISourceStep.h>
#include <Processors/QueryPipeline.h>

namespace DB
{

ISourceStep::ISourceStep(DataStream output_stream_)
{
    output_stream = std::move(output_stream_);
}

QueryPipelinePtr ISourceStep::updatePipeline(QueryPipelines)
{
    auto pipeline = std::make_unique<QueryPipeline>();
    QueryPipelineProcessorsCollector collector(*pipeline, this);
    initializePipeline(*pipeline);
    auto added_processors = collector.detachProcessors();
    processors.insert(processors.end(), added_processors.begin(), added_processors.end());
    return pipeline;
}

void ISourceStep::describePipeline(QueryPlanStepFormatSettings & settings) const
{
    IQueryPlanStep::describePipeline(processors, settings);
}

}
