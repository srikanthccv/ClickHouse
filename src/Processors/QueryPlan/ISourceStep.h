#pragma once
#include <Processors/QueryPlan/IQueryPlanStep.h>

namespace DB
{

/// Step which takes empty pipeline and initializes it. Returns single logical DataStream.
class ISourceStep : public IQueryPlanStep
{
public:
    explicit ISourceStep(DataStream output_stream_);

    String getName() const override { return "ISourceStep"; }

    QueryPipelineBuilderPtr updatePipeline(QueryPipelineBuilders pipelines, const BuildQueryPipelineSettings & settings) override;

    virtual void initializePipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings & settings) = 0;

    void describePipeline(FormatSettings & settings) const override;
};

}
