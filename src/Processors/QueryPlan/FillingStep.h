#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>
#include <Core/SortDescription.h>

namespace DB
{

/// Implements modifier WITH FILL of ORDER BY clause. See FillingTransform.
class FillingStep : public ITransformingStep
{
public:
    FillingStep(const DataStream & input_stream_, SortDescription sort_description_);

    String getName() const override { return "Filling"; }

    void transformPipeline(QueryPipeline & pipeline) override;

private:
    SortDescription sort_description;
};

}
