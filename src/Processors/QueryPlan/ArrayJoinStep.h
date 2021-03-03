#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>

namespace DB
{

class ArrayJoinAction;
using ArrayJoinActionPtr = std::shared_ptr<ArrayJoinAction>;

class Context;

class ArrayJoinStep : public ITransformingStep
{
public:
    explicit ArrayJoinStep(const DataStream & input_stream_, ArrayJoinActionPtr array_join_, const Context & context_);
    String getName() const override { return "ArrayJoin"; }

    void transformPipeline(QueryPipeline & pipeline) override;

    void describeActions(FormatSettings & settings) const override;

    void updateInputStream(DataStream input_stream, Block result_header);

    const ArrayJoinActionPtr & arrayJoin() const { return array_join; }

private:
    ArrayJoinActionPtr array_join;
    Block res_header;
    const Context & context;
};

}
