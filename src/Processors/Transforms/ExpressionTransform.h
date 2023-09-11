#pragma once
#include <Processors/Transforms/ExceptionKeepingTransform.h>
#include <Processors/ISimpleTransform.h>

namespace DB
{

class ExpressionActions;
using ExpressionActionsPtr = std::shared_ptr<ExpressionActions>;

class ActionsDAG;

/** Executes a certain expression over the block.
  * The expression consists of column identifiers from the block, constants, common functions.
  * For example: hits * 2 + 3, url LIKE '%clickhouse%'
  * The expression processes each row independently of the others.
  */
class ExpressionTransform final : public ISimpleTransform
{
public:
    ExpressionTransform(
            const Block & header_,
            ExpressionActionsPtr expression_);

    String getName() const override { return "ExpressionTransform"; }

    static Block transformHeader(Block header, const ActionsDAG & expression);

    PartialResultStatus getPartialResultProcessorSupportStatus() const override { return PartialResultStatus::FullSupported; }

protected:
    void transform(Chunk & chunk) override;

    ProcessorPtr getPartialResultProcessor(const ProcessorPtr & current_processor, UInt64 partial_result_limit, UInt64 partial_result_duration_ms) override;

private:

    ExpressionActionsPtr expression;
};

class ConvertingTransform final : public ExceptionKeepingTransform
{
public:
    ConvertingTransform(
            const Block & header_,
            ExpressionActionsPtr expression_);

    String getName() const override { return "ConvertingTransform"; }

protected:
    void onConsume(Chunk chunk) override;
    GenerateResult onGenerate() override
    {
        GenerateResult res;
        res.chunk = std::move(cur_chunk);
        return res;
    }

private:
    ExpressionActionsPtr expression;
    Chunk cur_chunk;
};

}
