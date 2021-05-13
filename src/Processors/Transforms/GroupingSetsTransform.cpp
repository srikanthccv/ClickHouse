#include <Processors/Transforms/GroupingSetsTransform.h>
#include <Processors/Transforms/TotalsHavingTransform.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

GroupingSetsTransform::GroupingSetsTransform(Block header, AggregatingTransformParamsPtr params_)
    : IAccumulatingTransform(std::move(header), params_->getHeader())
    , params(std::move(params_))
    , keys(params->params.keys)
{
//    if (keys.size() >= 8 * sizeof(mask))
//        throw Exception("Too many keys are used for CubeTransform.", ErrorCodes::LOGICAL_ERROR);
}

Chunk GroupingSetsTransform::merge(Chunks && chunks, bool final)
{
    LOG_DEBUG(log, "merge {} blocks", chunks.size());
    BlocksList rollup_blocks;
    for (auto & chunk : chunks)
        rollup_blocks.emplace_back(getInputPort().getHeader().cloneWithColumns(chunk.detachColumns()));

    auto rollup_block = params->aggregator.mergeBlocks(rollup_blocks, final);
    auto num_rows = rollup_block.rows();
    return Chunk(rollup_block.getColumns(), num_rows);
}

void GroupingSetsTransform::consume(Chunk chunk)
{
    consumed_chunks.emplace_back(std::move(chunk));
    LOG_DEBUG(log, "consumed block, now consumed_chunks size is {}", consumed_chunks.size());
}

Chunk GroupingSetsTransform::generate()
{
    LOG_DEBUG(log, "generate start, mask = {}", mask);
    if (!consumed_chunks.empty())
    {
        LOG_DEBUG(log, "consumed_chunks not empty, size is {}", consumed_chunks.size());
        if (consumed_chunks.size() > 1)
            grouping_sets_chunk = merge(std::move(consumed_chunks), false);
        else
            grouping_sets_chunk = std::move(consumed_chunks.front());

        consumed_chunks.clear();

        auto num_rows = grouping_sets_chunk.getNumRows();
        mask = (UInt64(1) << keys.size());
        LOG_DEBUG(log, "changed mask, mask = {}", mask);

        current_columns = grouping_sets_chunk.getColumns();
        current_zero_columns.clear();
        current_zero_columns.reserve(keys.size());

        for (auto key : keys)
            current_zero_columns.emplace_back(current_columns[key]->cloneEmpty()->cloneResized(num_rows));
    }

    // auto gen_chunk = std::move(cube_chunk);
    LOG_DEBUG(log, "before if mask");
    if (mask > 1)
    {
        LOG_DEBUG(log, "in if mask > 1");
        mask = mask >> 1;

        auto columns = current_columns;
        auto size = keys.size();
        for (size_t i = 0; i < size; ++i)
            /// Reverse bit order to support previous behaviour.
            if ((mask & (UInt64(1) << (size - i - 1))) == 0)
                columns[keys[i]] = current_zero_columns[i];

        Chunks chunks;
        chunks.emplace_back(std::move(columns), current_columns.front()->size());
        grouping_sets_chunk = merge(std::move(chunks), false);
    }
    LOG_DEBUG(log, "before gen_chunk");
    auto gen_chunk = std::move(grouping_sets_chunk);

    finalizeChunk(gen_chunk);
    return gen_chunk;
}

}
