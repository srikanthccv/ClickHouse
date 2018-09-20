#include <DataStreams/CubeBlockInputStream.h>
#include <DataStreams/finalizeBlock.h>
#include <DataTypes/DataTypeAggregateFunction.h>
#include <Columns/ColumnAggregateFunction.h>
#include <Columns/FilterDescription.h>
#include <Common/typeid_cast.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_MANY_COLUMNS;
}

CubeBlockInputStream::CubeBlockInputStream(
    const BlockInputStreamPtr & input_, const Aggregator::Params & params_) : aggregator(params_),
        keys(params_.keys)
{
    if (keys.size() > 30)
        throw Exception("Too many columns for cube", ErrorCodes::TOO_MANY_COLUMNS);
        
    children.push_back(input_);
    Aggregator::CancellationHook hook = [this]() { return this->isCancelled(); };
    aggregator.setCancellationHook(hook);
}


Block CubeBlockInputStream::getHeader() const
{
    Block res = children.at(0)->getHeader();
    finalizeBlock(res);
    return res;
}


Block CubeBlockInputStream::readImpl()
{
    /** After reading a block from input stream,
      * we will calculate all subsets of columns on next iterations of readImpl
      * by zeroing columns at positions, where bits are zero in current bitmask.
      */

    if (mask) 
    {
        --mask;
        Block cube_block = source_block;
        for (size_t i = 0; i < keys.size(); ++i) 
        {
            if (!((mask >> i) & 1)) 
            {
                size_t pos = keys.size() - i - 1;
                auto & current = cube_block.getByPosition(keys[pos]);
                current.column = empty_block.getByPosition(pos).column->cloneResized(cube_block.rows()); 
            }
        }

        BlocksList cube_blocks = { cube_block };
        Block finalized = aggregator.mergeBlocks(cube_blocks, true);
        return finalized;
    }

    source_block = children[0]->read();
    empty_block = source_block.cloneEmpty();
    Block finalized = source_block;
    finalizeBlock(finalized);
    mask = (1 << keys.size()) - 1;

    return finalized;
}
}
