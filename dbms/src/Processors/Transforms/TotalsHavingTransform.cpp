#include <Processors/Transforms/TotalsHavingTransform.h>
#include <Processors/Transforms/AggregatingTransform.h>

#include <Columns/ColumnAggregateFunction.h>
#include <Columns/FilterDescription.h>

#include <Common/typeid_cast.h>
#include <DataStreams/finalizeBlock.h>
#include <Interpreters/ExpressionActions.h>

namespace DB
{

static Chunk finalizeChunk(Chunk chunk)
{
    auto num_rows = chunk.getNumRows();
    auto columns = chunk.detachColumns();

    for (auto & column : columns)
        if (auto * agg_function = typeid_cast<const ColumnAggregateFunction *>(column.get()))
            column = agg_function->convertToValues();

    return Chunk(std::move(columns), num_rows);
}

static Block createOutputHeader(Block block)
{
    finalizeBlock(block);
    return block;
}

TotalsHavingTransform::TotalsHavingTransform(
    const Block & header,
    bool overflow_row_,
    const ExpressionActionsPtr & expression_,
    const std::string & filter_column_,
    TotalsMode totals_mode_,
    double auto_include_threshold_,
    bool final_)
    : ISimpleTransform(header, createOutputHeader(header), true)
    , overflow_row(overflow_row_)
    , expression(expression_)
    , filter_column_name(filter_column_)
    , totals_mode(totals_mode_)
    , auto_include_threshold(auto_include_threshold_)
    , final(final_)
    , arena(std::make_shared<Arena>())
{
    /// Port for Totals.
    outputs.emplace_back(outputs.front().getHeader(), this);

    filter_column_pos = outputs.front().getHeader().getPositionByName(filter_column_name);

    /// Initialize current totals with initial state.
    current_totals.reserve(header.columns());
    for (const auto & elem : header)
    {
        if (const auto * column = typeid_cast<const ColumnAggregateFunction *>(elem.column.get()))
        {
            /// Create ColumnAggregateFunction with initial aggregate function state.

            IAggregateFunction * function = column->getAggregateFunction().get();
            auto target = ColumnAggregateFunction::create(column->getAggregateFunction(), Arenas(1, arena));
            AggregateDataPtr data = arena->alignedAlloc(function->sizeOfData(), function->alignOfData());
            function->create(data);
            target->getData().push_back(data);
            current_totals.emplace_back(std::move(target));
        }
        else
        {
            /// Not an aggregate function state. Just create a column with default value.

            MutableColumnPtr new_column = elem.type->createColumn();
            elem.type->insertDefaultInto(*new_column);
            current_totals.emplace_back(std::move(new_column));
        }
    }
}

IProcessor::Status TotalsHavingTransform::prepare()
{
    if (!finished_transform)
    {
        auto status = ISimpleTransform::prepare();

        if (status != Status::Finished)
            return status;

        finished_transform = true;
    }

    auto & totals_output = getTotalsPort();

    /// Check can output.
    if (totals_output.isFinished())
        return Status::Finished;

    if (!totals_output.canPush())
        return Status::PortFull;

    if (!totals)
        return Status::Ready;

    totals_output.push(std::move(totals));
    totals_output.finish();
    return Status::Finished;
}

void TotalsHavingTransform::work()
{
    if (finished_transform)
        prepareTotals();
    else
        ISimpleTransform::work();
}

void TotalsHavingTransform::transform(Chunk & chunk)
{
    auto & info = chunk.getChunkInfo();
    if (!info)
        throw Exception("Chunk info was not set for chunk in MergingAggregatedTransform.", ErrorCodes::LOGICAL_ERROR);

    auto * agg_info = typeid_cast<const AggregatedChunkInfo *>(info.get());
    if (!agg_info)
        throw Exception("Chunk should have AggregatedChunkInfo in MergingAggregatedTransform.", ErrorCodes::LOGICAL_ERROR);


    /// Block with values not included in `max_rows_to_group_by`. We'll postpone it.
    if (overflow_row && agg_info->is_overflows)
    {
        overflow_aggregates = std::move(chunk);
        return;
    }

    if (!chunk)
        return;

    auto finalized = chunk;
    if (final)
        finalizeChunk(finalized);

    total_keys += finalized.getNumRows();

    if (filter_column_name.empty())
    {
        addToTotals(chunk, nullptr);
    }
    else
    {
        /// Compute the expression in HAVING.
        auto finalized_block = getOutputPort().getHeader().cloneWithColumns(finalized.detachColumns());
        expression->execute(finalized_block);
        auto columns = finalized_block.getColumns();

        ColumnPtr filter_column_ptr = finalized.getColumns()[filter_column_pos];
        ConstantFilterDescription const_filter_description(*filter_column_ptr);

        if (const_filter_description.always_true)
        {
            addToTotals(chunk, nullptr);
            return;
        }

        if (const_filter_description.always_false)
        {
            if (totals_mode == TotalsMode::BEFORE_HAVING)
                addToTotals(chunk, nullptr);

            chunk.clear();
            return;
        }

        FilterDescription filter_description(*filter_column_ptr);

        /// Add values to `totals` (if it was not already done).
        if (totals_mode == TotalsMode::BEFORE_HAVING)
            addToTotals(chunk, nullptr);
        else
            addToTotals(chunk, filter_description.data);

        /// Filter the block by expression in HAVING.
        for (auto & column : columns)
        {
            column = column->filter(*filter_description.data, -1);
            if (column->empty())
            {
                chunk.clear();
                return;
            }
        }

        auto num_rows = columns.front()->size();
        chunk.setColumns(std::move(columns), num_rows);
    }

    passed_keys += chunk.getNumRows();
}

void TotalsHavingTransform::addToTotals(const Chunk & chunk, const IColumn::Filter * filter)
{
    auto num_columns = chunk.getNumColumns();
    for (size_t col = 0; col < num_columns; ++col)
    {
        const auto & current = chunk.getColumns()[col];

        if (const auto * column = typeid_cast<const ColumnAggregateFunction *>(current.get()))
        {
            auto & target = typeid_cast<ColumnAggregateFunction &>(*current_totals[col]);
            IAggregateFunction * function = target.getAggregateFunction().get();
            AggregateDataPtr data = target.getData()[0];

            /// Accumulate all aggregate states into that value.

            const ColumnAggregateFunction::Container & vec = column->getData();
            size_t size = vec.size();

            if (filter)
            {
                for (size_t row = 0; row < size; ++row)
                    if ((*filter)[row])
                        function->merge(data, vec[row], arena.get());
            }
            else
            {
                for (size_t row = 0; row < size; ++row)
                    function->merge(data, vec[row], arena.get());
            }
        }
    }
}

void TotalsHavingTransform::prepareTotals()
{
    /// If totals_mode == AFTER_HAVING_AUTO, you need to decide whether to add aggregates to TOTALS for strings,
    /// not passed max_rows_to_group_by.
    if (overflow_aggregates)
    {
        if (totals_mode == TotalsMode::BEFORE_HAVING
            || totals_mode == TotalsMode::AFTER_HAVING_INCLUSIVE
            || (totals_mode == TotalsMode::AFTER_HAVING_AUTO
                && static_cast<double>(passed_keys) / total_keys >= auto_include_threshold))
            addToTotals(overflow_aggregates, nullptr);
    }

    totals = Chunk(std::move(current_totals), 1);
    finalizeChunk(totals);

    if (expression)
    {
        auto block = getOutputPort().getHeader().cloneWithColumns(totals.detachColumns());
        expression->execute(block);
        totals = Chunk(block.getColumns(), 1);
    }
}

}
