#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsCommon.h>
#include <Columns/ColumnConst.h>
#include <Interpreters/ExpressionActions.h>
#include <Common/typeid_cast.h>

#include <DataStreams/FilterBlockInputStream.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_COLUMN_FOR_FILTER;
}


FilterBlockInputStream::FilterBlockInputStream(const BlockInputStreamPtr & input, const ExpressionActionsPtr & expression_, ssize_t filter_column_)
    : expression(expression_), filter_column(filter_column_)
{
    children.push_back(input);
}

FilterBlockInputStream::FilterBlockInputStream(const BlockInputStreamPtr & input, const ExpressionActionsPtr & expression_, const String & filter_column_name_)
    : expression(expression_), filter_column(-1), filter_column_name(filter_column_name_)
{
    children.push_back(input);
}


String FilterBlockInputStream::getName() const { return "Filter"; }


String FilterBlockInputStream::getID() const
{
    std::stringstream res;
    res << "Filter(" << children.back()->getID() << ", " << expression->getID() << ", " << filter_column << ", " << filter_column_name << ")";
    return res.str();
}


const Block & FilterBlockInputStream::getTotals()
{
    if (IProfilingBlockInputStream * child = dynamic_cast<IProfilingBlockInputStream *>(&*children.back()))
    {
        totals = child->getTotals();
        expression->executeOnTotals(totals);
    }

    return totals;
}


static void analyzeConstantFilter(const IColumn & column, bool & filter_always_false, bool & filter_always_true)
{
    if (column.onlyNull())
    {
        filter_always_false = true;
    }
    else if (column.isColumnConst())
    {
        if (static_cast<const ColumnConst &>(column).getValue<UInt8>())
            filter_always_true = true;
        else
            filter_always_false = true;
    }
}


Block FilterBlockInputStream::readImpl()
{
    Block res;

    if (is_first)
    {
        is_first = false;

        const Block & sample_block = expression->getSampleBlock();

        /// Find the current position of the filter column in the block.
        /** sample_block has the result structure of evaluating the expression.
          * But this structure does not necessarily match expression->execute(res) below,
          *  because the expression can be applied to a block that also contains additional,
          *  columns unnecessary for this expression, but needed later, in the next stages of the query execution pipeline.
          * There will be no such columns in sample_block.
          * Therefore, the position of the filter column in it can be different.
          */
        ssize_t filter_column_in_sample_block = filter_column;
        if (filter_column_in_sample_block == -1)
            filter_column_in_sample_block = sample_block.getPositionByName(filter_column_name);

        /// Let's check if the filter column is a constant containing 0 or 1.
        ColumnPtr column = sample_block.safeGetByPosition(filter_column_in_sample_block).column;

        if (column)
            analyzeConstantFilter(*column, filter_always_false, filter_always_true);

        if (filter_always_false)
            return res;
    }

    /// Until non-empty block after filtering or end of stream.
    while (1)
    {
        res = children.back()->read();
        if (!res)
            return res;

        expression->execute(res);

        if (filter_always_true)
            return res;

        /// Find the current position of the filter column in the block.
        if (filter_column == -1)
            filter_column = res.getPositionByName(filter_column_name);

        size_t columns = res.columns();
        ColumnPtr column = res.safeGetByPosition(filter_column).column;

        IColumn * observed_column = column.get();
        bool is_nullable_column = observed_column->isColumnNullable();
        if (is_nullable_column)
            observed_column = &static_cast<const ColumnNullable &>(*column.get()).getNestedColumn();

        ColumnUInt8 * column_vec = typeid_cast<ColumnUInt8 *>(observed_column);
        if (!column_vec)
        {
            /** It happens that at the stage of analysis of expressions (in sample_block) the columns-constants have not been calculated yet,
              *  and now - are calculated. That is, not all cases are covered by the code above.
              * This happens if the function returns a constant for a non-constant argument.
              * For example, `ignore` function.
              */
            analyzeConstantFilter(*observed_column, filter_always_false, filter_always_true);

            if (filter_always_false)
            {
                res.clear();
                return res;
            }

            if (filter_always_true)
                return res;

            throw Exception("Illegal type " + column->getName() + " of column for filter. Must be ColumnUInt8 or ColumnConstUInt8 or Nullable variants of them.",
                ErrorCodes::ILLEGAL_TYPE_OF_COLUMN_FOR_FILTER);
        }

        IColumn::Filter & filter = column_vec->getData();

        if (is_nullable_column)
        {
            /// Exclude the entries of the filter column that actually are NULL values.

            const NullMap & null_map = static_cast<ColumnNullable &>(*column).getNullMapData();

            IColumn::Filter & filter = column_vec->getData();
            for (size_t i = 0, size = null_map.size(); i < size; ++i)
                if (null_map[i])
                    filter[i] = 0;
        }

        /** Let's find out how many rows will be in result.
          * To do this, we filter out the first non-constant column
          *  or calculate number of set bytes in the filter.
          */
        size_t first_non_constant_column = 0;
        for (size_t i = 0; i < columns; ++i)
        {
            if (!res.safeGetByPosition(i).column->isColumnConst())
            {
                first_non_constant_column = i;

                if (first_non_constant_column != static_cast<size_t>(filter_column))
                    break;
            }
        }

        size_t filtered_rows = 0;
        if (first_non_constant_column != static_cast<size_t>(filter_column))
        {
            ColumnWithTypeAndName & current_column = res.safeGetByPosition(first_non_constant_column);
            current_column.column = current_column.column->filter(filter, -1);
            filtered_rows = current_column.column->size();
        }
        else
        {
            filtered_rows = countBytesInFilter(filter);
        }

        /// If the current block is completely filtered out, let's move on to the next one.
        if (filtered_rows == 0)
            continue;

        /// If all the rows pass through the filter.
        if (filtered_rows == filter.size())
        {
            /// Replace the column with the filter by a constant.
            res.safeGetByPosition(filter_column).column = res.safeGetByPosition(filter_column).type->createColumnConst(filtered_rows, UInt64(1));
            /// No need to touch the rest of the columns.
            return res;
        }

        /// Filter the rest of the columns.
        for (size_t i = 0; i < columns; ++i)
        {
            ColumnWithTypeAndName & current_column = res.safeGetByPosition(i);

            if (i == static_cast<size_t>(filter_column))
            {
                /// The column with filter itself is replaced with a column with a constant `1`, since after filtering, nothing else will remain.
                /// NOTE User could pass column with something different than 0 and 1 for filter.
                /// Example:
                ///  SELECT materialize(100) AS x WHERE x
                /// will work incorrectly.
                current_column.column = current_column.type->createColumnConst(filtered_rows, UInt64(1));
                continue;
            }

            if (i == first_non_constant_column)
                continue;

            if (current_column.column->isColumnConst())
                current_column.column = current_column.column->cut(0, filtered_rows);
            else
                current_column.column = current_column.column->filter(filter, -1);
        }

        return res;
    }
}


}
