#include <Processors/Transforms/DistinctSortedTransform.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int SET_SIZE_LIMIT_EXCEEDED;
    extern const int LOGICAL_ERROR;
}

static void calcColumnPositionsInHeader(const Block& header, const Names & column_names, ColumnNumbers& column_positions)
{
    /// pre-calculate column positions to use during chunk transformation
    const size_t num_columns = column_names.empty() ? header.columns() : column_names.size();
    column_positions.clear();
    column_positions.reserve(num_columns);
    for (size_t i = 0; i < num_columns; ++i)
    {
        auto pos = column_names.empty() ? i : header.getPositionByName(column_names[i]);
        const auto & col = header.getByPosition(pos).column;
        if (col && !isColumnConst(*col))
            column_positions.emplace_back(pos);
    }
}

bool DistinctSortedTransform::isApplicable(const Block & header, const SortDescription & sort_description, const Names & column_names)
{
    if (sort_description.empty())
        return false;

    /// check if first sorted column in header
    const SortColumnDescription & column_sort_descr = sort_description.front();
    if (!header.has(column_sort_descr.column_name))
        return false;

    ColumnNumbers column_positions;
    calcColumnPositionsInHeader(header, column_names, column_positions);

    /// check if sorted column matches any DISTINCT column
    const auto pos = header.getPositionByName(column_sort_descr.column_name);
    return std::find(begin(column_positions), end(column_positions), pos) != column_positions.end();
}

DistinctSortedTransform::DistinctSortedTransform(
    const Block & header_,
    const SortDescription & sort_description_,
    const SizeLimits & set_size_limits_,
    UInt64 limit_hint_,
    const Names & columns)
    : ISimpleTransform(header_, header_, true)
    , header(header_)
    , sort_description(sort_description_)
    , column_names(columns)
    , limit_hint(limit_hint_)
    , set_size_limits(set_size_limits_)
{
    /// pre-calculate column positions to use during chunk transformation
    calcColumnPositionsInHeader(header, column_names, column_positions);
    column_ptrs.reserve(column_positions.size());

    /// pre-calculate DISTINCT column positions which form sort prefix of sort description
    sort_prefix_positions.reserve(sort_description.size());
    for (const auto & column_sort_descr : sort_description)
    {
        /// check if there is such column in header
        if (!header.has(column_sort_descr.column_name))
            break;

        /// check if sorted column position matches any DISTINCT column
        const auto pos = header.getPositionByName(column_sort_descr.column_name);
        if (std::find(begin(column_positions), end(column_positions), pos) == column_positions.end())
            break;

        sort_prefix_positions.emplace_back(pos);
    }
    if (sort_prefix_positions.empty())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR, "DistinctSortedTransform: columns have to form a sort prefix for provided sort description");

    sort_prefix_columns.reserve(sort_prefix_positions.size());
}

void DistinctSortedTransform::transform(Chunk & chunk)
{
    if (unlikely(!chunk.hasRows()))
        return;

    /// get DISTINCT columns from chunk
    column_ptrs.clear();
    for (const auto pos : column_positions)
    {
        const auto & column = chunk.getColumns()[pos];
        column_ptrs.emplace_back(column.get());
    }

    /// get DISTINCT columns from chunk which form sort prefix of sort description
    sort_prefix_columns.clear();
    for (const auto pos : sort_prefix_positions)
    {
        const auto & column = chunk.getColumns()[pos];
        sort_prefix_columns.emplace_back(column.get());
    }

    if (data.type == ClearableSetVariants::Type::EMPTY)
        data.init(ClearableSetVariants::chooseMethod(column_ptrs, key_sizes));

    const size_t rows = chunk.getNumRows();
    IColumn::Filter filter(rows);

    bool has_new_data = false;
    switch (data.type)
    {
        case ClearableSetVariants::Type::EMPTY:
            break;
            // clang-format off
#define M(NAME) \
        case ClearableSetVariants::Type::NAME: \
            has_new_data = buildFilter(*data.NAME, column_ptrs, sort_prefix_columns, filter, rows, data); \
            break;

        APPLY_FOR_SET_VARIANTS(M)
#undef M
            // clang-format on
    }

    /// Just go to the next block if there isn't any new record in the current one.
    if (!has_new_data)
    {
        chunk.clear();
        return;
    }

    if (!set_size_limits.check(data.getTotalRowCount(), data.getTotalByteCount(), "DISTINCT", ErrorCodes::SET_SIZE_LIMIT_EXCEEDED))
    {
        stopReading();
        chunk.clear();
        return;
    }

    /// Stop reading if we already reached the limit.
    if (limit_hint && data.getTotalRowCount() >= limit_hint)
        stopReading();

    prev_chunk.chunk = std::move(chunk);
    prev_chunk.clearing_hint_columns = std::move(sort_prefix_columns);

    const size_t all_columns = prev_chunk.chunk.getNumColumns();
    Chunk res_chunk;
    for (size_t i = 0; i < all_columns; ++i)
        res_chunk.addColumn(prev_chunk.chunk.getColumns().at(i)->filter(filter, -1));

    chunk = std::move(res_chunk);
}

template <typename Method>
bool DistinctSortedTransform::buildFilter(
    Method & method,
    const ColumnRawPtrs & columns,
    const ColumnRawPtrs & clearing_hint_columns,
    IColumn::Filter & filter,
    const size_t rows,
    ClearableSetVariants & variants) const
{
    typename Method::State state(columns, key_sizes, nullptr);

    /// Compare last row of previous block and first row of current block,
    /// If rows are NOT equal, we can clear HashSet
    if (!prev_chunk.clearing_hint_columns.empty()) /// it's not first chunk in stream
    {
        if (!rowsEqual(clearing_hint_columns, 0, prev_chunk.clearing_hint_columns, prev_chunk.chunk.getNumRows() - 1))
            method.data.clear();
    }

    bool has_new_data = false;
    { /// handle 0-indexed row to avoid index check in loop below
        const auto emplace_result = state.emplaceKey(method.data, 0, variants.string_pool);
        if (emplace_result.isInserted())
            has_new_data = true;
        filter[0] = emplace_result.isInserted();
    }
    for (size_t i = 1; i < rows; ++i)
    {
        /// Compare i-th row and i-1-th row,
        /// If rows are not equal, we can clear HashSet
        if (!rowsEqual(clearing_hint_columns, i, clearing_hint_columns, i - 1))
            method.data.clear();

        const auto emplace_result = state.emplaceKey(method.data, i, variants.string_pool);
        if (emplace_result.isInserted())
            has_new_data = true;

        /// Emit the record if there is no such key in the current set yet.
        /// Skip it otherwise.
        filter[i] = emplace_result.isInserted();
    }
    return has_new_data;
}

bool DistinctSortedTransform::rowsEqual(const ColumnRawPtrs & lhs, size_t n, const ColumnRawPtrs & rhs, size_t m)
{
    for (size_t column_index = 0, num_columns = lhs.size(); column_index < num_columns; ++column_index)
    {
        const auto & lhs_column = *lhs[column_index];
        const auto & rhs_column = *rhs[column_index];
        if (lhs_column.compareAt(n, m, rhs_column, 0) != 0) /// not equal
            return false;
    }
    return true;
}

}
