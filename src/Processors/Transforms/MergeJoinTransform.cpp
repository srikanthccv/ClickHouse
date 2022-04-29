#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include <base/defines.h>
#include <base/logger_useful.h>
#include <base/types.h>
#include <boost/core/noncopyable.hpp>

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Core/SortCursor.h>
#include <Core/SortDescription.h>
#include <IO/WriteHelpers.h>
#include <Interpreters/TableJoin.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Processors/Transforms/MergeJoinTransform.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
    extern const int LOGICAL_ERROR;
}

using JoinKind = ASTTableJoin::Kind;

namespace
{

FullMergeJoinCursorPtr createCursor(const Block & block, const Names & columns)
{
    SortDescription desc;
    desc.reserve(columns.size());
    for (const auto & name : columns)
        desc.emplace_back(name);
    return std::make_unique<FullMergeJoinCursor>(materializeBlock(block), desc);
}

template <bool has_left_nulls, bool has_right_nulls>
int nullableCompareAt(const IColumn & left_column, const IColumn & right_column, size_t lhs_pos, size_t rhs_pos, int null_direction_hint = 1)
{
    if constexpr (has_left_nulls && has_right_nulls)
    {
        const auto * left_nullable = checkAndGetColumn<ColumnNullable>(left_column);
        const auto * right_nullable = checkAndGetColumn<ColumnNullable>(right_column);

        if (left_nullable && right_nullable)
        {
            int res = left_column.compareAt(lhs_pos, rhs_pos, right_column, null_direction_hint);
            if (res)
                return res;

            /// NULL != NULL case
            if (left_column.isNullAt(lhs_pos))
                return null_direction_hint;

            return 0;
        }
    }

    if constexpr (has_left_nulls)
    {
        if (const auto * left_nullable = checkAndGetColumn<ColumnNullable>(left_column))
        {
            if (left_column.isNullAt(lhs_pos))
                return null_direction_hint;
            return left_nullable->getNestedColumn().compareAt(lhs_pos, rhs_pos, right_column, null_direction_hint);
        }
    }

    if constexpr (has_right_nulls)
    {
        if (const auto * right_nullable = checkAndGetColumn<ColumnNullable>(right_column))
        {
            if (right_column.isNullAt(rhs_pos))
                return -null_direction_hint;
            return left_column.compareAt(lhs_pos, rhs_pos, right_nullable->getNestedColumn(), null_direction_hint);
        }
    }

    return left_column.compareAt(lhs_pos, rhs_pos, right_column, null_direction_hint);
}

int ALWAYS_INLINE compareCursors(const SortCursorImpl & lhs, size_t lpos,
                                 const SortCursorImpl & rhs, size_t rpos)
{
    for (size_t i = 0; i < lhs.sort_columns_size; ++i)
    {
        /// TODO(@vdimir): use nullableCompareAt only if there's nullable columns
        int cmp = nullableCompareAt<true, true>(*lhs.sort_columns[i], *rhs.sort_columns[i], lpos, rpos);
        if (cmp != 0)
            return cmp;
    }
    return 0;
}

int ALWAYS_INLINE compareCursors(const SortCursorImpl & lhs, const SortCursorImpl & rhs)
{
    return compareCursors(lhs, lhs.getRow(), rhs, rhs.getRow());
}

bool ALWAYS_INLINE totallyLess(SortCursorImpl & lhs, SortCursorImpl & rhs)
{
    /// The last row of left cursor is less than the current row of the right cursor.
    int cmp = compareCursors(lhs, lhs.rows - 1, rhs, rhs.getRow());
    return cmp < 0;
}

int ALWAYS_INLINE totallyCompare(SortCursorImpl & lhs, SortCursorImpl & rhs)
{
    if (totallyLess(lhs, rhs))
        return -1;
    if (totallyLess(rhs, lhs))
        return 1;
    return 0;
}

void addIndexColumn(const Columns & columns, ColumnUInt64 & indices, Chunk & result, size_t start, size_t limit)
{
    for (const auto & col : columns)
    {
        if (indices.empty())
        {
            result.addColumn(col->cut(start, limit));
        }
        else
        {
            if (limit == 0)
                limit = indices.size();

            assert(limit == indices.size());
            /// rows where default value should be inserted have index == size
            /// add row with defaults to handle it
            auto tmp_col = col->cloneResized(col->size() + 1);
            ColumnPtr new_col = tmp_col->index(indices, limit);
            result.addColumn(std::move(new_col));
        }
    }
}

bool sameNext(const SortCursorImpl & impl, std::optional<size_t> pos_opt = {})
{
    size_t pos = pos_opt.value_or(impl.getRow());
    for (size_t i = 0; i < impl.sort_columns_size; ++i)
    {
        const auto & col = *impl.sort_columns[i];
        if (auto cmp = col.compareAt(pos, pos + 1, col, impl.desc[i].nulls_direction); cmp != 0)
            return false;
    }
    return true;
}

size_t nextDistinct(SortCursorImpl & impl)
{
    assert(impl.isValid());
    size_t start_pos = impl.getRow();
    while (!impl.isLast() && sameNext(impl))
    {
        impl.next();
    }
    impl.next();

    if (impl.isValid())
        return impl.getRow() - start_pos;
    return impl.rows - start_pos;
}

}

const Chunk & FullMergeJoinCursor::getCurrent() const
{
    return current_chunk;
}

Chunk FullMergeJoinCursor::detach()
{
    cursor = SortCursorImpl();
    return std::move(current_chunk);
}

void FullMergeJoinCursor::setChunk(Chunk && chunk)
{
    assert(!recieved_all_blocks);
    assert(!cursor.isValid());

    if (!chunk)
    {
        recieved_all_blocks = true;
        detach();
        return;
    }

    current_chunk = std::move(chunk);
    cursor = SortCursorImpl(sample_block, current_chunk.getColumns(), desc);
}

bool FullMergeJoinCursor::fullyCompleted() const
{
    return !cursor.isValid() && recieved_all_blocks;
}

MergeJoinAlgorithm::MergeJoinAlgorithm(
    JoinPtr table_join_,
    const Blocks & input_headers,
    size_t max_block_size_)
    : table_join(table_join_)
    , max_block_size(max_block_size_)
    , log(&Poco::Logger::get("MergeJoinAlgorithm"))
{
    if (input_headers.size() != 2)
        throw Exception("MergeJoinAlgorithm requires exactly two inputs", ErrorCodes::LOGICAL_ERROR);

    auto strictness = table_join->getTableJoin().strictness();
    if (strictness != ASTTableJoin::Strictness::Any && strictness != ASTTableJoin::Strictness::All)
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "MergeJoinAlgorithm is not implemented for strictness {}", strictness);

    auto kind = table_join->getTableJoin().kind();
    if (!isInner(kind) && !isLeft(kind) && !isRight(kind) && !isFull(kind))
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "MergeJoinAlgorithm is not implemented for kind {}", kind);

    const auto & join_on = table_join->getTableJoin().getOnlyClause();

    if (join_on.on_filter_condition_left || join_on.on_filter_condition_right)
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "MergeJoinAlgorithm does not support ON filter conditions");

    cursors.push_back(createCursor(input_headers[0], join_on.key_names_left));
    cursors.push_back(createCursor(input_headers[1], join_on.key_names_right));
}

static ColumnPtr replicateRow(const IColumn & column, size_t num)
{
    MutableColumnPtr res = column.cloneEmpty();
    res->insertManyFrom(column, 0, num);
    return res;
}

template <typename TColumns>
static void copyColumnsResized(const TColumns & cols, size_t start, size_t size, Chunk & result_chunk)
{
    for (const auto & col : cols)
    {
        if (col->empty())
        {
            /// add defaults
            result_chunk.addColumn(col->cloneResized(size));
        }
        else if (col->size() == 1)
        {
            /// copy same row n times
            result_chunk.addColumn(replicateRow(*col, size));
        }
        else
        {
            /// cut column
            assert(start + size <= col->size());
            result_chunk.addColumn(col->cut(start, size));
        }
    }
}

static Chunk copyChunkResized(const Chunk & lhs, const Chunk & rhs, size_t start, size_t num_rows)
{
    Chunk result;
    copyColumnsResized(lhs.getColumns(), start, num_rows, result);
    copyColumnsResized(rhs.getColumns(), start, num_rows, result);
    return result;
}

static void prepareChunk(Chunk & chunk)
{
    if (!chunk)
        return;

    auto num_rows = chunk.getNumRows();
    auto columns = chunk.detachColumns();
    for (auto & column : columns)
        column = column->convertToFullColumnIfConst();

    chunk.setColumns(std::move(columns), num_rows);
}

void MergeJoinAlgorithm::initialize(Inputs inputs)
{
    if (inputs.size() != 2)
        throw Exception("MergeJoinAlgorithm requires exactly two inputs", ErrorCodes::LOGICAL_ERROR);

    LOG_DEBUG(log, "MergeJoinAlgorithm initialize, number of inputs: {}", inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        assert(inputs[i].chunk.getNumColumns() == cursors[i]->sampleBlock().columns());
        prepareChunk(inputs[i].chunk);
        copyColumnsResized(inputs[i].chunk.getColumns(), 0, 0, sample_chunks.emplace_back());
        consume(inputs[i], i);
    }
}

void MergeJoinAlgorithm::consume(Input & input, size_t source_num)
{
    if (input.skip_last_row)
        throw Exception("skip_last_row is not supported", ErrorCodes::NOT_IMPLEMENTED);

    if (input.permutation)
        throw DB::Exception("permutation is not supported", ErrorCodes::NOT_IMPLEMENTED);

    if (input.chunk)
    {
        stat.num_blocks[source_num] += 1;
        stat.num_rows[source_num] += input.chunk.getNumRows();
    }

    prepareChunk(input.chunk);
    cursors[source_num]->setChunk(std::move(input.chunk));
}

static Chunk getRowFromChunk(const Chunk & chunk, size_t pos)
{
    Chunk result;
    copyColumnsResized(chunk.getColumns(), pos, 1, result);
    return result;
}

static void ALWAYS_INLINE addRange(PaddedPODArray<UInt64> & left_map, size_t start, size_t end)
{
    assert(end > start);
    for (size_t i = start; i < end; ++i)
        left_map.push_back(i);
}

static void ALWAYS_INLINE addMany(PaddedPODArray<UInt64> & left_map, size_t idx, size_t num)
{
    for (size_t i = 0; i < num; ++i)
        left_map.push_back(idx);
}

template <JoinKind kind>
struct AllJoinImpl
{
    constexpr static bool enabled = isInner(kind) || isLeft(kind) || isRight(kind) || isFull(kind);

    static void join(FullMergeJoinCursor & left_cursor,
                     FullMergeJoinCursor & right_cursor,
                     size_t max_block_size,
                     PaddedPODArray<UInt64> & left_map,
                     PaddedPODArray<UInt64> & right_map,
                     std::unique_ptr<AllJoinState> & state)
    {
        right_map.clear();
        right_map.reserve(max_block_size);
        left_map.clear();
        left_map.reserve(max_block_size);

        size_t rpos = std::numeric_limits<size_t>::max();
        size_t lpos = std::numeric_limits<size_t>::max();
        int cmp = 0;
        assert(left_cursor->isValid() && right_cursor->isValid());
        while (left_cursor->isValid() && right_cursor->isValid())
        {
            lpos = left_cursor->getRow();
            rpos = right_cursor->getRow();

            cmp = compareCursors(left_cursor.cursor, right_cursor.cursor);
            if (cmp == 0)
            {
                size_t lnum = nextDistinct(left_cursor.cursor);
                size_t rnum = nextDistinct(right_cursor.cursor);

                bool all_fit_in_block = std::max(left_map.size(), right_map.size()) + lnum * rnum <= max_block_size;
                bool have_all_ranges = left_cursor.cursor.isValid() && right_cursor.cursor.isValid();
                if (all_fit_in_block && have_all_ranges)
                {
                    /// fast path if all joined rows fit in one block
                    for (size_t i = 0; i < rnum; ++i)
                    {
                        addRange(left_map, lpos, left_cursor.cursor.getRow());
                        addMany(right_map, rpos + i, lnum);
                    }
                }
                else
                {
                    assert(state == nullptr);
                    state = std::make_unique<AllJoinState>(left_cursor.cursor, lpos, right_cursor.cursor, rpos);
                    state->addRange(0, left_cursor.getCurrent().clone(), lpos, lnum);
                    state->addRange(1, right_cursor.getCurrent().clone(), rpos, rnum);
                    return;
                }
            }
            else if (cmp < 0)
            {
                size_t num = nextDistinct(left_cursor.cursor);
                if constexpr (isLeftOrFull(kind))
                {
                    right_map.resize_fill(right_map.size() + num, right_cursor->rows);
                    for (size_t i = lpos; i < left_cursor->getRow(); ++i)
                        left_map.push_back(i);
                }
            }
            else
            {
                size_t num = nextDistinct(right_cursor.cursor);
                if constexpr (isRightOrFull(kind))
                {
                    left_map.resize_fill(left_map.size() + num, left_cursor->rows);
                    for (size_t i = rpos; i < right_cursor->getRow(); ++i)
                        right_map.push_back(i);
                }
            }
        }
    }
};

template <template<JoinKind> class Impl, typename ... Args>
void dispatchKind(JoinKind kind, Args && ... args)
{
    if (Impl<JoinKind::Inner>::enabled && kind == JoinKind::Inner)
        return Impl<JoinKind::Inner>::join(std::forward<Args>(args)...);
    else if (Impl<JoinKind::Left>::enabled && kind == JoinKind::Left)
        return Impl<JoinKind::Left>::join(std::forward<Args>(args)...);
    else if (Impl<JoinKind::Right>::enabled && kind == JoinKind::Right)
        return Impl<JoinKind::Right>::join(std::forward<Args>(args)...);
    else if (Impl<JoinKind::Full>::enabled && kind == JoinKind::Full)
        return Impl<JoinKind::Full>::join(std::forward<Args>(args)...);
    else
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported join kind: \"{}\"", kind);

    __builtin_unreachable();
}

std::optional<MergeJoinAlgorithm::Status> MergeJoinAlgorithm::handleAllJoinState()
{
    if (all_join_state && all_join_state->finished())
    {
        all_join_state.reset();
    }

    if (all_join_state)
    {
        /// Accumulate blocks with same key in all_join_state
        for (size_t i = 0; i < cursors.size(); ++i)
        {
            if (cursors[i]->cursor.isValid() && all_join_state->keys[i].equals(cursors[i]->cursor))
            {
                size_t pos = cursors[i]->cursor.getRow();
                size_t num = nextDistinct(cursors[i]->cursor);
                all_join_state->addRange(i, cursors[i]->getCurrent().clone(), pos, num);
            }
        }

        for (size_t i = 0; i < cursors.size(); ++i)
        {
            if (!cursors[i]->cursor.isValid() && !cursors[i]->fullyCompleted())
            {
                return Status(i);
            }
        }
        /// If current position is valid, then we've found new key, can join accumulated data

        stat.max_blocks_loaded = std::max(stat.max_blocks_loaded, all_join_state->blocksStored());

        /// join all rows with current key
        MutableColumns result_cols;
        for (size_t i = 0; i < 2; ++i)
        {
            for (const auto & col : sample_chunks[i].getColumns())
                result_cols.push_back(col->cloneEmpty());
        }

        size_t total_rows = 0;
        while (total_rows < max_block_size)
        {
            const auto & left_range = all_join_state->getLeft();
            const auto & right_range = all_join_state->getRight();

            total_rows += left_range.length;

            size_t i = 0;
            /// Copy left block
            for (const auto & col : left_range.chunk.getColumns())
                result_cols[i++]->insertRangeFrom(*col, left_range.begin, left_range.length);
            /// And replicate current right column
            for (const auto & col : right_range.chunk.getColumns())
                result_cols[i++]->insertManyFrom(*col, right_range.current, left_range.length);
            bool valid = all_join_state->next();
            if (!valid)
                break;
        }

        if (total_rows)
            return Status(Chunk(std::move(result_cols), total_rows));
    }
    return {};
}

MergeJoinAlgorithm::Status MergeJoinAlgorithm::allJoin(JoinKind kind)
{
    auto left_map = ColumnUInt64::create();
    auto right_map = ColumnUInt64::create();
    size_t prev_pos[] = {cursors[0]->cursor.getRow(), cursors[1]->cursor.getRow()};

    dispatchKind<AllJoinImpl>(kind, *cursors[0], *cursors[1], max_block_size, left_map->getData(), right_map->getData(), all_join_state);

    assert(left_map->empty() || right_map->empty() || left_map->size() == right_map->size());
    size_t num_result_rows = std::max(left_map->size(), right_map->size());

    Chunk result;
    addIndexColumn(cursors[0]->getCurrent().getColumns(), *left_map, result, prev_pos[0], num_result_rows);
    addIndexColumn(cursors[1]->getCurrent().getColumns(), *right_map, result, prev_pos[1], num_result_rows);
    return Status(std::move(result));
}


template <JoinKind kind>
struct AnyJoinImpl
{
    constexpr static bool enabled = isInner(kind) || isLeft(kind) || isRight(kind);

    static void join(FullMergeJoinCursor & left_cursor,
                     FullMergeJoinCursor & right_cursor,
                     PaddedPODArray<UInt64> & left_map,
                     PaddedPODArray<UInt64> & right_map,
                     AnyJoinState & state)
    {
        assert(enabled);

        size_t num_rows = isLeft(kind) ? left_cursor->rowsLeft() :
                          isRight(kind) ? right_cursor->rowsLeft() :
                          std::min(left_cursor->rowsLeft(), right_cursor->rowsLeft());

        if constexpr (isLeft(kind) || isInner(kind))
            right_map.reserve(num_rows);

        if constexpr (isRight(kind) || isInner(kind))
            left_map.reserve(num_rows);

        size_t rpos = std::numeric_limits<size_t>::max();
        size_t lpos = std::numeric_limits<size_t>::max();
        assert(left_cursor->isValid() && right_cursor->isValid());
        int cmp = 0;
        while (left_cursor->isValid() && right_cursor->isValid())
        {
            lpos = left_cursor->getRow();
            rpos = right_cursor->getRow();

            cmp = compareCursors(left_cursor.cursor, right_cursor.cursor);
            if (cmp == 0)
            {
                if constexpr (isLeftOrFull(kind))
                {
                    size_t lnum = nextDistinct(left_cursor.cursor);
                    right_map.resize_fill(right_map.size() + lnum, rpos);
                }

                if constexpr (isRightOrFull(kind))
                {
                    size_t rnum = nextDistinct(right_cursor.cursor);
                    left_map.resize_fill(left_map.size() + rnum, lpos);
                }

                if constexpr (isInner(kind))
                {
                    nextDistinct(left_cursor.cursor);
                    nextDistinct(right_cursor.cursor);
                    left_map.emplace_back(lpos);
                    right_map.emplace_back(rpos);
                }
            }
            else if (cmp < 0)
            {
                size_t num = nextDistinct(left_cursor.cursor);
                if constexpr (isLeftOrFull(kind))
                    right_map.resize_fill(right_map.size() + num, right_cursor->rows);
            }
            else
            {
                size_t num = nextDistinct(right_cursor.cursor);
                if constexpr (isRightOrFull(kind))
                    left_map.resize_fill(left_map.size() + num, left_cursor->rows);
            }
        }

        /// Remember index of last joined row to propagate it to next block

        state.setValue({});
        if (!left_cursor->isValid())
        {
            state.set(0, left_cursor.cursor);
            if (cmp == 0 && isLeft(kind))
                state.setValue(getRowFromChunk(right_cursor.getCurrent(), rpos));
        }

        if (!right_cursor->isValid())
        {
            state.set(1, right_cursor.cursor);
            if (cmp == 0 && isRight(kind))
                state.setValue(getRowFromChunk(left_cursor.getCurrent(), lpos));
        }
    }
};

std::optional<MergeJoinAlgorithm::Status> MergeJoinAlgorithm::handleAnyJoinState()
{
    if (any_join_state.empty())
        return {};

    auto kind = table_join->getTableJoin().kind();

    Chunk result;

    for (size_t source_num = 0; source_num < 2; ++source_num)
    {
        auto & current = *cursors[source_num];
        auto & state = any_join_state;
        if (any_join_state.keys[source_num].equals(current.cursor))
        {
            size_t start_pos = current->getRow();
            size_t length = nextDistinct(current.cursor);
            if (length && isLeft(kind) && source_num == 0)
                result = copyChunkResized(current.getCurrent(), state.value ? state.value : sample_chunks[1 - source_num], start_pos, length);
            if (length && isRight(kind) && source_num == 1)
                result = copyChunkResized(state.value ? state.value : sample_chunks[1 - source_num], current.getCurrent(), start_pos, length);

            /// We've found row with other key, no need to skip more rows with current key
            if (current->isValid())
            {
                state.keys[source_num].reset();
            }
        }
        else
        {
            any_join_state.keys[source_num].reset();
        }
    }

    if (result)
        return Status(std::move(result));
    return {};
}

MergeJoinAlgorithm::Status MergeJoinAlgorithm::anyJoin(JoinKind kind)
{
    if (auto result = handleAnyJoinState())
        return std::move(*result);

    auto & current_left = cursors[0]->cursor;
    if (!current_left.isValid())
        return Status(0);

    auto & current_right = cursors[1]->cursor;
    if (!current_right.isValid())
        return Status(1);

    /// join doesn't build result block, but returns indices where result rows should be placed
    auto left_map = ColumnUInt64::create();
    auto right_map = ColumnUInt64::create();
    size_t prev_pos[] = {current_left.getRow(), current_right.getRow()};

    dispatchKind<AnyJoinImpl>(kind, *cursors[0], *cursors[1], left_map->getData(), right_map->getData(), any_join_state);

    assert(left_map->empty() || right_map->empty() || left_map->size() == right_map->size());
    size_t num_result_rows = std::max(left_map->size(), right_map->size());

    /// build result block from indices
    Chunk result;
    addIndexColumn(cursors[0]->getCurrent().getColumns(), *left_map, result, prev_pos[0], num_result_rows);
    addIndexColumn(cursors[1]->getCurrent().getColumns(), *right_map, result, prev_pos[1], num_result_rows);
    return Status(std::move(result));
}


String logRow(const SortCursorImpl & lhs, std::optional<size_t> pos = {})
{
    std::vector<String> text;
    size_t lhs_pos = pos.value_or(lhs.getRow());

    if (lhs_pos >= lhs.rows)
        return fmt::format("[{}/{}] : END", lhs_pos, lhs.rows);

    for (size_t i = 0; i < lhs.sort_columns_size; ++i)
    {
        if (const auto * left_nullable = checkAndGetColumn<ColumnNullable>(lhs.sort_columns[i]))
        {
            if (left_nullable->isNullAt(lhs_pos))
            {
                text.push_back("<NULL>");
            }
            else
            {
                Field f;
                left_nullable->getNestedColumn().get(lhs_pos, f);
                text.push_back(fmt::format("<{}>", f.dump()));
            }
        }
        else
        {
            Field f;
            lhs.sort_columns[i]->get(lhs_pos, f);
            text.push_back(fmt::format("{}", f.dump()));
        }
    }
    return fmt::format("[{}/{}] : ({})", lhs_pos, lhs.rows, fmt::join(text, ", "));
}


String dumpBlock(Block block, const Chunk & chunk = {})
{
    if (chunk)
    {
        assert(block.columns() == chunk.getNumColumns());
        block.setColumns(chunk.getColumns());
    }

    WriteBufferFromOwnString out;
    for (size_t i = 0; i < block.columns(); ++i)
    {
        const auto & col = block.safeGetByPosition(i);
        writeString(col.name, out);
        writeString(" :: ", out);
        writeString(col.type->getName(), out);
        writeString(" [", out);

        size_t limit = std::min<size_t>(col.column->size(), 10);
        for (size_t j = 0; j < limit; ++j)
        {
            if (j != 0)
                writeString(", ", out);
            {
                Field f;
                col.column->get(j, f);
                writeString(f.dump(), out);
            }

            // {
            //     FormatSettings fs;
            //     col.type->getDefaultSerialization()->serializeText(*col.column, j, out, fs);
            // }
        }
        if (limit < col.column->size())
            writeString("...", out);
        writeString("] ", out);
    }
    out.finalize();
    return out.str();
}

/// if `source_num == 0` get data from left cursor and fill defaults at right
/// otherwise vise versa
Chunk MergeJoinAlgorithm::createBlockWithDefaults(size_t source_num)
{
    size_t start = cursors[source_num]->cursor.getRow();
    size_t num_rows = cursors[source_num]->cursor.rowsLeft();

    ColumnRawPtrs cols;
    {
        const Chunk & chunk_left = source_num == 0 ? cursors[0]->getCurrent() : sample_chunks[0];
        for (const auto & col : chunk_left.getColumns())
            cols.push_back(col.get());

        const Chunk & chunk_right = source_num == 1 ? cursors[1]->getCurrent() : sample_chunks[1];
        for (const auto & col : chunk_right.getColumns())
            cols.push_back(col.get());
    }

    Chunk result_chunk;
    copyColumnsResized(cols, start, num_rows, result_chunk);
    cursors[source_num]->detach();
    return result_chunk;
}

IMergingAlgorithm::Status MergeJoinAlgorithm::merge()
{
    auto kind = table_join->getTableJoin().kind();

    if (!cursors[0]->cursor.isValid() && !cursors[0]->fullyCompleted())
        return Status(0);

    if (!cursors[1]->cursor.isValid() && !cursors[1]->fullyCompleted())
        return Status(1);

    if (auto result = handleAllJoinState())
        return std::move(*result);

    if (cursors[0]->fullyCompleted() || cursors[1]->fullyCompleted())
    {
        if (!cursors[0]->fullyCompleted() && isLeftOrFull(kind))
            return Status(createBlockWithDefaults(0));

        if (!cursors[1]->fullyCompleted() && isRightOrFull(kind))
            return Status(createBlockWithDefaults(1));

        return Status({}, true);
    }

    /// check if blocks are not intersecting at all
    if (int cmp = totallyCompare(cursors[0]->cursor, cursors[1]->cursor); cmp == 111)
    {
        if (cmp < 0)
        {
            if (isLeftOrFull(kind))
                return Status(createBlockWithDefaults(0));
            cursors[0]->detach();
            return Status(0);
        }

        if (cmp > 0)
        {
            if (isRightOrFull(kind))
                return Status(createBlockWithDefaults(1));
            cursors[1]->detach();
            return Status(1);
        }
    }

    auto strictness = table_join->getTableJoin().strictness();

    if (strictness == ASTTableJoin::Strictness::Any)
        return anyJoin(kind);

    if (strictness == ASTTableJoin::Strictness::All)
        return allJoin(kind);

    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported strictness '{}'", strictness);
}

MergeJoinTransform::MergeJoinTransform(
        JoinPtr table_join,
        const Blocks & input_headers,
        const Block & output_header,
        size_t max_block_size,
        UInt64 limit_hint)
    : IMergingTransform<MergeJoinAlgorithm>(input_headers, output_header, true, limit_hint, table_join, input_headers, max_block_size)
    , log(&Poco::Logger::get("MergeJoinTransform"))
{
    LOG_TRACE(log, "Use MergeJoinTransform");
}

void MergeJoinTransform::onFinish()
{
    algorithm.onFinish(total_stopwatch.elapsedSeconds());
}

}
