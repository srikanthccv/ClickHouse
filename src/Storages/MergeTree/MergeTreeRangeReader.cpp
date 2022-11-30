#include <Storages/MergeTree/IMergeTreeReader.h>
#include <Columns/FilterDescription.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnsCommon.h>
#include <Common/TargetSpecific.h>
#include <IO/WriteBufferFromString.h>
#include <IO/Operators.h>
#include <base/range.h>
#include <Interpreters/castColumn.h>
#include <DataTypes/DataTypeNothing.h>
#include <bit>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#    include <arm_neon.h>
#    ifdef HAS_RESERVED_IDENTIFIER
#        pragma clang diagnostic ignored "-Wreserved-identifier"
#    endif
#endif

namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int BAD_ARGUMENTS;
}


//static
void filterColumns(Columns & columns, const IColumn::Filter & filter)
{
    for (auto & column : columns)
    {
        if (column)
        {
            assert(column->size() == filter.size());

            column = column->filter(filter, -1);

            if (column->empty())
            {
                columns.clear();
                return;
            }
        }
    }
}

//static
void filterColumns(Columns & columns, const ColumnPtr & filter)
{
    ConstantFilterDescription const_descr(*filter);
    if (const_descr.always_true)
        return;

    if (const_descr.always_false)
    {
        for (auto & col : columns)
            if (col)
                col = col->cloneEmpty();

        return;
    }

    FilterDescription descr(*filter);
    filterColumns(columns, *descr.data);
}


size_t MergeTreeRangeReader::ReadResult::getLastMark(const MergeTreeRangeReader::ReadResult::RangesInfo & ranges)
{
    size_t current_task_last_mark = 0;
    for (const auto & mark_range : ranges)
        current_task_last_mark = std::max(current_task_last_mark, mark_range.range.end);
    return current_task_last_mark;
}


MergeTreeRangeReader::DelayedStream::DelayedStream(
    size_t from_mark,
    size_t current_task_last_mark_,
    IMergeTreeReader * merge_tree_reader_)
        : current_mark(from_mark), current_offset(0), num_delayed_rows(0)
        , current_task_last_mark(current_task_last_mark_)
        , merge_tree_reader(merge_tree_reader_)
        , index_granularity(&(merge_tree_reader->data_part_info_for_read->getIndexGranularity()))
        , continue_reading(false), is_finished(false)
//        , last_read_end(0) // track the offset where the last read from merge tree ended
{
}

size_t MergeTreeRangeReader::DelayedStream::position() const
{
    size_t num_rows_before_current_mark = index_granularity->getMarkStartingRow(current_mark);
    return num_rows_before_current_mark + current_offset + num_delayed_rows;
}

size_t MergeTreeRangeReader::DelayedStream::readRows(Columns & columns, size_t num_rows)
{
    if (num_rows)
    {
        size_t rows_read = merge_tree_reader->readRows(
            current_mark, current_task_last_mark, continue_reading, num_rows, columns);

//const size_t start_row = continue_reading ? last_read_end : index_granularity->getMarkStartingRow(current_mark);
//const size_t end_row = start_row + rows_read;
//std::cerr << "READ from " << start_row << " to " << end_row << "\n\n\n";

        continue_reading = true;
//        last_read_end = end_row;

        /// Zero rows_read maybe either because reading has finished
        ///  or because there is no columns we can read in current part (for example, all columns are default).
        /// In the last case we can't finish reading, but it's also ok for the first case
        ///  because we can finish reading by calculation the number of pending rows.
        if (0 < rows_read && rows_read < num_rows)
            is_finished = true;

        return rows_read;
    }

    return 0;
}

size_t MergeTreeRangeReader::DelayedStream::read(Columns & columns, size_t from_mark, size_t offset, size_t num_rows)
{
    size_t num_rows_before_from_mark = index_granularity->getMarkStartingRow(from_mark);
    /// We already stand accurately in required position,
    /// so because stream is lazy, we don't read anything
    /// and only increment amount delayed_rows
    if (position() == num_rows_before_from_mark + offset)
    {
        num_delayed_rows += num_rows;
        return 0;
    }
    else
    {
        size_t read_rows = finalize(columns);

        continue_reading = false;
        current_mark = from_mark;
        current_offset = offset;
        num_delayed_rows = num_rows;

        return read_rows;
    }
}

size_t MergeTreeRangeReader::DelayedStream::finalize(Columns & columns)
{
    /// We need to skip some rows before reading
    if (current_offset && !continue_reading)
    {
        for (size_t mark_num : collections::range(current_mark, index_granularity->getMarksCount()))
        {
            size_t mark_index_granularity = index_granularity->getMarkRows(mark_num);
            if (current_offset >= mark_index_granularity)
            {
                current_offset -= mark_index_granularity;
                current_mark++;
            }
            else
                break;

        }

        /// Skip some rows from begin of granule.
        /// We don't know size of rows in compressed granule,
        /// so have to read them and throw out.
        if (current_offset)
        {
            Columns tmp_columns;
            tmp_columns.resize(columns.size());
            readRows(tmp_columns, current_offset);
        }
    }

    size_t rows_to_read = num_delayed_rows;
    current_offset += num_delayed_rows;
    num_delayed_rows = 0;

    return readRows(columns, rows_to_read);
}


MergeTreeRangeReader::Stream::Stream(
        size_t from_mark, size_t to_mark, size_t current_task_last_mark, IMergeTreeReader * merge_tree_reader_)
        : current_mark(from_mark), offset_after_current_mark(0)
        , last_mark(to_mark)
        , merge_tree_reader(merge_tree_reader_)
        , index_granularity(&(merge_tree_reader->data_part_info_for_read->getIndexGranularity()))
        , current_mark_index_granularity(index_granularity->getMarkRows(from_mark))
        , stream(from_mark, current_task_last_mark, merge_tree_reader)
{
    size_t marks_count = index_granularity->getMarksCount();
    if (from_mark >= marks_count)
        throw Exception("Trying create stream to read from mark №"+ toString(current_mark) + " but total marks count is "
            + toString(marks_count), ErrorCodes::LOGICAL_ERROR);

    if (last_mark > marks_count)
        throw Exception("Trying create stream to read to mark №"+ toString(current_mark) + " but total marks count is "
            + toString(marks_count), ErrorCodes::LOGICAL_ERROR);
}

void MergeTreeRangeReader::Stream::checkNotFinished() const
{
    if (isFinished())
        throw Exception("Cannot read out of marks range.", ErrorCodes::BAD_ARGUMENTS);
}

void MergeTreeRangeReader::Stream::checkEnoughSpaceInCurrentGranule(size_t num_rows) const
{
    if (num_rows + offset_after_current_mark > current_mark_index_granularity)
        throw Exception("Cannot read from granule more than index_granularity.", ErrorCodes::LOGICAL_ERROR);
}

size_t MergeTreeRangeReader::Stream::readRows(Columns & columns, size_t num_rows)
{
    size_t rows_read = stream.read(columns, current_mark, offset_after_current_mark, num_rows);

    if (stream.isFinished())
        finish();

    return rows_read;
}

void MergeTreeRangeReader::Stream::toNextMark()
{
    ++current_mark;

    size_t total_marks_count = index_granularity->getMarksCount();
    if (current_mark < total_marks_count)
        current_mark_index_granularity = index_granularity->getMarkRows(current_mark);
    else if (current_mark == total_marks_count)
        current_mark_index_granularity = 0; /// HACK?
    else
        throw Exception("Trying to read from mark " + toString(current_mark) + ", but total marks count " + toString(total_marks_count), ErrorCodes::LOGICAL_ERROR);

    offset_after_current_mark = 0;
}

size_t MergeTreeRangeReader::Stream::read(Columns & columns, size_t num_rows, bool skip_remaining_rows_in_current_granule)
{
    checkEnoughSpaceInCurrentGranule(num_rows);

    if (num_rows)
    {
        checkNotFinished();

        size_t read_rows = readRows(columns, num_rows);

        offset_after_current_mark += num_rows;

        /// Start new granule; skipped_rows_after_offset is already zero.
        if (offset_after_current_mark == current_mark_index_granularity || skip_remaining_rows_in_current_granule)
            toNextMark();

        return read_rows;
    }
    else
    {
        /// Nothing to read.
        if (skip_remaining_rows_in_current_granule)
        {
            /// Skip the rest of the rows in granule and start new one.
            checkNotFinished();
            toNextMark();
        }

        return 0;
    }
}

void MergeTreeRangeReader::Stream::skip(size_t num_rows)
{
    if (num_rows)
    {
        checkNotFinished();
        checkEnoughSpaceInCurrentGranule(num_rows);

        offset_after_current_mark += num_rows;

        if (offset_after_current_mark == current_mark_index_granularity)
        {
            /// Start new granule; skipped_rows_after_offset is already zero.
            toNextMark();
        }
    }
}

size_t MergeTreeRangeReader::Stream::finalize(Columns & columns)
{
    size_t read_rows = stream.finalize(columns);

    if (stream.isFinished())
        finish();

    return read_rows;
}


void MergeTreeRangeReader::ReadResult::addGranule(size_t num_rows_)
{
    rows_per_granule.push_back(num_rows_);
    total_rows_per_granule += num_rows_;
}

void MergeTreeRangeReader::ReadResult::adjustLastGranule()
{
    size_t num_rows_to_subtract = total_rows_per_granule - num_read_rows;

    if (rows_per_granule.empty())
        throw Exception("Can't adjust last granule because no granules were added", ErrorCodes::LOGICAL_ERROR);

    if (num_rows_to_subtract > rows_per_granule.back())
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Can't adjust last granule because it has {} rows, but try to subtract {} rows.",
                        toString(rows_per_granule.back()), toString(num_rows_to_subtract));

    rows_per_granule.back() -= num_rows_to_subtract;
    total_rows_per_granule -= num_rows_to_subtract;
}

void MergeTreeRangeReader::ReadResult::clear()
{
    /// Need to save information about the number of granules.
    num_rows_to_skip_in_last_granule += rows_per_granule.back();
    rows_per_granule.assign(rows_per_granule.size(), 0);
    total_rows_per_granule = 0;
    final_filter = FilterWithCachedCount();
    num_rows = 0;
    columns.clear();
}

void MergeTreeRangeReader::ReadResult::clearFilter()
{
    // TODO: old version didn't clear filter_holder. WTF??????
    final_filter = FilterWithCachedCount();
}

void MergeTreeRangeReader::ReadResult::shrink(Columns & old_columns, const NumRows & rows_per_granule_previous) const
{
    for (auto & column : old_columns)
    {
        if (!column)
            continue;

        if (const auto * column_const = typeid_cast<const ColumnConst *>(column.get()))
        {
            column = column_const->cloneResized(total_rows_per_granule);
            continue;
        }

        LOG_TEST(log, "ReadResult::shrink() column size: {} total_rows_per_granule: {}",
            column->size(), total_rows_per_granule);

        auto new_column = column->cloneEmpty();
        new_column->reserve(total_rows_per_granule);
        for (size_t j = 0, pos = 0; j < rows_per_granule_previous.size(); pos += rows_per_granule_previous[j++])
        {
            if (rows_per_granule[j])
                new_column->insertRangeFrom(*column, pos, rows_per_granule[j]);
        }
        column = std::move(new_column);
    }
}

void MergeTreeRangeReader::ReadResult::checkInternalConsistency() const
{
    assert(!final_filter.present() || final_filter.size() == total_rows_per_granule);
    assert(!final_filter.present() || final_filter.countBytesInFilter() == num_rows
     || total_rows_per_granule == num_rows /*if filter has not been applied*/);
//    assert(!block_before_prewhere || block_before_prewhere.rows() == total_rows_per_granule);

    for (const auto & column : columns)
    {
        if (column)
            assert(column->size() == num_rows);
    }
}

std::string MergeTreeRangeReader::ReadResult::dumpInfo() const
{
    WriteBufferFromOwnString out;
    out << "num_rows: " << num_rows
        << ", columns: " << columns.size()
        << ", total_rows_per_granule: " << total_rows_per_granule
        << ", need_filter: " << need_filter;
    if (final_filter.present())
    {
        out << ", filter_size: " << final_filter.size()
        << ", filter_1s: " << final_filter.countBytesInFilter();
    }
    else
    {
        out << ", no filter";
    }
    for (size_t ci = 0; ci < columns.size(); ++ci)
    {
        out << ", column[" << ci << "]: ";
        if (!columns[ci])
            out << " nullptr";
        else
        {
            out << " " << columns[ci]->dumpStructure();
        }
    }
    if (block_before_prewhere)
    {
        out << ", block_before_prewhere: " << block_before_prewhere.dumpStructure();
    }
    return out.str();
}

#if 1
static void checkCombinedFiltersSize(size_t bytes_in_first_filter, size_t second_filter_size)
{
    if (bytes_in_first_filter != second_filter_size)
        throw Exception(ErrorCodes::LOGICAL_ERROR,
            "Cannot combine filters because number of bytes in a first filter ({}) "
            "does not match second filter size ({})", bytes_in_first_filter, second_filter_size);
}

/// Second filter size must be equal to number of 1s in the first filter.
/// The result size is equal to first filter size.
static ColumnPtr combineFilters(ColumnPtr first, ColumnPtr second)
{
    ConstantFilterDescription first_const_descr(*first);

    if (first_const_descr.always_true)
    {
        checkCombinedFiltersSize(first->size(), second->size());
        return second;
    }

    if (first_const_descr.always_false)
    {
        checkCombinedFiltersSize(0, second->size());
        return first;
    }

    FilterDescription first_descr(*first);

    size_t bytes_in_first_filter = countBytesInFilter(*first_descr.data);
    checkCombinedFiltersSize(bytes_in_first_filter, second->size());

    ConstantFilterDescription second_const_descr(*second);

    if (second_const_descr.always_true)
        return first;

    if (second_const_descr.always_false)
        return second->cloneResized(first->size());

    FilterDescription second_descr(*second);

    MutableColumnPtr mut_first;
    if (first_descr.data_holder)
        mut_first = IColumn::mutate(std::move(first_descr.data_holder));
    else
        mut_first = IColumn::mutate(std::move(first));

    auto & first_data = typeid_cast<ColumnUInt8 *>(mut_first.get())->getData();
    const auto * second_data = second_descr.data->data();

    for (auto & val : first_data)
    {
        if (val)
        {
            val = *second_data;
            ++second_data;
        }
    }

    return mut_first;
}
#endif

/*
ColumnPtr andFilters(ColumnPtr c1, ColumnPtr c2)
{
    // TODO: use proper vectorized implementation of AND?

    auto res = ColumnUInt8::create(c1->size());
    auto & res_data = res->getData();
    const auto & c1_data = dynamic_cast<const ColumnUInt8*>(c1.get())->getData();
    const auto & c2_data = dynamic_cast<const ColumnUInt8*>(c2.get())->getData();
    const size_t size = c1->size();
    const size_t step = 16;
    size_t i = 0;
    for (; i + step < size; i += step)
        for (size_t j = 0; j < step; ++j)
            res_data[i+j] = c1_data[i+j] & c2_data[i+j];
    for (; i < size; ++i)
        res_data[i] = c1_data[i] & c2_data[i];
    return res;
}
*/

void MergeTreeRangeReader::ReadResult::optimize(ColumnPtr current_filter, bool can_read_incomplete_granules)
{
    /// Combine new filter with the previous one if it is present

    auto combined_filter = current_filter;
    if (final_filter.present())
        combined_filter = combineFilters(final_filter.getColumn(), current_filter);//        andFilters(current_filter, final_filter.getColumn());

    FilterWithCachedCount filter(combined_filter);

    if (total_rows_per_granule == 0 || !filter.present())
        return;

    NumRows zero_tails;
    auto total_zero_rows_in_tails = countZeroTails(filter.getData(), zero_tails, can_read_incomplete_granules);

    LOG_TEST(log, "ReadResult::optimize() before: {}", dumpInfo());

//    checkInternalConsistency();

//    SCOPE_EXIT(checkInternalConsistency());

    SCOPE_EXIT({
        LOG_TEST(log, "ReadResult::optimize() after: {}", dumpInfo());
    });

    if (total_zero_rows_in_tails == filter.size())
    {
        clear();
        return;
    }
    else if (total_zero_rows_in_tails == 0 && filter.countBytesInFilter() == filter.size())
    {
        setFilterConstTrue();
        return;
    }
    /// Just a guess. If only a few rows may be skipped, it's better not to skip at all.
    else if (2 * total_zero_rows_in_tails > filter.size())
    {
        const NumRows rows_per_granule_previous = rows_per_granule;
        const size_t total_rows_per_granule_previous = total_rows_per_granule;

        for (auto i : collections::range(0, rows_per_granule.size()))
        {
            rows_per_granule[i] -= zero_tails[i];
        }
        num_rows_to_skip_in_last_granule += rows_per_granule_previous.back() - rows_per_granule.back();

/*        {
            shrink(columns, rows_per_granule_previous); /// shrink acts as filtering in such case

            auto c = block_before_prewhere.getColumns();
            shrink(c, rows_per_granule_previous); /// shrink acts as filtering in such case
            block_before_prewhere.setColumns(c);
        }
*/
        /// Check if const 1 after shrink
        if (
            num_rows == total_rows_per_granule_previous &&   /// We can apply shrink only if after the previous step the number of rows in the result
                                                    /// matches the rows_per_granule info. Otherwise we will not be able to match newly added zeros in granule tails.
            filter.countBytesInFilter() + total_zero_rows_in_tails == total_rows_per_granule)  /// All zeros are in tails?
        {
            total_rows_per_granule = total_rows_per_granule_previous - total_zero_rows_in_tails;
//            num_rows = total_rows_per_granule;
            setFilterConstTrue();


            LOG_TEST(log, "ReadResult::optimize() after shrink {}", dumpInfo());
        }
        else
        {
            auto new_filter = ColumnUInt8::create(filter.size() - total_zero_rows_in_tails);
            IColumn::Filter & new_data = new_filter->getData();

            collapseZeroTails(filter.getData(), rows_per_granule_previous, new_data);
            total_rows_per_granule = new_filter->size();
//            num_rows = total_rows_per_granule;
            final_filter = FilterWithCachedCount(new_filter->getPtr());

            LOG_TEST(log, "ReadResult::optimize() after colapseZeroTails {}", dumpInfo());
        }
        need_filter = true;
    }
    /// Another guess, if it's worth filtering at PREWHERE
    else
    {
        if (filter.countBytesInFilter() < 0.6 * filter.size())
        {
            need_filter = true;
        }

        final_filter = std::move(filter);
    }
}


#if 1
void MergeTreeRangeReader::ReadResult::setFilterConstTrue()
{
    clearFilter();
}

void MergeTreeRangeReader::ReadResult::setFilterConstFalse()
{
    clearFilter();
    columns.clear();
    num_rows = 0;
}
#endif

#if 0
void MergeTreeRangeReader::ReadResult::optimize(bool can_read_incomplete_granules)
{
    if (total_rows_per_granule == 0 || !filter.present())
        return;

    NumRows zero_tails;
    auto total_zero_rows_in_tails = countZeroTails(filter.getData(), zero_tails, can_read_incomplete_granules);

    LOG_TEST(log, "ReadResult::optimize() before: {}", dumpInfo());

//    checkInternalConsistency();

//    SCOPE_EXIT(checkInternalConsistency());

    SCOPE_EXIT({
        LOG_TEST(log, "ReadResult::optimize() after: {}", dumpInfo());
    });

    if (total_zero_rows_in_tails == filter.size())
    {
        clear();
        return;
    }
    else if (total_zero_rows_in_tails == 0 && filter.countBytesInFilter() == filter.size())
    {
        setFilterConstTrue();
        return;
    }
    /// Just a guess. If only a few rows may be skipped, it's better not to skip at all.
    else if (2 * total_zero_rows_in_tails > filter.size())
    {
        const NumRows rows_per_granule_previous = rows_per_granule;
        const size_t total_rows_per_granule_previous = total_rows_per_granule;

        for (auto i : collections::range(0, rows_per_granule.size()))
        {
            rows_per_granule[i] -= zero_tails[i];
        }
        num_rows_to_skip_in_last_granule += rows_per_granule_previous.back() - rows_per_granule.back();

        {
            shrink(columns, rows_per_granule_previous); /// shrink acts as filtering in such case

            auto c = block_before_prewhere.getColumns();
            shrink(c, rows_per_granule_previous); /// shrink acts as filtering in such case
            block_before_prewhere.setColumns(c);
        }

        /// Check if const 1 after shrink
        if (
            num_rows == total_rows_per_granule_previous &&   /// We can apply shrink only if after the previous step the number of rows in the result
                                                    /// matches the rows_per_granule info. Otherwise we will not be able to match newly added zeros in granule tails.
            filter.countBytesInFilter() + total_zero_rows_in_tails == total_rows_per_granule)  /// All zeros are in tails?
        {
            total_rows_per_granule = total_rows_per_granule_previous - total_zero_rows_in_tails;
            num_rows = total_rows_per_granule;
            setFilterConstTrue();


            LOG_TEST(log, "ReadResult::optimize() after shrink {}", dumpInfo());
        }
        else
        {
            auto new_filter = ColumnUInt8::create(filter.size() - total_zero_rows_in_tails);
            IColumn::Filter & new_data = new_filter->getData();

            collapseZeroTails(filter.getData(), rows_per_granule_previous, new_data);
            total_rows_per_granule = new_filter->size();
            num_rows = total_rows_per_granule;
            filter = FilterWithCachedCount(new_filter->getPtr());

            LOG_TEST(log, "ReadResult::optimize() after colapseZeroTails {}", dumpInfo());
        }
        need_filter = true;
    }
    /// Another guess, if it's worth filtering at PREWHERE
    else if (filter.countBytesInFilter() < 0.6 * filter.size())
        need_filter = true;
}
#endif

size_t MergeTreeRangeReader::ReadResult::countZeroTails(const IColumn::Filter & filter_vec, NumRows & zero_tails, bool can_read_incomplete_granules) const
{
    zero_tails.resize(0);
    zero_tails.reserve(rows_per_granule.size());

    const auto * filter_data = filter_vec.data();

    size_t total_zero_rows_in_tails = 0;

    for (auto rows_to_read : rows_per_granule)
    {
        /// Count the number of zeros at the end of filter for rows were read from current granule.
        size_t zero_tail = numZerosInTail(filter_data, filter_data + rows_to_read);
        if (!can_read_incomplete_granules && zero_tail != rows_to_read)
            zero_tail = 0;
        zero_tails.push_back(zero_tail);
        total_zero_rows_in_tails += zero_tails.back();
        filter_data += rows_to_read;
    }

    return total_zero_rows_in_tails;
}

void MergeTreeRangeReader::ReadResult::collapseZeroTails(const IColumn::Filter & filter_vec, const NumRows & rows_per_granule_previous, IColumn::Filter & new_filter_vec) const
{
    const auto * filter_data = filter_vec.data();
    auto * new_filter_data = new_filter_vec.data();

    for (auto i : collections::range(0, rows_per_granule.size()))
    {
        memcpySmallAllowReadWriteOverflow15(new_filter_data, filter_data, rows_per_granule[i]);
        filter_data += rows_per_granule_previous[i];
        new_filter_data += rows_per_granule[i];
    }

    new_filter_vec.resize(new_filter_data - new_filter_vec.data());
}

DECLARE_AVX512BW_SPECIFIC_CODE(
size_t numZerosInTail(const UInt8 * begin, const UInt8 * end)
{
    size_t count = 0;
    const __m512i zero64 = _mm512_setzero_epi32();
    while (end - begin >= 64)
    {
        end -= 64;
        const auto * pos = end;
        UInt64 val = static_cast<UInt64>(_mm512_cmp_epi8_mask(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pos)),
                        zero64,
                        _MM_CMPINT_EQ));
        val = ~val;
        if (val == 0)
            count += 64;
        else
        {
            count += std::countl_zero(val);
            return count;
        }
    }
    while (end > begin && *(--end) == 0)
    {
        ++count;
    }
    return count;
}
) /// DECLARE_AVX512BW_SPECIFIC_CODE

DECLARE_AVX2_SPECIFIC_CODE(
size_t numZerosInTail(const UInt8 * begin, const UInt8 * end)
{
    size_t count = 0;
    const __m256i zero32 = _mm256_setzero_si256();
    while (end - begin >= 64)
    {
        end -= 64;
        const auto * pos = end;
        UInt64 val =
            (static_cast<UInt64>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pos)),
                        zero32))) & 0xffffffffu)
            | (static_cast<UInt64>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pos + 32)),
                        zero32))) << 32u);

        val = ~val;
        if (val == 0)
            count += 64;
        else
        {
            count += std::countl_zero(val);
            return count;
        }
    }
    while (end > begin && *(--end) == 0)
    {
        ++count;
    }
    return count;
}
) /// DECLARE_AVX2_SPECIFIC_CODE

size_t MergeTreeRangeReader::ReadResult::numZerosInTail(const UInt8 * begin, const UInt8 * end)
{
#if USE_MULTITARGET_CODE
    /// check if cpu support avx512 dynamically, haveAVX512BW contains check of haveAVX512F
    if (isArchSupported(TargetArch::AVX512BW))
        return TargetSpecific::AVX512BW::numZerosInTail(begin, end);
    else if (isArchSupported(TargetArch::AVX2))
        return TargetSpecific::AVX2::numZerosInTail(begin, end);
#endif

    size_t count = 0;

#if defined(__SSE2__)
    const __m128i zero16 = _mm_setzero_si128();
    while (end - begin >= 64)
    {
        end -= 64;
        const auto * pos = end;
        UInt64 val =
                static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(pos)),
                        zero16)))
                | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(pos + 16)),
                        zero16))) << 16u)
                | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(pos + 32)),
                        zero16))) << 32u)
                | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(pos + 48)),
                        zero16))) << 48u);
        val = ~val;
        if (val == 0)
            count += 64;
        else
        {
            count += std::countl_zero(val);
            return count;
        }
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const uint8x16_t bitmask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    while (end - begin >= 64)
    {
        end -= 64;
        const auto * src = reinterpret_cast<const unsigned char *>(end);
        const uint8x16_t p0 = vceqzq_u8(vld1q_u8(src));
        const uint8x16_t p1 = vceqzq_u8(vld1q_u8(src + 16));
        const uint8x16_t p2 = vceqzq_u8(vld1q_u8(src + 32));
        const uint8x16_t p3 = vceqzq_u8(vld1q_u8(src + 48));
        uint8x16_t t0 = vandq_u8(p0, bitmask);
        uint8x16_t t1 = vandq_u8(p1, bitmask);
        uint8x16_t t2 = vandq_u8(p2, bitmask);
        uint8x16_t t3 = vandq_u8(p3, bitmask);
        uint8x16_t sum0 = vpaddq_u8(t0, t1);
        uint8x16_t sum1 = vpaddq_u8(t2, t3);
        sum0 = vpaddq_u8(sum0, sum1);
        sum0 = vpaddq_u8(sum0, sum0);
        UInt64 val = vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
        val = ~val;
        if (val == 0)
            count += 64;
        else
        {
            count += std::countl_zero(val);
            return count;
        }
    }
#endif

    while (end > begin && *(--end) == 0)
    {
        ++count;
    }
    return count;
}
#if 0
/// Filter size must match total_rows_per_granule
void MergeTreeRangeReader::ReadResult::setFilter(const ColumnPtr & new_filter)
{
    if (!new_filter && filter.present())
        throw Exception("Can't replace existing filter with empty.", ErrorCodes::LOGICAL_ERROR);

    if (filter.present())
    {
        size_t new_size = new_filter->size();

        if (new_size != total_rows_per_granule)
            throw Exception("Can't set filter because it's size is " + toString(new_size) + " but "
                            + toString(total_rows_per_granule) + " rows was read.", ErrorCodes::LOGICAL_ERROR);
    }

    ConstantFilterDescription const_description(*new_filter);
    if (const_description.always_true)
    {
        setFilterConstTrue();
    }
    else if (const_description.always_false)
    {
        clear();
    }
    else
    {
        FilterDescription filter_description(*new_filter);
        filter = FilterWithCachedCount(filter_description.data_holder ? filter_description.data_holder : new_filter);
        if (!filter.present()) /// TODO: move this check into FilterWithCachedCount(ColumnPtr) ctor
            throw Exception("setFilter function expected ColumnUInt8.", ErrorCodes::LOGICAL_ERROR);
    }
}
#endif

MergeTreeRangeReader::MergeTreeRangeReader(
    IMergeTreeReader * merge_tree_reader_,
    MergeTreeRangeReader * prev_reader_,
    const PrewhereExprStep * prewhere_info_,
    bool last_reader_in_chain_,
    const Names & non_const_virtual_column_names_)
    : merge_tree_reader(merge_tree_reader_)
    , index_granularity(&(merge_tree_reader->data_part_info_for_read->getIndexGranularity()))
    , prev_reader(prev_reader_)
    , prewhere_info(prewhere_info_)
    , last_reader_in_chain(last_reader_in_chain_)
    , is_initialized(true)
{
    if (prev_reader)
        sample_block = prev_reader->getSampleBlock();

    for (const auto & name_and_type : merge_tree_reader->getColumns())
        sample_block.insert({name_and_type.type->createColumn(), name_and_type.type, name_and_type.name});

    for (const auto & column_name : non_const_virtual_column_names_)
    {
        if (sample_block.has(column_name))
            continue;

        non_const_virtual_column_names.push_back(column_name);

//        if (column_name == "_part_offset")
//            sample_block.insert(ColumnWithTypeAndName(ColumnUInt64::create(), std::make_shared<DataTypeUInt64>(), column_name));
    }

    if (prewhere_info)
    {
        const auto & step = *prewhere_info;
        if (step.actions)
            step.actions->execute(sample_block, true);

        if (step.remove_column)
            sample_block.erase(step.column_name);
    }
}

bool MergeTreeRangeReader::isReadingFinished() const
{
    return prev_reader ? prev_reader->isReadingFinished() : stream.isFinished();
}

size_t MergeTreeRangeReader::numReadRowsInCurrentGranule() const
{
    return prev_reader ? prev_reader->numReadRowsInCurrentGranule() : stream.numReadRowsInCurrentGranule();
}

size_t MergeTreeRangeReader::numPendingRowsInCurrentGranule() const
{
    if (prev_reader)
        return prev_reader->numPendingRowsInCurrentGranule();

    auto pending_rows = stream.numPendingRowsInCurrentGranule();

    if (pending_rows)
        return pending_rows;

    return numRowsInCurrentGranule();
}


size_t MergeTreeRangeReader::numRowsInCurrentGranule() const
{
    /// If pending_rows is zero, than stream is not initialized.
    if (stream.current_mark_index_granularity)
        return stream.current_mark_index_granularity;

    /// We haven't read anything, return first
    size_t first_mark = merge_tree_reader->getFirstMarkToRead();
    return index_granularity->getMarkRows(first_mark);
}

size_t MergeTreeRangeReader::currentMark() const
{
    return stream.currentMark();
}

size_t MergeTreeRangeReader::Stream::numPendingRows() const
{
    size_t rows_between_marks = index_granularity->getRowsCountInRange(current_mark, last_mark);
    return rows_between_marks - offset_after_current_mark;
}

UInt64 MergeTreeRangeReader::Stream::currentPartOffset() const
{
    return index_granularity->getMarkStartingRow(current_mark) + offset_after_current_mark;
}

UInt64 MergeTreeRangeReader::Stream::lastPartOffset() const
{
    return index_granularity->getMarkStartingRow(last_mark);
}


size_t MergeTreeRangeReader::Stream::ceilRowsToCompleteGranules(size_t rows_num) const
{
    /// FIXME suboptimal
    size_t result = 0;
    size_t from_mark = current_mark;
    while (result < rows_num && from_mark < last_mark)
        result += index_granularity->getMarkRows(from_mark++);

    return result;
}


bool MergeTreeRangeReader::isCurrentRangeFinished() const
{
    return prev_reader ? prev_reader->isCurrentRangeFinished() : stream.isFinished();
}

#if 0
MergeTreeRangeReader::ReadResult MergeTreeRangeReader::read(size_t max_rows, MarkRanges & ranges)
{
    if (max_rows == 0)
        throw Exception("Expected at least 1 row to read, got 0.", ErrorCodes::LOGICAL_ERROR);

    ReadResult read_result(log);

    if (prev_reader)
    {
        read_result = prev_reader->read(max_rows, ranges);

        LOG_TEST(log, "Previous reader returned {}", read_result.dumpInfo());

        size_t num_read_rows;
        Columns columns = continueReadingChain(read_result, num_read_rows);

        /// Nothing to do. Return empty result.
        if (read_result.num_rows == 0)
            return read_result;

        bool has_columns = false;
        size_t total_bytes = 0;
        for (auto & column : columns)
        {
            if (column)
            {
                total_bytes += column->byteSize();
                has_columns = true;
            }
        }

        assert((!has_columns && num_read_rows == 0) ||
            num_read_rows == read_result.total_rows_per_granule);

        read_result.addNumBytesRead(total_bytes);

        bool should_evaluate_missing_defaults = false;

        if (has_columns)
        {
            /// num_read_rows >= read_result.num_rows
            /// We must filter block before adding columns to read_result.block

            /// Fill missing columns before filtering because some arrays from Nested may have empty data.
            merge_tree_reader->fillMissingColumns(columns, should_evaluate_missing_defaults, num_read_rows);

            if (read_result.filter.present())
                filterColumns(columns, read_result.filter.getData());
        }
        else
        {
            const size_t num_rows = read_result.num_rows;

            /// If block is empty, we still may need to add missing columns.
            /// In that case use number of rows in result block and don't filter block.
            if (num_rows)
                merge_tree_reader->fillMissingColumns(columns, should_evaluate_missing_defaults, num_rows);
        }

        if (!columns.empty())
        {
            /// If some columns absent in part, then evaluate default values
            if (should_evaluate_missing_defaults)
            {
                auto block = prev_reader->sample_block.cloneWithColumns(read_result.columns);
                auto block_before_prewhere = read_result.block_before_prewhere;
                for (const auto & column : block)
                {
                    if (block_before_prewhere.has(column.name))
                        block_before_prewhere.erase(column.name);
                }

                if (block_before_prewhere)
                {
//                    if (read_result.need_filter)
//                    {
//                        auto old_columns = block_before_prewhere.getColumns();
//                        if (read_result.filter.present()) // TODO: fix this properly: need_filter shouldn't be set(?)
//                            filterColumns(old_columns, read_result.filter./*getFilterOriginal()->*/getData());
//                        block_before_prewhere.setColumns(old_columns);
//                    }

                    for (auto & column : block_before_prewhere)
                        block.insert(std::move(column));
                }
                merge_tree_reader->evaluateMissingDefaults(block, columns);
            }
            /// If columns not empty, then apply on-fly alter conversions if any required
            merge_tree_reader->performRequiredConversions(columns);
        }

        read_result.columns.reserve(read_result.columns.size() + columns.size());
        for (auto & column : columns)
            read_result.columns.emplace_back(std::move(column));

        read_result.checkInternalConsistency();
    }
    else
    {
        read_result = startReadingChain(max_rows, ranges);
        read_result.num_rows = read_result.numReadRows();

        if (read_result.num_rows)
        {
            /// Physical columns go first and then some virtual columns follow
            /// TODO: is there a better way to account for virtual columns that were filled by previous readers?
            size_t physical_columns_count = read_result.columns.size() - read_result.extra_columns_filled.size();
            Columns physical_columns(read_result.columns.begin(), read_result.columns.begin() + physical_columns_count);

            bool should_evaluate_missing_defaults;
            merge_tree_reader->fillMissingColumns(physical_columns, should_evaluate_missing_defaults,
                                                  read_result.num_rows);

            /// If some columns absent in part, then evaluate default values
            if (should_evaluate_missing_defaults)
                merge_tree_reader->evaluateMissingDefaults({}, physical_columns);

            /// If result not empty, then apply on-fly alter conversions if any required
            merge_tree_reader->performRequiredConversions(physical_columns);

            for (size_t i = 0; i < physical_columns.size(); ++i)
                read_result.columns[i] = std::move(physical_columns[i]);

            read_result.checkInternalConsistency();
        }
        else
            read_result.columns.clear();

        size_t total_bytes = 0;
        for (auto & column : read_result.columns)
            total_bytes += column->byteSize();

        read_result.addNumBytesRead(total_bytes);
    }

    if (read_result.num_rows == 0)
        return read_result;

    executePrewhereActionsAndFilterColumns(read_result);

    read_result.checkInternalConsistency();

    return read_result;
}
#endif

MergeTreeRangeReader::ReadResult MergeTreeRangeReader::read(size_t max_rows, MarkRanges & ranges)
{
    if (max_rows == 0)
        throw Exception("Expected at least 1 row to read, got 0.", ErrorCodes::LOGICAL_ERROR);

    ReadResult read_result(log);

    SCOPE_EXIT({
        LOG_TEST(log, "read() returned {}, sample block {}",
            read_result.dumpInfo(), this->getSampleBlock().dumpNames());
    });

    if (prev_reader)
    {
        read_result = prev_reader->read(max_rows, ranges);

        size_t num_read_rows;
        Columns columns = continueReadingChain(read_result, num_read_rows);

        if (!columns.empty())
        {

            /// fillMissingColumns() must be called after reading but befoe any filterings because
            /// some columns (e.g. arrays) might be only partially filled and thus not be valid and
            /// fillMissingColumns() fixes this.
            bool should_evaluate_missing_defaults;
            merge_tree_reader->fillMissingColumns(columns, should_evaluate_missing_defaults,
                                                    num_read_rows);


            if (read_result.total_rows_per_granule == num_read_rows && read_result.num_rows != num_read_rows)
            {
                /// We have filter applied from the previous step
                /// So we need to apply it to the newly read rows
                assert(read_result.final_filter.present());
                assert(read_result.final_filter.countBytesInFilter() == read_result.num_rows);

                filterColumns(columns, read_result.final_filter.getData());
            }

            /// If some columns absent in part, then evaluate default values
            if (should_evaluate_missing_defaults)
                // TODO: must pass proper block here, not block_before_prewhere!
                merge_tree_reader->evaluateMissingDefaults(read_result.block_before_prewhere, columns);

            /// If result not empty, then apply on-fly alter conversions if any required
            merge_tree_reader->performRequiredConversions(columns);

        }

//        fillMissingColumns(columns, read_result);

        read_result.columns.insert(read_result.columns.end(), columns.begin(), columns.end());
    }
    else
    {
        read_result = startReadingChain(max_rows, ranges);
        read_result.num_rows = read_result.numReadRows();

        LOG_TEST(log, "First reader returned {}, requested columns {}",
            read_result.dumpInfo(), merge_tree_reader->getColumns().toString());

        fillMissingColumns(read_result.columns, read_result); // TODO: can we move it outside if-else??

        size_t total_bytes = 0;
        for (auto & column : read_result.columns)
            total_bytes += column->byteSize();

        read_result.addNumBytesRead(total_bytes);
    }

    if (read_result.num_rows == 0)
        return read_result;

    executePrewhereActionsAndFilterColumns(read_result);

    read_result.checkInternalConsistency();

    assert(read_result.num_rows == 0 || read_result.columns.size() == getSampleBlock().columns());

    if (/*prewhere_info && prewhere_info->need_filter &&*/ read_result.final_filter.present())
    {
        if (read_result.num_rows == read_result.total_rows_per_granule)
        {
            /// Filter has not been applied yet, do it now
            filterColumns(read_result.columns, read_result.final_filter.getData());
            auto columns_before_prewhere = read_result.block_before_prewhere.getColumns();
            filterColumns(columns_before_prewhere, read_result.final_filter.getData());
            read_result.block_before_prewhere.setColumns(columns_before_prewhere);
            read_result.num_rows = read_result.final_filter.countBytesInFilter();
        }
        else
        {
            /// Filter has already been applied
            assert(read_result.num_rows == read_result.final_filter.countBytesInFilter());
        }
    }

    return read_result;
}

void MergeTreeRangeReader::fillMissingColumns(Columns & physical_columns, const ReadResult & read_result)
{
    if (read_result.num_rows)
    {
        /// TODO: need to assert that physical columns have the same number of rows as block_before_prewhere

        /// Physical columns go first and then some virtual columns follow
        /// TODO: is there a better way to account for virtual columns that were filled by previous readers?
//        size_t physical_columns_count = read_result.columns.size() - read_result.extra_columns_filled.size();
//        Columns physical_columns(read_result.columns.begin(), read_result.columns.begin() + physical_columns_count);

        bool should_evaluate_missing_defaults;
        merge_tree_reader->fillMissingColumns(physical_columns, should_evaluate_missing_defaults,
                                                read_result.num_rows);

        /// If some columns absent in part, then evaluate default values
        if (should_evaluate_missing_defaults)
            // TODO: must pass proper block here, not block_before_prewhere!
            merge_tree_reader->evaluateMissingDefaults(read_result.block_before_prewhere, physical_columns);

        /// If result not empty, then apply on-fly alter conversions if any required
        merge_tree_reader->performRequiredConversions(physical_columns);

/*
        for (const auto & column_name : non_const_virtual_column_names)
        {
            if (column_name == "_part_offset")
            {
                // TODO: properly fill _part_offset!
                physical_columns.emplace_back(ColumnUInt64::create(read_result.num_rows));
            }
        }
//*/
//        for (size_t i = 0; i < physical_columns.size(); ++i)
//            read_result.columns[i] = std::move(physical_columns[i]);

//        read_result.checkInternalConsistency();
    }
}


void MergeTreeRangeReader::executePrewhereActionsAndFilterColumns(ReadResult & result) const
{
    result.checkInternalConsistency();

    if (!prewhere_info)
        return;

    const auto & header = merge_tree_reader->getColumns();
    size_t num_columns = header.size();

    /// Check that we have columns from previous steps and newly read required columns
    if (result.columns.size() < num_columns + result.extra_columns_filled.size())
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Invalid number of columns passed to MergeTreeRangeReader. Expected {}, got {}",
                        num_columns, result.columns.size());

    /// This filter has the size of total_rows_per granule. It is applied after reading contiguous chunks from
    /// the start of each granule.
//    ColumnPtr combined_filter;
    /// Filter computed at the current step. Its size is equal to num_rows which is <= total_rows_per_granule
    ColumnPtr current_step_filter;
    size_t prewhere_column_pos;

    {
        /// Restore block from columns list.
        Block block;
        size_t pos = 0;

        if (prev_reader)
        {
            for (const auto & col : prev_reader->getSampleBlock())
            {
                block.insert({result.columns[pos], col.type, col.name});
                ++pos;
            }
        }

        for (auto name_and_type = header.begin(); name_and_type != header.end() && pos < result.columns.size(); ++pos, ++name_and_type)
            block.insert({result.columns[pos], name_and_type->type, name_and_type->name});


    /*// HACK!! fix it
        if (getSampleBlock().has("_part_offset"))
        {
            const auto & col = getSampleBlock().getByName("_part_offset");
            block.insert({result.columns.back(), col.type, col.name});
        }
/////////////*/

        /// Columns might be projected out. We need to store them here so that default columns can be evaluated later.
        result.block_before_prewhere = block;

        if (prewhere_info->actions)
           prewhere_info->actions->execute(block);

        prewhere_column_pos = block.getPositionByName(prewhere_info->column_name);

        result.columns.clear();
        result.columns.reserve(block.columns());
        for (auto & col : block)
            result.columns.emplace_back(std::move(col.column));

        current_step_filter = result.columns[prewhere_column_pos];
//        combined_filter = current_step_filter;
    }

    if (prewhere_info->remove_column)
        result.columns.erase(result.columns.begin() + prewhere_column_pos);

//*////////////////////
// HACK: Always apply current filter
    //if (result.need_filter || prewhere_info->need_filter)
    {
        /// Filter has not been applied yet, do it now
        filterColumns(result.columns, current_step_filter);

        if (!last_reader_in_chain)
        {
            auto columns_before_prewhere = result.block_before_prewhere.getColumns();
            filterColumns(columns_before_prewhere, current_step_filter);
            if (!columns_before_prewhere.empty())
                result.block_before_prewhere.setColumns(columns_before_prewhere);
            else
                result.block_before_prewhere.clear();
        }
        else
        {
            result.block_before_prewhere.clear();
        }

        {
            /// TODO: clenup this logic, filterColumns internally has similar one
            ConstantFilterDescription const_descr(*current_step_filter);
            if (const_descr.always_true)
                ;
            else if (const_descr.always_false)
                result.num_rows = 0;
            else
                result.num_rows = FilterDescription(*current_step_filter).countBytesInFilter();
        }
    }

//*///////////////////

    result.optimize(current_step_filter, merge_tree_reader->canReadIncompleteGranules());

    LOG_TEST(log, "After execute prewhere {}", result.dumpInfo());
}

/*
void MergeTreeRangeReader::optimize(ReadResult & result, ColumnPtr current_step_filter) const
{
    NumRows zero_tails;
    auto total_zero_rows_in_tails = countZeroTails(filter.getData(), zero_tails, can_read_incomplete_granules);

    FilterWithCachedCount filter(current_step_filter);

    filterColumns(result.columns, filter.getColumn());
    auto before_columns = result.block_before_prewhere.getColumns();
    filterColumns(before_columns, filter.getColumn());
    result.block_before_prewhere.setColumns(before_columns);

    result.num_rows = filter.countBytesInFilter();
}
*/


MergeTreeRangeReader::ReadResult MergeTreeRangeReader::startReadingChain(size_t max_rows, MarkRanges & ranges)
{
    ReadResult result(log);
    result.columns.resize(merge_tree_reader->getColumns().size());

    size_t current_task_last_mark = getLastMark(ranges);

    /// The stream could be unfinished by the previous read request because of max_rows limit.
    /// In this case it will have some rows from the previously started range. We need to save their begin and
    /// end offsets to properly fill _part_offset column.
//    UInt64 leading_begin_part_offset = 0;
//    UInt64 leading_end_part_offset = 0;
//    if (!stream.isFinished())
//    {
//        leading_begin_part_offset = stream.currentPartOffset();
//        leading_end_part_offset = stream.lastPartOffset();
//    }

    /// Stream is lazy. result.num_added_rows is the number of rows added to block which is not equal to
    /// result.num_rows_read until call to stream.finalize(). Also result.num_added_rows may be less than
    /// result.num_rows_read if the last granule in range also the last in part (so we have to adjust last granule).
    {
        size_t space_left = max_rows;
        while (space_left && (!stream.isFinished() || !ranges.empty()))
        {
            if (stream.isFinished())
            {
                result.addRows(stream.finalize(result.columns));
                stream = Stream(ranges.front().begin, ranges.front().end, current_task_last_mark, merge_tree_reader);
                result.addRange(ranges.front());
                ranges.pop_front();
            }

            size_t current_space = space_left;

            /// If reader can't read part of granule, we have to increase number of reading rows
            ///  to read complete granules and exceed max_rows a bit.
            if (!merge_tree_reader->canReadIncompleteGranules())
                current_space = stream.ceilRowsToCompleteGranules(space_left);

            auto rows_to_read = std::min(current_space, stream.numPendingRowsInCurrentGranule());

            bool last = rows_to_read == space_left;
            result.addRows(stream.read(result.columns, rows_to_read, !last));
            result.addGranule(rows_to_read);
            space_left = (rows_to_read > space_left ? 0 : space_left - rows_to_read);
        }
    }

    result.addRows(stream.finalize(result.columns));

    /// Last granule may be incomplete.
    if (!result.rows_per_granule.empty())
        result.adjustLastGranule();

//    for (const auto & column_name : non_const_virtual_column_names)
//    {
//        if (column_name == "_part_offset")
//            fillPartOffsetColumn(result, leading_begin_part_offset, leading_end_part_offset);
//    }

    return result;
}
#if 0
void MergeTreeRangeReader::fillPartOffsetColumn(ReadResult & result, UInt64 leading_begin_part_offset, UInt64 leading_end_part_offset)
{
    size_t num_rows = result.numReadRows();

    auto column = ColumnUInt64::create(num_rows);
    ColumnUInt64::Container & vec = column->getData();

    UInt64 * pos = vec.data();
    UInt64 * end = &vec[num_rows];

    while (pos < end && leading_begin_part_offset < leading_end_part_offset)
        *pos++ = leading_begin_part_offset++;

    const auto & start_ranges = result.started_ranges;

    for (const auto & start_range : start_ranges)
    {
        UInt64 start_part_offset = index_granularity->getMarkStartingRow(start_range.range.begin);
        UInt64 end_part_offset = index_granularity->getMarkStartingRow(start_range.range.end);

        while (pos < end && start_part_offset < end_part_offset)
            *pos++ = start_part_offset++;
    }

    result.columns.emplace_back(std::move(column));
    result.extra_columns_filled.push_back("_part_offset");
}
#endif

Columns MergeTreeRangeReader::continueReadingChain(const ReadResult & result, size_t & num_rows)
{
    Columns columns;
    num_rows = 0;

    /// No columns need to be read at this step? (only more filtering)
    if (merge_tree_reader->getColumns().empty())
        return columns;

    if (result.rows_per_granule.empty())
    {
        /// If zero rows were read on prev step, than there is no more rows to read.
        /// Last granule may have less rows than index_granularity, so finish reading manually.
        stream.finish();
        return columns;
    }

    columns.resize(merge_tree_reader->numColumnsInResult());

    const auto & rows_per_granule = result.rows_per_granule;
    const auto & started_ranges = result.started_ranges;

    size_t current_task_last_mark = ReadResult::getLastMark(started_ranges);
    size_t next_range_to_start = 0;

    auto size = rows_per_granule.size();
    for (auto i : collections::range(0, size))
    {
        if (next_range_to_start < started_ranges.size()
            && i == started_ranges[next_range_to_start].num_granules_read_before_start)
        {
            num_rows += stream.finalize(columns);
            const auto & range = started_ranges[next_range_to_start].range;
            ++next_range_to_start;
            stream = Stream(range.begin, range.end, current_task_last_mark, merge_tree_reader);
        }

        bool last = i + 1 == size;
        num_rows += stream.read(columns, rows_per_granule[i], !last);
    }

    stream.skip(result.num_rows_to_skip_in_last_granule);
    num_rows += stream.finalize(columns);

    /// added_rows may be zero if all columns were read in prewhere and it's ok.
    if (num_rows && num_rows != result.total_rows_per_granule)
        throw Exception("RangeReader read " + toString(num_rows) + " rows, but "
                        + toString(result.total_rows_per_granule) + " expected.", ErrorCodes::LOGICAL_ERROR);

    return columns;
}

#if 0
void MergeTreeRangeReader::executePrewhereActionsAndFilterColumns(ReadResult & result)
{
    result.checkInternalConsistency();

    if (!prewhere_info)
        return;

    const auto & header = merge_tree_reader->getColumns();
    size_t num_columns = header.size();

    /// Check that we have columns from previous steps and newly read required columns
    if (result.columns.size() < num_columns + result.extra_columns_filled.size())
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Invalid number of columns passed to MergeTreeRangeReader. Expected {}, got {}",
                        num_columns, result.columns.size());

    /// This filter has the size of total_rows_per granule. It is applied after reading contiguous chunks from
    /// the start of each granule.
    ColumnPtr combined_filter;
    /// Filter computed at the current step. Its size is equal to num_rows which is <= total_rows_per_granule
    ColumnPtr current_step_filter;
    size_t prewhere_column_pos;

    {
        /// Restore block from columns list.
        Block block;
        size_t pos = 0;

        if (prev_reader)
        {
            for (const auto & col : prev_reader->getSampleBlock())
            {
                block.insert({result.columns[pos], col.type, col.name});
                ++pos;
            }
        }

        for (auto name_and_type = header.begin(); name_and_type != header.end() && pos < result.columns.size(); ++pos, ++name_and_type)
            block.insert({result.columns[pos], name_and_type->type, name_and_type->name});

        for (const auto & column_name : non_const_virtual_column_names)
        {
            if (block.has(column_name))
                continue;

            if (column_name == "_part_offset")
            {
                if (pos >= result.columns.size())
                    throw Exception(ErrorCodes::LOGICAL_ERROR,
                                    "Invalid number of columns passed to MergeTreeRangeReader. Expected {}, got {}",
                                    num_columns, result.columns.size());

                block.insert({result.columns[pos], std::make_shared<DataTypeUInt64>(), column_name});
            }
            else if (column_name == LightweightDeleteDescription::FILTER_COLUMN.name)
            {
                /// Do nothing, it will be added later
            }
            else
                throw Exception("Unexpected non-const virtual column: " + column_name, ErrorCodes::LOGICAL_ERROR);
            ++pos;
        }

        /// Columns might be projected out. We need to store them here so that default columns can be evaluated later.
        result.block_before_prewhere = block;

        if (prewhere_info->actions)
           prewhere_info->actions->execute(block);

        prewhere_column_pos = block.getPositionByName(prewhere_info->column_name);

        result.columns.clear();
        result.columns.reserve(block.columns());
        for (auto & col : block)
            result.columns.emplace_back(std::move(col.column));

        current_step_filter.swap(result.columns[prewhere_column_pos]);
        combined_filter = current_step_filter;
    }

    if (result.filter.present())
    {
        ColumnPtr prev_filter = result.filter.getColumn();
        combined_filter = combineFilters(prev_filter, std::move(combined_filter));
    }

    result.setFilter(combined_filter);

    /// If there is a WHERE, we filter in there, and only optimize IO and shrink columns here
    if (!last_reader_in_chain)
        result.optimize(merge_tree_reader->canReadIncompleteGranules());

    /// If we read nothing or filter gets optimized to nothing
    if (result.total_rows_per_granule == 0)
        result.setFilterConstFalse();
    /// If we need to filter in PREWHERE
    else if (prewhere_info->need_filter || result.need_filter)
    {
        /// If there is a filter and without optimized
        if (result.filter.present() && last_reader_in_chain)
        {
            /// optimize is not called, need to check const 1 and const 0
            size_t bytes_in_filter = result.filter.countBytesInFilter();
            if (bytes_in_filter == 0)
                result.setFilterConstFalse();
            else if (bytes_in_filter == result.num_rows)
                result.setFilterConstTrue();
        }

        /// If there is still a filter, do the filtering now
        if (result.filter.present())
        {
            LOG_TEST(log, "filterColumns before: {}", result.dumpInfo());

            filterColumns(result.columns, result.filter.getColumn());//current_step_filter);
            //////
            if (result.block_before_prewhere)
            {
                auto c = result.block_before_prewhere.getColumns();
                filterColumns(c, result.filter.getColumn());//current_step_filter);
                result.block_before_prewhere.setColumns(c);
            }
            //////

            result.need_filter = true;

            bool has_column = false;
            for (auto & column : result.columns)
            {
                if (column)
                {
                    has_column = true;
                    result.num_rows = column->size();
                    break;
                }
            }

            /// There is only one filter column. Record the actual number
            if (!has_column)
                result.num_rows = result.filter.countBytesInFilter();
        }

        /// Check if the PREWHERE column is needed
        if (!result.columns.empty())
        {
            if (prewhere_info->remove_column)
                result.columns.erase(result.columns.begin() + prewhere_column_pos);
            else
                result.columns[prewhere_column_pos] =
                        getSampleBlock().getByName(prewhere_info->column_name).type->
                                createColumnConst(result.num_rows, 1u)->convertToFullColumnIfConst();
        }
    }
    /// Filter in WHERE instead
    else
    {
        if (prewhere_info->remove_column)
            result.columns.erase(result.columns.begin() + prewhere_column_pos);
        else
        {
            auto type = getSampleBlock().getByName(prewhere_info->column_name).type;
            auto filter_column = result.filter.getColumn();
            if (!filter_column)
            {
                result.columns[prewhere_column_pos] = type->createColumnConst(result.num_rows, 1u);
            }
            else
            {
                ColumnWithTypeAndName col(filter_column->convertToFullIfNeeded(), std::make_shared<DataTypeUInt8>(), "");
                result.columns[prewhere_column_pos] = castColumn(col, type);
            }
            result.clearFilter(); // TODO: is this still relevant? : Acting as a flag to not filter in PREWHERE
        }
    }

    LOG_TEST(log, "After execute prewhere {}", result.dumpInfo());
}
#endif

std::string PrewhereExprInfo::dump() const
{
    WriteBufferFromOwnString s;

    for (size_t i = 0; i < steps.size(); ++i)
    {
        s << "STEP " << i << ":\n"
            << "  ACTIONS: " << (steps[i].actions ? steps[i].actions->dumpActions() : "nullptr") << "\n"
            << "  COLUMN: " << steps[i].column_name << "\n"
            << "  REMOVE_COLUMN: " << steps[i].remove_column << "\n"
            << "  NEED_FILTER: " << steps[i].need_filter << "\n";
    }

    return s.str();
}

}
