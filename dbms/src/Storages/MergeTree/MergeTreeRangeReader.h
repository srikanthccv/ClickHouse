#pragma once
#include <Core/Block.h>
#include <common/logger_useful.h>

namespace DB
{

class MergeTreeReader;

/// MergeTreeReader iterator which allows sequential reading for arbitrary number of rows between pairs of marks in the same part.
/// Stores reading state, which can be inside granule. Can skip rows in current granule and start reading from next mark.
/// Used generally for reading number of rows less than index granularity to decrease cache misses for fat blocks.
class MergeTreePrewhereRangeReader
{
public:
    MergeTreePrewhereRangeReader(MergeTreePrewhereRangeReader * prev_reader, MergeTreeReader * merge_tree_reader,
                                 size_t from_mark, size_t to_mark, size_t index_granularity,
                                 ExpressionActionsPtr prewhere_actions, const String * prewhere_column_name,
                                 const Names * ordered_names, bool always_reorder);

    MergeTreePrewhereRangeReader() : is_initialized(false) { }

    bool isReadingFinished() const { return prev_reader ? prev_reader->isReadingFinished() : stream.isFinished(); }

    size_t numReadRowsInCurrentGranule() const { return prev_reader ? prev_reader->numReadRowsInCurrentGranule() : stream.numReadRowsInCurrentGranule(); }
    size_t numPendingRowsInCurrentGranule() const { return prev_reader ? prev_reader->numPendingRowsInCurrentGranule() : stream.numPendingRowsInCurrentGranule(); }
    size_t numPendingRows() const { return prev_reader ? prev_reader->numPendingRows() : stream.numPendingRows(); }

    operator bool() const { return is_initialized; }
    void reset() { is_initialized = false; }

    class DelayedStream
    {
    public:
        DelayedStream() {}
        DelayedStream(size_t from_mark, size_t index_granularity, MergeTreeReader * merge_tree_reader);

        /// Returns the number of rows added to block.
        /// NOTE: have to return number of rows because block has broken invariant:
        ///       some columns may have different size (for example, default columns may be zero size).
        size_t read(Block & block, size_t from_mark, size_t offset, size_t num_rows);
        size_t finalize(Block & block);

        bool isFinished() const { return is_finished; }

        MergeTreeReader * reader() const { return merge_tree_reader; }

    private:
        size_t current_mark;
        size_t current_offset;
        size_t num_delayed_rows;

        size_t index_granularity;
        MergeTreeReader * merge_tree_reader;
        bool continue_reading;
        bool is_finished;

        size_t position() const;
        size_t readRows(Block & block, size_t num_rows);
    };

    class Stream
    {

    public:
        Stream() {}
        Stream(size_t from_mark, size_t to_mark, size_t index_granularity, MergeTreeReader * merge_tree_reader);

        /// Returns the n
        size_t read(Block & block, size_t num_rows, bool skip_remaining_rows_in_current_granule);
        size_t finalize(Block & block);
        void skip(size_t num_rows);

        void finish() { current_mark = last_mark; }
        bool isFinished() const { return current_mark >= last_mark; }

        size_t numReadRowsInCurrentGranule() const { return offset_after_current_mark; }
        size_t numPendingRowsInCurrentGranule() const { return index_granularity - numReadRowsInCurrentGranule(); }
        size_t numRendingGranules() const { return last_mark - current_mark; }
        size_t numPendingRows() const { return numRendingGranules() * index_granularity - offset_after_current_mark; }

        MergeTreeReader * reader() const { return stream.reader(); }

    private:
        size_t current_mark;
        /// Invariant: offset_after_current_mark + skipped_rows_after_offset < index_granularity
        size_t offset_after_current_mark;

        size_t index_granularity;
        size_t last_mark;

        DelayedStream stream;

        void checkNotFinished() const;
        void checkEnoughSpaceInCurrentGranula(size_t num_rows) const;
        size_t readRows(Block & block, size_t num_rows);
    };

    /// Statistics after next reading step.
    class ReadResult
    {
    public:
        const std::vector<size_t> & rowsPerGranule() const { return rows_per_granule; }
        /// The number of rows were read at LAST iteration in chain. <= num_added_rows + num_filtered_rows.
        size_t numReadRows() const { return num_read_rows; }
        /// The number of rows were added to block as a result of reading chain.
        size_t getNumAddedRows() const { return num_added_rows; }
        /// The number of filtered rows at all steps in reading chain.
        size_t getNumFilteredRows() const { return num_filtered_rows; }
        /// Filter you need to allply to newly-read columns in order to add them to block.
        const ColumnPtr & getFilter() const { return filter; }

        void addGranule(size_t num_rows);
        void adjustLastGranule(size_t num_rows_to_subtract);
        void addRows(size_t rows) { num_added_rows += rows; }

        /// Set filter or replace old one. Filter must have more zeroes than previous.
        void setFilter(ColumnPtr filter_);
        /// For each granule calculate the number of filtered rows at the end. Remove them and update filter.
        void optimize();
        /// Remove all rows from granules.
        void clear();

    private:
        /// The number of rows read from each granule.
        std::vector<size_t> rows_per_granule;
        /// Sum(rows_per_granule)
        size_t num_read_rows = 0;
        /// The number of rows was added to block while reading columns. May be zero if no read columns present in part.
        size_t num_added_rows = 0;
        /// num_zeros_in_filter + the number of rows removed after optimizes.
        size_t num_filtered_rows = 0;
        /// Zero if filter is nullptr.
        size_t num_zeros_in_filter = 0;
        /// nullptr if prev reader hasn't prewhere_actions. Otherwise filter.size() >= total_rows_read.
        ColumnPtr filter;

        void collapseZeroTails(const IColumn::Filter & filter, IColumn::Filter & new_filter);
        size_t numZerosInFilter() const;
        static size_t numZerosInTail(const UInt8 * begin, const UInt8 * end);
    };

    ReadResult read(Block & res, size_t max_rows);

private:

    void readRows(Block & block, size_t max_rows, ReadResult & result);
    void executePrewhereActionsAndFilterColumns(Block & block, ReadResult & result);

    Stream stream;
    MergeTreePrewhereRangeReader * prev_reader; /// If not nullptr, read from prev_reader firstly.

    ExpressionActionsPtr prewhere_actions; /// If not nullptr, calculate filter.
    const String * prewhere_column_name;
    const Names * ordered_names;
    bool always_reorder;

    bool is_initialized = true;
};

}

