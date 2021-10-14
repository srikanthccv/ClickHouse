#pragma once
/**
  * This file implements template methods of IColumn that depend on other types
  * we don't want to include.
  * Currently, this is only the scatterImpl method that depends on PODArray
  * implementation.
  */

#include <Columns/IColumn.h>
#include <Common/PODArray.h>
#include <base/sort.h>
#include <algorithm>

namespace DB
{
namespace ErrorCodes
{
    extern const int SIZES_OF_COLUMNS_DOESNT_MATCH;
}

template <typename Derived>
std::vector<IColumn::MutablePtr> IColumn::scatterImpl(ColumnIndex num_columns,
                                             const Selector & selector) const
{
    size_t num_rows = size();

    if (num_rows != selector.size())
        throw Exception(
                "Size of selector: " + std::to_string(selector.size()) + " doesn't match size of column: " + std::to_string(num_rows),
                ErrorCodes::SIZES_OF_COLUMNS_DOESNT_MATCH);

    std::vector<MutablePtr> columns(num_columns);
    for (auto & column : columns)
        column = cloneEmpty();

    {
        size_t reserve_size = num_rows * 1.1 / num_columns;    /// 1.1 is just a guess. Better to use n-sigma rule.

        if (reserve_size > 1)
            for (auto & column : columns)
                column->reserve(reserve_size);
    }

    for (size_t i = 0; i < num_rows; ++i)
        static_cast<Derived &>(*columns[selector[i]]).insertFrom(*this, i);

    return columns;
}

template <typename Derived, bool reversed, bool use_indexes>
void IColumn::compareImpl(const Derived & rhs, size_t rhs_row_num,
                          PaddedPODArray<UInt64> * row_indexes [[maybe_unused]],
                          PaddedPODArray<Int8> & compare_results,
                          int nan_direction_hint) const
{
    size_t num_rows = size();
    size_t num_indexes = num_rows;
    UInt64 * indexes [[maybe_unused]];
    UInt64 * next_index [[maybe_unused]];

    if constexpr (use_indexes)
    {
        num_indexes = row_indexes->size();
        next_index = indexes = row_indexes->data();
    }

    compare_results.resize(num_rows);

    if (compare_results.empty())
        compare_results.resize(num_rows);
    else if (compare_results.size() != num_rows)
        throw Exception(
                "Size of compare_results: " + std::to_string(compare_results.size()) + " doesn't match rows_num: " + std::to_string(num_rows),
                ErrorCodes::SIZES_OF_COLUMNS_DOESNT_MATCH);

    for (size_t i = 0; i < num_indexes; ++i)
    {
        UInt64 row = i;

        if constexpr (use_indexes)
            row = indexes[i];

        int res = compareAt(row, rhs_row_num, rhs, nan_direction_hint);

        /// We need to convert int to Int8. Sometimes comparison return values which do not fit in one byte.
        if (res < 0)
            compare_results[row] = -1;
        else if (res > 0)
            compare_results[row] = 1;
        else
            compare_results[row] = 0;

        if constexpr (reversed)
            compare_results[row] = -compare_results[row];

        if constexpr (use_indexes)
        {
            if (compare_results[row] == 0)
            {
                *next_index = row;
                ++next_index;
            }
        }
    }

    if constexpr (use_indexes)
        row_indexes->resize(next_index - row_indexes->data());
}

template <typename Derived>
void IColumn::doCompareColumn(const Derived & rhs, size_t rhs_row_num,
                              PaddedPODArray<UInt64> * row_indexes,
                              PaddedPODArray<Int8> & compare_results,
                              int direction, int nan_direction_hint) const
{
    if (direction < 0)
    {
        if (row_indexes)
            compareImpl<Derived, true, true>(rhs, rhs_row_num, row_indexes, compare_results, nan_direction_hint);
        else
            compareImpl<Derived, true, false>(rhs, rhs_row_num, row_indexes, compare_results, nan_direction_hint);
    }
    else
    {
        if (row_indexes)
            compareImpl<Derived, false, true>(rhs, rhs_row_num, row_indexes, compare_results, nan_direction_hint);
        else
            compareImpl<Derived, false, false>(rhs, rhs_row_num, row_indexes, compare_results, nan_direction_hint);
    }
}

template <typename Derived>
bool IColumn::hasEqualValuesImpl() const
{
    size_t num_rows = size();
    for (size_t i = 1; i < num_rows; ++i)
    {
        if (compareAt(i, 0, static_cast<const Derived &>(*this), false) != 0)
            return false;
    }
    return true;
}

template <typename Derived>
double IColumn::getRatioOfDefaultRowsImpl(double sample_ratio) const
{
    if (sample_ratio <= 0.0 || sample_ratio > 1.0)
        throw Exception(ErrorCodes::LOGICAL_ERROR,
            "Value of 'sample_ratio' must be in interval (0.0; 1.0], but got: {}", sample_ratio);

    size_t num_rows = size();
    size_t num_sampled_rows = static_cast<size_t>(num_rows * sample_ratio);
    if (num_sampled_rows == 0)
        return 0.0;

    size_t step = num_rows / num_sampled_rows;
    std::uniform_int_distribution<size_t> dist(1, step);

    size_t res = 0;
    for (size_t i = 0; i < num_rows; i += step)
    {
        size_t idx = std::min(i + dist(thread_local_rng), num_rows - 1);
        res += static_cast<const Derived &>(*this).isDefaultAt(idx);
    }

    return static_cast<double>(res) / num_sampled_rows;
}

template <typename Derived>
void IColumn::getIndicesOfNonDefaultRowsImpl(Offsets & indices, size_t from, size_t limit) const
{
    size_t to = limit && from + limit < size() ? from + limit : size();
    indices.reserve(indices.size() + to - from);

    for (size_t i = from; i < to; ++i)
    {
        if (!static_cast<const Derived &>(*this).isDefaultAt(i))
            indices.push_back(i);
    }
}

template <typename Comparator>
void IColumn::updatePermutationImpl(
    size_t limit,
    Permutation & res,
    EqualRanges & equal_ranges,
    Comparator cmp) const
{
    updatePermutationImpl(
        limit, res, equal_ranges,
        [&cmp](size_t lhs, size_t rhs) { return cmp(lhs, rhs) < 0; },
        [&cmp](size_t lhs, size_t rhs) { return cmp(lhs, rhs) == 0; },
        [](auto begin, auto end, auto pred) { std::sort(begin, end, pred); },
        [](auto begin, auto mid, auto end, auto pred) { ::partial_sort(begin, mid, end, pred); });
}

template <typename Less, typename Equals, typename Sort, typename PartialSort>
void IColumn::updatePermutationImpl(
    size_t limit,
    Permutation & res,
    EqualRanges & equal_ranges,
    Less less,
    Equals equals,
    Sort full_sort,
    PartialSort partial_sort) const
{
    if (equal_ranges.empty())
        return;

    if (limit >= size() || limit > equal_ranges.back().second)
        limit = 0;

    EqualRanges new_ranges;

    size_t number_of_ranges = equal_ranges.size();
    if (limit)
        --number_of_ranges;

    for (size_t i = 0; i < number_of_ranges; ++i)
    {
        const auto & [first, last] = equal_ranges[i];
        full_sort(res.begin() + first, res.begin() + last, less);

        size_t new_first = first;
        for (size_t j = first + 1; j < last; ++j)
        {
            if (!equals(res[j], res[new_first]))
            {
                if (j - new_first > 1)
                    new_ranges.emplace_back(new_first, j);

                new_first = j;
            }
        }

        if (last - new_first > 1)
            new_ranges.emplace_back(new_first, last);
    }

    if (limit)
    {
        const auto & [first, last] = equal_ranges.back();

        if (limit < first || limit > last)
        {
            equal_ranges = std::move(new_ranges);
            return;
        }

        /// Since then we are working inside the interval.
        partial_sort(res.begin() + first, res.begin() + limit, res.begin() + last, less);

        size_t new_first = first;
        for (size_t j = first + 1; j < limit; ++j)
        {
            if (!equals(res[j], res[new_first]))
            {
                if (j - new_first > 1)
                    new_ranges.emplace_back(new_first, j);
                new_first = j;
            }
        }

        size_t new_last = limit;
        for (size_t j = limit; j < last; ++j)
        {
            if (equals(res[j], res[new_first]))
            {
                std::swap(res[j], res[new_last]);
                ++new_last;
            }
        }

        if (new_last - new_first > 1)
            new_ranges.emplace_back(new_first, new_last);
    }

    equal_ranges = std::move(new_ranges);
}

}
