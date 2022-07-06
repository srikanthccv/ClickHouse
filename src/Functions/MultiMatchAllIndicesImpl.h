#pragma once

#include <base/types.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnString.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeArray.h>
#include <Functions/checkHyperscanRegexp.h>
#include "Regexps.h"

#include "config_functions.h"
#include <Common/config.h>

#if USE_VECTORSCAN
#    include <hs.h>
#endif


namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_ALLOCATE_MEMORY;
    extern const int FUNCTION_NOT_ALLOWED;
    extern const int HYPERSCAN_CANNOT_SCAN_TEXT;
    extern const int NOT_IMPLEMENTED;
    extern const int TOO_MANY_BYTES;
}


template <typename Name, typename ResultType_, bool WithEditDistance>
struct MultiMatchAllIndicesImpl
{
    using ResultType = ResultType_;

    /// Variable for understanding, if we used offsets for the output, most
    /// likely to determine whether the function returns ColumnVector of ColumnArray.
    static constexpr bool is_column_array = true;
    static constexpr auto name = Name::name;

    static auto getReturnType()
    {
        return std::make_shared<DataTypeArray>(std::make_shared<DataTypeUInt64>());
    }

    static void vectorConstant(
        const ColumnString::Chars & haystack_data,
        const ColumnString::Offsets & haystack_offsets,
        const Array & needles_arr,
        PaddedPODArray<ResultType> & res,
        PaddedPODArray<UInt64> & offsets,
        bool allow_hyperscan,
        size_t max_hyperscan_regexp_length,
        size_t max_hyperscan_regexp_total_length)
    {
        vectorConstant(haystack_data, haystack_offsets, needles_arr, res, offsets, std::nullopt, allow_hyperscan, max_hyperscan_regexp_length, max_hyperscan_regexp_total_length);
    }

    static void vectorConstant(
        const ColumnString::Chars & haystack_data,
        const ColumnString::Offsets & haystack_offsets,
        const Array & needles_arr,
        PaddedPODArray<ResultType> & res,
        PaddedPODArray<UInt64> & offsets,
        std::optional<UInt32> edit_distance,
        bool allow_hyperscan,
        size_t max_hyperscan_regexp_length,
        size_t max_hyperscan_regexp_total_length)
    {
        if (!allow_hyperscan)
            throw Exception(ErrorCodes::FUNCTION_NOT_ALLOWED, "Hyperscan functions are disabled, because setting 'allow_hyperscan' is set to 0");
#if USE_VECTORSCAN
        std::vector<std::string_view> needles;
        needles.reserve(needles_arr.size());
        for (const auto & needle : needles_arr)
            needles.emplace_back(needle.get<String>());

        checkHyperscanRegexp(needles, max_hyperscan_regexp_length, max_hyperscan_regexp_total_length);

        offsets.resize(haystack_offsets.size());

        const auto & hyperscan_regex = MultiRegexps::get</*SaveIndices=*/true, WithEditDistance>(needles, edit_distance);
        hs_scratch_t * scratch = nullptr;
        hs_error_t err = hs_clone_scratch(hyperscan_regex->getScratch(), &scratch);

        if (err != HS_SUCCESS)
            throw Exception("Could not clone scratch space for hyperscan", ErrorCodes::CANNOT_ALLOCATE_MEMORY);

        MultiRegexps::ScratchPtr smart_scratch(scratch);

        auto on_match = [](unsigned int id,
                           unsigned long long /* from */, // NOLINT
                           unsigned long long /* to */, // NOLINT
                           unsigned int /* flags */,
                           void * context) -> int
        {
            static_cast<PaddedPODArray<ResultType>*>(context)->push_back(id);
            return 0;
        };
        const size_t haystack_offsets_size = haystack_offsets.size();
        UInt64 offset = 0;
        for (size_t i = 0; i < haystack_offsets_size; ++i)
        {
            UInt64 length = haystack_offsets[i] - offset - 1;
            /// vectorscan restriction.
            if (length > std::numeric_limits<UInt32>::max())
                throw Exception("Too long string to search", ErrorCodes::TOO_MANY_BYTES);
            /// scan, check, update the offsets array and the offset of haystack.
            err = hs_scan(
                hyperscan_regex->getDB(),
                reinterpret_cast<const char *>(haystack_data.data()) + offset,
                length,
                0,
                smart_scratch.get(),
                on_match,
                &res);
            if (err != HS_SUCCESS)
                throw Exception("Failed to scan with vectorscan", ErrorCodes::HYPERSCAN_CANNOT_SCAN_TEXT);
            offsets[i] = res.size();
            offset = haystack_offsets[i];
        }
#else
        (void)haystack_data;
        (void)haystack_offsets;
        (void)needles_arr;
        (void)res;
        (void)offsets;
        (void)edit_distance;
        (void)max_hyperscan_regexp_length;
        (void)max_hyperscan_regexp_total_length;
        throw Exception(
            "multi-search all indices is not implemented when vectorscan is off",
            ErrorCodes::NOT_IMPLEMENTED);
#endif // USE_VECTORSCAN
    }

    static void vectorVector(
        const ColumnString::Chars & haystack_data,
        const ColumnString::Offsets & haystack_offsets,
        const IColumn & needles_data,
        const ColumnArray::Offsets & needles_offsets,
        PaddedPODArray<ResultType> & res,
        PaddedPODArray<UInt64> & offsets,
        bool allow_hyperscan,
        size_t max_hyperscan_regexp_length,
        size_t max_hyperscan_regexp_total_length)
    {
        vectorVector(haystack_data, haystack_offsets, needles_data, needles_offsets, res, offsets, std::nullopt, allow_hyperscan, max_hyperscan_regexp_length, max_hyperscan_regexp_total_length);
    }

    static void vectorVector(
        const ColumnString::Chars & haystack_data,
        const ColumnString::Offsets & haystack_offsets,
        const IColumn & needles_data,
        const ColumnArray::Offsets & needles_offsets,
        PaddedPODArray<ResultType> & res,
        PaddedPODArray<UInt64> & offsets,
        std::optional<UInt32> edit_distance,
        bool allow_hyperscan,
        size_t max_hyperscan_regexp_length,
        size_t max_hyperscan_regexp_total_length)
    {
        if (!allow_hyperscan)
            throw Exception(ErrorCodes::FUNCTION_NOT_ALLOWED, "Hyperscan functions are disabled, because setting 'allow_hyperscan' is set to 0");
#if USE_VECTORSCAN
        offsets.resize(haystack_offsets.size());
        size_t prev_haystack_offset = 0;

        const ColumnString * needles_data_string = checkAndGetColumn<ColumnString>(&needles_data);
        const ColumnString::Offsets & needles_data_string_offsets = needles_data_string->getOffsets();
        const ColumnString::Chars & needles_data_string_chars = needles_data_string->getChars();

        std::vector<std::string_view> needles;

        size_t prev_needles_offsets_offset = 0;
        size_t prev_needles_data_offset = 0;

        for (size_t i = 0; i < haystack_offsets.size(); ++i)
        {
            needles.reserve(needles_offsets[i] - prev_needles_offsets_offset);

            for (size_t j = prev_needles_offsets_offset; j < needles_offsets[i]; ++j)
            {
                const auto * p = reinterpret_cast<const char *>(needles_data_string_chars.data()) + prev_needles_data_offset;
                auto sz = needles_data_string_offsets[j] - prev_needles_data_offset - 1;
                needles.emplace_back(std::string_view(p, sz));
                prev_needles_data_offset = needles_data_string_offsets[j];
            }

            checkHyperscanRegexp(needles, max_hyperscan_regexp_length, max_hyperscan_regexp_total_length);

            const auto & hyperscan_regex = MultiRegexps::get</*SaveIndices=*/true, WithEditDistance>(needles, edit_distance);
            hs_scratch_t * scratch = nullptr;
            hs_error_t err = hs_clone_scratch(hyperscan_regex->getScratch(), &scratch);

            if (err != HS_SUCCESS)
                throw Exception("Could not clone scratch space for hyperscan", ErrorCodes::CANNOT_ALLOCATE_MEMORY);

            MultiRegexps::ScratchPtr smart_scratch(scratch);

            auto on_match = [](unsigned int id,
                               unsigned long long /* from */, // NOLINT
                               unsigned long long /* to */, // NOLINT
                               unsigned int /* flags */,
                               void * context) -> int
            {
                static_cast<PaddedPODArray<ResultType>*>(context)->push_back(id);
                return 0;
            };

            const size_t cur_haystack_length = haystack_offsets[i] - prev_haystack_offset - 1;

            /// vectorscan restriction.
            if (cur_haystack_length > std::numeric_limits<UInt32>::max())
                throw Exception("Too long string to search", ErrorCodes::TOO_MANY_BYTES);

            /// scan, check, update the offsets array and the offset of haystack.
            err = hs_scan(
                hyperscan_regex->getDB(),
                reinterpret_cast<const char *>(haystack_data.data()) + prev_haystack_offset,
                cur_haystack_length,
                0,
                smart_scratch.get(),
                on_match,
                &res);
            if (err != HS_SUCCESS)
                throw Exception("Failed to scan with vectorscan", ErrorCodes::HYPERSCAN_CANNOT_SCAN_TEXT);

            offsets[i] = res.size();

            prev_haystack_offset = haystack_offsets[i];
            prev_needles_offsets_offset = needles_offsets[i];
            needles.clear();
        }
#else
        (void)haystack_data;
        (void)haystack_offsets;
        (void)needles_data;
        (void)needles_offsets;
        (void)res;
        (void)offsets;
        (void)edit_distance;
        (void)max_hyperscan_regexp_length;
        (void)max_hyperscan_regexp_total_length;
        throw Exception(
            "multi-search all indices is not implemented when vectorscan is off",
            ErrorCodes::NOT_IMPLEMENTED);
#endif // USE_VECTORSCAN
    }
};

}
