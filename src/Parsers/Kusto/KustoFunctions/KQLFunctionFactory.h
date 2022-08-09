#pragma once

#include <Parsers/IParserBase.h>
#include <Parsers/Kusto/KustoFunctions/IParserKQLFunction.h>
#include <unordered_map>
namespace DB
{
    enum class KQLFunctionValue : uint16_t
    {   none,
        timespan,
      //  datetime,
        ago,
        datetime_add,
        datetime_part,
        datetime_diff,
        dayofmonth,
        dayofweek,
        dayofyear,
        endofday,
        endofweek,
        endofyear,
        monthofyear,
        format_datetime,
        format_timespan,
        getmonth,
        getyear,
        hourofday,
        make_timespan,
        make_datetime,
        now,
        startofday,
        startofmonth,
        startofweek,
        startofyear,
        todatetime,
        totimespan,
        unixtime_microseconds_todatetime,
        unixtime_milliseconds_todatetime,
        unixtime_nanoseconds_todatetime,
        unixtime_seconds_todatetime,
        week_of_year,

        base64_encode_tostring,
        base64_encode_fromguid,
        base64_decode_tostring,
        base64_decode_toarray,
        base64_decode_toguid,
        countof,
        extract,
        extract_all,
        extractjson,
        has_any_index,
        indexof,
        isempty,
        isnotempty,
        isnotnull,
        isnull,
        parse_command_line,
        parse_csv,
        parse_json,
        parse_url,
        parse_urlquery,
        parse_version,
        replace_regex,
        reverse,
        split,
        strcat,
        strcat_delim,
        strcmp,
        strlen,
        strrep,
        substring,
        tolower,
        toupper,
        translate,
        trim,
        trim_end,
        trim_start,
        url_decode,
        url_encode,

        array_concat,
        array_iif,
        array_index_of,
        array_length,
        array_reverse,
        array_rotate_left,
        array_rotate_right,
        array_shift_left,
        array_shift_right,
        array_slice,
        array_sort_asc,
        array_sort_desc,
        array_split,
        array_sum,
        bag_keys,
        bag_merge,
        bag_remove_keys,
        jaccard_index,
        pack,
        pack_all,
        pack_array,
        repeat,
        set_difference,
        set_has_element,
        set_intersect,
        set_union,
        treepath,
        zip,

        tobool,
        todouble,
        toint,
        tostring,

        arg_max,
        arg_min,
        avg,
        avgif,
        binary_all_and,
        binary_all_or,
        binary_all_xor,
        buildschema,
        count,
        countif,
        dcount,
        dcountif,
        make_bag,
        make_bag_if,
        make_list,
        make_list_if,
        make_list_with_nulls,
        make_set,
        make_set_if,
        max,
        maxif,
        min,
        minif,
        percentiles,
        percentiles_array,
        percentilesw,
        percentilesw_array,
        stdev,
        stdevif,
        sum,
        sumif,
        take_any,
        take_anyif,
        variance,
        varianceif,

        series_fir,
        series_iir,
        series_fit_line,
        series_fit_line_dynamic,
        series_fit_2lines,
        series_fit_2lines_dynamic,
        series_outliers,
        series_periods_detect,
        series_periods_validate,
        series_stats_dynamic,
        series_stats,
        series_fill_backward,
        series_fill_const,
        series_fill_forward,
        series_fill_linear,

        ipv4_compare,
        ipv4_is_in_range,
        ipv4_is_match,
        ipv4_is_private,
        ipv4_netmask_suffix,
        parse_ipv4,
        parse_ipv4_mask,
        ipv6_compare,
        ipv6_is_match,
        parse_ipv6,
        parse_ipv6_mask,
        format_ipv4,
        format_ipv4_mask,

        binary_and,
        binary_not,
        binary_or,
        binary_shift_left,
        binary_shift_right,
        binary_xor,
        bitset_count_ones,

        bin,
        bin_at,

        datatype_bool,
        datatype_datetime,
        datatype_dynamic,
        datatype_guid,
        datatype_int,
        datatype_long,
        datatype_real,
        datatype_string,
        datatype_timespan,
        datatype_decimal
    };
class KQLFunctionFactory
{
public:
   static std::unique_ptr<IParserKQLFunction> get(String &kql_function);

protected:
    static std::unordered_map <String,KQLFunctionValue> kql_functions;
};

}

