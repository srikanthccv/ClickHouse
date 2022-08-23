#include <Parsers/tests/gtest_common.h>

#include <Parsers/Kusto/ParserKQLQuery.h>

INSTANTIATE_TEST_SUITE_P(ParserKQLQuery, ParserTest,
    ::testing::Combine(
        ::testing::Values(std::make_shared<DB::ParserKQLQuery>()),
        ::testing::ValuesIn(std::initializer_list<ParserTestCase>{
        {
            "print Quine = base64_encode_fromguid('ae3133f2-6e22-49ae-b06a-16e6a9b212eb')",
            "SELECT base64Encode('ae3133f2-6e22-49ae-b06a-16e6a9b212eb') AS Quine"
        },
        {
            "print base64_decode_toguid('YWUzMTMzZjItNmUyMi00OWFlLWIwNmEtMTZlNmE5YjIxMmVi')",
            "SELECT base64Decode('YWUzMTMzZjItNmUyMi00OWFlLWIwNmEtMTZlNmE5YjIxMmVi')"
        },
        {
            "print base64_decode_toarray('S3VzdG8=')",
            "SELECT arrayMap(x -> reinterpretAsUInt8(x), splitByRegexp('', base64Decode('S3VzdG8=')))"
        },
        {
            "print replace_regex('Hello, World!', '.', '\\0\\0')",
            "SELECT replaceRegexpAll('Hello, World!', '.', '\\0\\0')"
        },
        {
            "print idx = has_any_index('this is an example', dynamic(['this', 'example'])) ",
            "SELECT if(empty(['this', 'example']), -1, indexOf(arrayMap(x -> (x IN splitByChar(' ', 'this is an example')), if(empty(['this', 'example']), [''], arrayMap(x -> toString(x), ['this', 'example']))), 1) - 1) AS idx"
        },
        {
            "print idx = has_any_index('this is an example', dynamic([]))",
            "SELECT if(empty([]), -1, indexOf(arrayMap(x -> (x IN splitByChar(' ', 'this is an example')), if(empty([]), [''], arrayMap(x -> toString(x), []))), 1) - 1) AS idx"
        },
        {
            "print translate('krasp', 'otsku', 'spark')",
            "SELECT if(length('otsku') = 0, '', translate('spark', 'krasp', multiIf(length('otsku') = 0, 'krasp', (length('krasp') - length('otsku')) > 0, concat('otsku', repeat(substr('otsku', length('otsku'), 1), toUInt16(length('krasp') - length('otsku')))), (length('krasp') - length('otsku')) < 0, substr('otsku', 1, length('krasp')), 'otsku')))"
        },
        {
            "print trim_start('[^\\w]+', strcat('-  ','Te st1','// $'))",
            "SELECT if((replaceRegexpOne(concat('start_random_str_', concat('-  ', 'Te st1', '// $')) AS src, concat('start_random_str_', '[^\\\\w]+'), '') AS dst) = src, concat('-  ', 'Te st1', '// $'), dst)"
        },
        {
            "print trim_end('.com', 'bing.com')",
            "SELECT if((replaceRegexpOne(concat('bing.com', '_end_random_str') AS src, concat('.com', '_end_random_str'), '') AS dst) = src, 'bing.com', dst)"
        },
        {
            "print trim('--', '--https://bing.com--')",
            "SELECT if((replaceRegexpOne(concat(if((replaceRegexpOne(concat('start_random_str_', '--https://bing.com--') AS srcl, concat('start_random_str_', '--'), '') AS dstl) = srcl, '--https://bing.com--', dstl), '_end_random_str') AS srcr, concat('--', '_end_random_str'), '') AS dstr) = srcr, if(dstl = srcl, '--https://bing.com--', dstl), dstr)"
        },
        {
            "print bool(1)",
            "SELECT toBool(1)"
        },
        {
            "print datetime(2015-12-31 23:59:59.9)",
            "SELECT parseDateTime64BestEffortOrNull('2015-12-31 23:59:59.9', 9, 'UTC')"
        },
        {
            "print datetime(\"2015-12-31 23:59:59.9\")",
            "SELECT parseDateTime64BestEffortOrNull('2015-12-31 23:59:59.9', 9, 'UTC')"
        },
        {
            "print datetime('2015-12-31 23:59:59.9')",
            "SELECT parseDateTime64BestEffortOrNull('2015-12-31 23:59:59.9', 9, 'UTC')"
        },
        {
            "print guid(74be27de-1e4e-49d9-b579-fe0b331d3642)",
            "SELECT toUUID('74be27de-1e4e-49d9-b579-fe0b331d3642')"
        },
        {
            "print guid('74be27de-1e4e-49d9-b579-fe0b331d3642')",
            "SELECT toUUID('74be27de-1e4e-49d9-b579-fe0b331d3642')"
        },
        {
            "print guid('74be27de1e4e49d9b579fe0b331d3642')",
            "SELECT toUUID('74be27de1e4e49d9b579fe0b331d3642')"
        },
        {
            "print int(32.5)",
            "SELECT toInt32(32.5)"
        },
        {
            "print long(32.5)",
            "SELECT toInt64(32.5)"
        },
        {
            "print real(32.5)",
            "SELECT toFloat64(32.5)"
        },
        {
            "print time('1.22:34:8.128')",
            "SELECT 167648."
        },
        {
            "print time('1d')",
            "SELECT 86400."
        },
        {
            "print time('1.5d')",
            "SELECT 129600."
        },
        {
            "print timespan('1.5d')",
            "SELECT 129600."
        },
        {
            "print res = bin_at(6.5, 2.5, 7)",
            "SELECT toFloat64(7) + (toInt64(((toFloat64(6.5) - toFloat64(7)) / 2.5) + -1) * 2.5) AS res"
        },
        {
            "print res = bin_at(1h, 1d, 12h)",
            "SELECT concat(toString(toInt32(((toFloat64(43200.) + (toInt64(((toFloat64(3600.) - toFloat64(43200.)) / 86400) + -1) * 86400)) AS x) / 3600)), ':', toString(toInt32((x % 3600) / 60)), ':', toString(toInt32((x % 3600) % 60))) AS res"
        },
        {
            "print res = bin_at(datetime(2017-05-15 10:20:00.0), 1d, datetime(1970-01-01 12:00:00.0))",
            "SELECT parseDateTime64BestEffortOrNull(toFloat64(parseDateTime64BestEffortOrNull('1970-01-01 12:00:00.0', 9, 'UTC')) + (toInt64(((toFloat64(parseDateTime64BestEffortOrNull('2017-05-15 10:20:00.0', 9, 'UTC')) - toFloat64(parseDateTime64BestEffortOrNull('1970-01-01 12:00:00.0', 9, 'UTC'))) / 86400) + 0) * 86400), 9, 'UTC') AS res"
        },
        {
            "print bin(4.5, 1)",
            "SELECT toInt64(toFloat64(4.5) / 1) * 1"
        },
        {
            "print bin(time(16d), 7d)",
            "SELECT concat(toString(toInt32(((toInt64(toFloat64(1382400.) / 604800) * 604800) AS x) / 3600)), ':', toString(toInt32((x % 3600) / 60)), ':', toString(toInt32((x % 3600) % 60)))"
        },
        {
            "print bin(datetime(1970-05-11 13:45:07), 1d)",
            "SELECT parseDateTime64BestEffortOrNull(toInt64(toFloat64(parseDateTime64BestEffortOrNull('1970-05-11 13:45:07', 9, 'UTC')) / 86400) * 86400, 9, 'UTC')"
        }

})));   
