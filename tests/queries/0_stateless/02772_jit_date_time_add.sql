SET compile_expressions = 1;
SET min_count_to_compile_expression = 0;

SELECT DISTINCT result FROM (SELECT toStartOfFifteenMinutes(toDateTime(toStartOfFifteenMinutes(toDateTime(1000.0001220703125) + (number * 65536))) + (number * 9223372036854775807)) AS result FROM system.numbers LIMIT 1048576) ORDER BY result DESC NULLS FIRST FORMAT Null; -- { serverError 407 }
SELECT DISTINCT result FROM (SELECT toStartOfFifteenMinutes(toDateTime(toStartOfFifteenMinutes(toDateTime(1000.0001220703125) + (number * 65536))) + toInt64(number * 9223372036854775807)) AS result FROM system.numbers LIMIT 1048576) ORDER BY result DESC NULLS FIRST FORMAT Null;
SELECT round(round(round(round(round(100)), round(round(round(round(NULL), round(65535)), toTypeName(now() + 9223372036854775807) LIKE 'DateTime%DateTime%DateTime%DateTime%', round(-2)), 255), round(NULL))));
