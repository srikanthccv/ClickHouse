CREATE TABLE test1__fuzz_37 (`i` Date) ENGINE = MergeTree ORDER BY i;
insert into test1__fuzz_37 values ('2020-10-10');
SELECT count() FROM test1__fuzz_37 GROUP BY dictHas(NULL, (dictHas(NULL, (('', materialize(NULL)), materialize(NULL))), 'KeyKey')), dictHas('test_dictionary', tuple(materialize('Ke\0'))), tuple(dictHas(NULL, (tuple('Ke\0Ke\0Ke\0Ke\0Ke\0Ke\0\0\0\0Ke\0'), materialize(NULL)))), 'test_dicti\0nary', (('', materialize(NULL)), dictHas(NULL, (dictHas(NULL, tuple(materialize(NULL))), 'KeyKeyKeyKeyKeyKeyKeyKey')), materialize(NULL)); -- { serverError BAD_ARGUMENTS }

SELECT count() FROM test1__fuzz_37 GROUP BY dictHas('non_existing_dictionary', materialize('a')); -- { serverError BAD_ARGUMENTS }


