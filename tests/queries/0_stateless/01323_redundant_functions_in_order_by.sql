DROP TABLE IF EXISTS test;

CREATE TABLE test (key UInt64, a UInt8, b String, c Float64) ENGINE = MergeTree() ORDER BY key;
INSERT INTO test SELECT number, number, toString(number), number from numbers(4);

set enable_debug_queries = 1;
set optimize_redundant_functions_in_order_by = 1;

SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(x));
SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(exp(x)));
SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY exp(x), x);
SELECT * FROM (SELECT number + 2 AS key FROM numbers(4)) s FULL JOIN test t USING(key) ORDER BY s.key, t.key;
SELECT key, a FROM test ORDER BY key, a, exp(key + a);
SELECT key, a FROM test ORDER BY key, exp(key + a);
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(x));
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(exp(x)));
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY exp(x), x);
analyze SELECT * FROM (SELECT number + 2 AS key FROM numbers(4)) s FULL JOIN test t USING(key) ORDER BY s.key, t.key;
analyze SELECT key, a FROM test ORDER BY key, a, exp(key + a);
analyze SELECT key, a FROM test ORDER BY key, exp(key + a);

set optimize_redundant_functions_in_order_by = 0;

SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(x));
SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(exp(x)));
SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY exp(x), x);
SELECT * FROM (SELECT number + 2 AS key FROM numbers(4)) s FULL JOIN test t USING(key) ORDER BY s.key, t.key;
SELECT key, a FROM test ORDER BY key, a, exp(key + a);
SELECT key, a FROM test ORDER BY key, exp(key + a);
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(x));
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY x, exp(exp(x)));
analyze SELECT groupArray(x) from (SELECT number as x FROM numbers(3) ORDER BY exp(x), x);
analyze SELECT * FROM (SELECT number + 2 AS key FROM numbers(4)) s FULL JOIN test t USING(key) ORDER BY s.key, t.key;
analyze SELECT key, a FROM test ORDER BY key, a, exp(key + a);
analyze SELECT key, a FROM test ORDER BY key, exp(key + a);

DROP TABLE test;
