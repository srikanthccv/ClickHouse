#!/usr/bin/env bash
# Tags: no-ubsan, no-fasttest

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh


${CLICKHOUSE_CLIENT} --multiquery --multiline --query="""

DROP TABLE IF EXISTS t_01411;
DROP TABLE IF EXISTS t_01411_num;

CREATE TABLE t_01411(
    str LowCardinality(String),
    arr Array(LowCardinality(String)) default [str]
) ENGINE = MergeTree()
ORDER BY tuple();

INSERT INTO t_01411 (str) SELECT concat('asdf', toString(number % 10000)) FROM numbers(1000000);

CREATE TABLE t_01411_num(
    num UInt8,
    arr Array(LowCardinality(Int64)) default [num]
) ENGINE = MergeTree()
ORDER BY tuple();

INSERT INTO t_01411_num (num) SELECT number % 1000 FROM numbers(100000);

"""

function go()
{

${CLICKHOUSE_CLIENT} --multiquery --multiline --query="""

SELECT count() FROM t_01411 WHERE str = 'asdf337';
SELECT count() FROM t_01411 WHERE arr[1] = 'asdf337';
SELECT count() FROM t_01411 WHERE has(arr, 'asdf337');
SELECT count() FROM t_01411 WHERE indexOf(arr, 'asdf337') > 0;

SELECT count() FROM t_01411 WHERE arr[1] = str;
SELECT count() FROM t_01411 WHERE has(arr, str);
SELECT count() FROM t_01411 WHERE indexOf(arr, str) > 0;

SELECT count() FROM t_01411_num WHERE num = 42;
SELECT count() FROM t_01411_num WHERE arr[1] = 42;
SELECT count() FROM t_01411_num WHERE has(arr, 42);
SELECT count() FROM t_01411_num WHERE indexOf(arr, 42) > 0;

SELECT count() FROM t_01411_num WHERE arr[1] = num;
SELECT count() FROM t_01411_num WHERE has(arr, num);
SELECT count() FROM t_01411_num WHERE indexOf(arr, num) > 0;
SELECT count() FROM t_01411_num WHERE indexOf(arr, num % 337) > 0;

SELECT indexOf(['a', 'b', 'c'], toLowCardinality('a'));
SELECT indexOf(['a', 'b', NULL], toLowCardinality('a'));

"""
}

for i in `seq 1 32`; do go | grep -q "Exception" && echo 'FAIL' || echo 'OK' ||: & done

wait

${CLICKHOUSE_CLIENT} --multiquery --multiline --query="""
DROP TABLE IF EXISTS t_01411;
DROP TABLE IF EXISTS t_01411_num;
"""
