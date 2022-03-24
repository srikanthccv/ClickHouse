#!/usr/bin/env bash
# Tags: deadlock

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

set -e

$CLICKHOUSE_CLIENT --query "DROP TABLE IF EXISTS test1";
$CLICKHOUSE_CLIENT --query "DROP TABLE IF EXISTS test2";
$CLICKHOUSE_CLIENT --query "CREATE TABLE test1 (x UInt8) ENGINE = MergeTree ORDER BY x";
$CLICKHOUSE_CLIENT --query "CREATE TABLE test2 (x UInt8) ENGINE = MergeTree ORDER BY x";

function thread1()
{
    $CLICKHOUSE_CLIENT --query "RENAME TABLE test1 TO test_tmp, test2 TO test1, test_tmp TO test2"
}

function thread2()
{
    $CLICKHOUSE_CLIENT --query "SELECT * FROM test1 UNION ALL SELECT * FROM test2" --format Null
}

function thread3()
{
    # NOTE: database = $CLICKHOUSE_DATABASE is unwanted
    $CLICKHOUSE_CLIENT --query "SELECT * FROM system.tables" --format Null
}

export -f thread1
export -f thread2
export -f thread3

TIMEOUT=10

clickhouse_client_loop_timeout $TIMEOUT thread1 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread2 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread3 2> /dev/null &

clickhouse_client_loop_timeout $TIMEOUT thread1 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread2 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread3 2> /dev/null &

clickhouse_client_loop_timeout $TIMEOUT thread1 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread2 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread3 2> /dev/null &

clickhouse_client_loop_timeout $TIMEOUT thread1 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread2 2> /dev/null &
clickhouse_client_loop_timeout $TIMEOUT thread3 2> /dev/null &

wait
sleep 1

$CLICKHOUSE_CLIENT -q "DROP TABLE test1"
$CLICKHOUSE_CLIENT -q "DROP TABLE test2"
