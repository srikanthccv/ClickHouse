#!/usr/bin/env bash
# Tags: long

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

set -e

$CLICKHOUSE_CLIENT --query "DROP TABLE IF EXISTS mt";
$CLICKHOUSE_CLIENT --query "CREATE TABLE mt (n Int64) ENGINE=MergeTree ORDER BY n SETTINGS old_parts_lifetime=0";


function begin_commit_readonly()
{
    $CLICKHOUSE_CLIENT --multiquery --query "
            BEGIN TRANSACTION;
            COMMIT;";
}

function begin_rollback_readonly()
{
    $CLICKHOUSE_CLIENT --multiquery --query "
            BEGIN TRANSACTION;
            ROLLBACK;";
}

function begin_insert_commit()
{
    $CLICKHOUSE_CLIENT --multiquery --query "
            BEGIN TRANSACTION;
            INSERT INTO mt VALUES ($RANDOM);
            COMMIT;";
}

function introspection()
{
    $CLICKHOUSE_CLIENT -q "SELECT * FROM system.transactions FORMAT Null"
    $CLICKHOUSE_CLIENT -q "SELECT transactionLatestSnapshot(), transactionOldestSnapshot() FORMAT Null"
}

export -f begin_commit_readonly
export -f begin_rollback_readonly
export -f begin_insert_commit
export -f introspection

TIMEOUT=20

clickhouse_client_loop_timeout $TIMEOUT begin_commit_readonly &
clickhouse_client_loop_timeout $TIMEOUT begin_rollback_readonly &
clickhouse_client_loop_timeout $TIMEOUT begin_insert_commit &
clickhouse_client_loop_timeout $TIMEOUT introspection &

wait

$CLICKHOUSE_CLIENT --query "DROP TABLE mt";
