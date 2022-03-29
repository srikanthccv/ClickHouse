#!/usr/bin/env bash
# Tags: no-parallel

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS async_inserts_02246"
${CLICKHOUSE_CLIENT} -q "DROP ROLE IF EXISTS r02246"
${CLICKHOUSE_CLIENT} -q "DROP USER IF EXISTS u02246"
${CLICKHOUSE_CLIENT} -q "DROP QUOTA IF EXISTS q02246"

${CLICKHOUSE_CLIENT} -q "CREATE TABLE async_inserts_02246(a UInt32, s String) ENGINE = Memory"

${CLICKHOUSE_CLIENT} -q "CREATE ROLE r02246"
${CLICKHOUSE_CLIENT} -q "CREATE USER u02246"
${CLICKHOUSE_CLIENT} -q "GRANT INSERT ON async_inserts_02246 TO r02246"
${CLICKHOUSE_CLIENT} -q "GRANT r02246 to u02246"
${CLICKHOUSE_CLIENT} -q "CREATE QUOTA q02246 FOR INTERVAL 1 HOUR MAX QUERY INSERTS = 2 TO r02246"

${CLICKHOUSE_CLIENT} --user u02246 --async_insert 1 -q "INSERT INTO async_inserts_02246 VALUES (1, 'a')"
${CLICKHOUSE_CLIENT} --user u02246 --async_insert 1 -q "INSERT INTO async_inserts_02246 VALUES (2, 'b')"
${CLICKHOUSE_CLIENT} --user u02246 --async_insert 1 -q "INSERT INTO async_inserts_02246 VALUES (3, 'c')" 2>&1 | grep -m1 -o QUOTA_EXPIRED

sleep 1.0

${CLICKHOUSE_CLIENT} -q "SELECT count() FROM async_inserts_02246"

${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS async_inserts_02246"
${CLICKHOUSE_CLIENT} -q "DROP ROLE IF EXISTS r02246"
${CLICKHOUSE_CLIENT} -q "DROP USER IF EXISTS u02246"
${CLICKHOUSE_CLIENT} -q "DROP QUOTA IF EXISTS q02246"
