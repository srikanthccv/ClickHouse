#!/usr/bin/env bash
# Tags: no-fasttest, no-parallel

# set -x

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh


$CLICKHOUSE_CLIENT -nm --query """
DROP TABLE IF EXISTS test;
CREATE TABLE test (a Int32, b String)
ENGINE = MergeTree() ORDER BY tuple()
SETTINGS disk = disk_s3disk(type = cache, max_size = '100Ki', path = ${CLICKHOUSE_TEST_UNIQUE_NAME}, disk = s3disk);
""" 2>&1 | grep -q "Disk with name \`s3disk\` already exist" && echo 'OK' || echo 'FAIL'

disk_name="${CLICKHOUSE_TEST_UNIQUE_NAME}"

$CLICKHOUSE_CLIENT -nm --query """
SELECT count() FROM system.disks WHERE name = '$disk_name'
"""

$CLICKHOUSE_CLIENT -nm --query """
DROP TABLE IF EXISTS test;
CREATE TABLE test (a Int32, b String)
ENGINE = MergeTree() ORDER BY tuple()
SETTINGS disk = disk_$disk_name(type = cache, max_size = '100Ki', path = ${CLICKHOUSE_TEST_UNIQUE_NAME}, disk = s3disk);
"""

$CLICKHOUSE_CLIENT -nm --query """
SELECT count() FROM system.disks WHERE name = '$disk_name'
"""
