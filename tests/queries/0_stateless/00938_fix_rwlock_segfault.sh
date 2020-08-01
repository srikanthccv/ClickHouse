#!/usr/bin/env bash

# Test fix for issue #5066

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $CURDIR/../shell_config.sh

set -e

for _ in {1..100}; do \
$CLICKHOUSE_CLIENT -q "SELECT name FROM system.tables UNION ALL SELECT name FROM system.columns format Null";
done
