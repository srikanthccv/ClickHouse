#!/bin/bash
set -e

dockerd --host=unix:///var/run/docker.sock --host=tcp://0.0.0.0:2375 &>/var/log/somefile &

set +e
reties=0
while true; do
    docker info &>/dev/null && break
    reties=$[$reties+1]
    if [[ $reties -ge 100 ]]; then # 10 sec max
        echo "Can't start docker daemon, timeout exceeded." >&2
        exit 1;
    fi
    sleep 0.1
done
set -e

echo "Start tests"
export CLICKHOUSE_TESTS_SERVER_BIN_PATH=/clickhouse
export CLICKHOUSE_TESTS_CLIENT_BIN_PATH=/clickhouse
export CLICKHOUSE_TESTS_BASE_CONFIG_DIR=/clickhouse-base-config
export CLICKHOUSE_ODBC_BRIDGE_BINARY_PATH=/clickhouse-odbc-bridge

#backward compatibility with older stables
ln -s /clickhouse-base-config /clickhouse-config

cd /ClickHouse/tests/integration
exec "$@"
