#!/usr/bin/env bash


CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

${CLICKHOUSE_CLIENT_BINARY}  --query "drop database if exists db_01939"
${CLICKHOUSE_CLIENT_BINARY}  --query "create database db_01939"

#create user by sql
${CLICKHOUSE_CLIENT_BINARY}  --query "drop user if exists u_01939"
${CLICKHOUSE_CLIENT_BINARY}  --query "create user u_01939 default database db_01939"

${CLICKHOUSE_CLIENT_BINARY}  --query "SELECT currentDatabase();"
${CLICKHOUSE_CLIENT_BINARY}  --user=u_01939 --query "SELECT currentDatabase();"

${CLICKHOUSE_CLIENT_BINARY}  --query "drop user u_01939 "
${CLICKHOUSE_CLIENT_BINARY}  --query "drop database db_01939"
