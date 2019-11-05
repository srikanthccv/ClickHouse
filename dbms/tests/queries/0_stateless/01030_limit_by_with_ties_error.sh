#!/usr/bin/env bash

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $CURDIR/../shell_config.sh

CLICKHOUSE_CLIENT=`echo ${CLICKHOUSE_CLIENT} | sed 's/'"--send_logs_level=${CLICKHOUSE_CLIENT_SERVER_LOGS_LEVEL}"'/--send_logs_level=none/g'`

$CLICKHOUSE_CLIENT --query="SELECT * FROM (SELECT number % 5 AS a, count() AS b, c FROM numbers(10) ARRAY JOIN [1,2] AS c GROUP BY a,c) AS table ORDER BY a LIMIT 3 WITH TIES BY a" 2>&1 | grep -q "Code: 490." && echo 'OK' || echo 'FAIL' ||:

