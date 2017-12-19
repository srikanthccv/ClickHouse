
export CLICKHOUSE_BINARY=${CLICKHOUSE_BINARY:="clickhouse"}
export CLICKHOUSE_CLIENT=${CLICKHOUSE_CLIENT:="${CLICKHOUSE_BINARY}-client"}
export CLICKHOUSE_LOCAL=${CLICKHOUSE_LOCAL:="${CLICKHOUSE_BINARY}-local"}

export CLICKHOUSE_CONFIG=${CLICKHOUSE_CONFIG:="/etc/clickhouse-server/config.xml"}
export CLICKHOUSE_EXTRACT_CONFIG=${CLICKHOUSE_EXTRACT_CONFIG:="$CLICKHOUSE_BINARY-extract-from-config -c $CLICKHOUSE_CONFIG"}

export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:="localhost"}
export CLICKHOUSE_PORT_TCP=${CLICKHOUSE_PORT_TCP:=`${CLICKHOUSE_EXTRACT_CONFIG} --try --key=tcp_port 2>/dev/null`} 2>/dev/null
export CLICKHOUSE_PORT_TCP=${CLICKHOUSE_PORT_TCP:="9000"}
export CLICKHOUSE_PORT_TCP_SSL=${CLICKHOUSE_PORT_TCP_SSL:=`${CLICKHOUSE_EXTRACT_CONFIG} --try --key=tcp_ssl_port 2>/dev/null`} 2>/dev/null
export CLICKHOUSE_PORT_TCP_SSL=${CLICKHOUSE_PORT_TCP_SSL:="9440"}
export CLICKHOUSE_PORT_HTTP=${CLICKHOUSE_PORT_HTTP:=`${CLICKHOUSE_EXTRACT_CONFIG} --key=http_port 2>/dev/null`}
export CLICKHOUSE_PORT_HTTP=${CLICKHOUSE_PORT_HTTP:="8123"}
export CLICKHOUSE_PORT_HTTPS=${CLICKHOUSE_PORT_HTTPS:=`${CLICKHOUSE_EXTRACT_CONFIG} --try --key=https_port 2>/dev/null`} 2>/dev/null
export CLICKHOUSE_PORT_HTTPS=${CLICKHOUSE_PORT_HTTPS:="8443"}
export CLICKHOUSE_PORT_HTTP_PROTO=${CLICKHOUSE_PORT_HTTP_PROTO:="http"}
export CLICKHOUSE_URL=${CLICKHOUSE_URL:="${CLICKHOUSE_PORT_HTTP_PROTO}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT_HTTP}/"}
export CLICKHOUSE_CURL=${CLICKHOUSE_CURL:="curl --max-time 5"}
