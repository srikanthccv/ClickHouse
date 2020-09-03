export CLICKHOUSE_DATABASE=${CLICKHOUSE_DATABASE:="test"}
export CLICKHOUSE_CLIENT_SERVER_LOGS_LEVEL=${CLICKHOUSE_CLIENT_SERVER_LOGS_LEVEL:="warning"}
[ -n "$CLICKHOUSE_CONFIG_CLIENT" ] && CLICKHOUSE_CLIENT_OPT0+=" --config-file=${CLICKHOUSE_CONFIG_CLIENT} "
[ -n "${CLICKHOUSE_HOST}" ] && CLICKHOUSE_CLIENT_OPT0+=" --host=${CLICKHOUSE_HOST} "
[ -n "${CLICKHOUSE_PORT_TCP}" ] && CLICKHOUSE_CLIENT_OPT0+=" --port=${CLICKHOUSE_PORT_TCP} "
[ -n "${CLICKHOUSE_CLIENT_SERVER_LOGS_LEVEL}" ] && CLICKHOUSE_CLIENT_OPT0+=" --send_logs_level=${CLICKHOUSE_CLIENT_SERVER_LOGS_LEVEL} "
[ -n "${CLICKHOUSE_DATABASE}" ] && CLICKHOUSE_CLIENT_OPT0+=" --database=${CLICKHOUSE_DATABASE} "

export CLICKHOUSE_BINARY=${CLICKHOUSE_BINARY:="clickhouse"}
[ -x "$CLICKHOUSE_BINARY-client" ] && CLICKHOUSE_CLIENT_BINARY=${CLICKHOUSE_CLIENT_BINARY:=$CLICKHOUSE_BINARY-client}
[ -x "$CLICKHOUSE_BINARY" ] && CLICKHOUSE_CLIENT_BINARY=${CLICKHOUSE_CLIENT_BINARY:=$CLICKHOUSE_BINARY client}
export CLICKHOUSE_CLIENT_BINARY=${CLICKHOUSE_CLIENT_BINARY:=$CLICKHOUSE_BINARY-client}
export CLICKHOUSE_CLIENT=${CLICKHOUSE_CLIENT:="$CLICKHOUSE_CLIENT_BINARY ${CLICKHOUSE_CLIENT_OPT0} ${CLICKHOUSE_CLIENT_OPT}"}
[ -x "${CLICKHOUSE_BINARY}-local" ] && CLICKHOUSE_LOCAL=${CLICKHOUSE_LOCAL:="${CLICKHOUSE_BINARY}-local"}
[ -x "${CLICKHOUSE_BINARY}" ] && CLICKHOUSE_LOCAL=${CLICKHOUSE_LOCAL:="${CLICKHOUSE_BINARY} local"}
export CLICKHOUSE_LOCAL=${CLICKHOUSE_LOCAL:="${CLICKHOUSE_BINARY}-local"}
export CLICKHOUSE_OBFUSCATOR=${CLICKHOUSE_OBFUSCATOR:="${CLICKHOUSE_BINARY}-obfuscator"}
export CLICKHOUSE_BENCHMARK=${CLICKHOUSE_BENCHMARK:="${CLICKHOUSE_BINARY}-benchmark"}

export CLICKHOUSE_CONFIG=${CLICKHOUSE_CONFIG:="/etc/clickhouse-server/config.xml"}
export CLICKHOUSE_CONFIG_CLIENT=${CLICKHOUSE_CONFIG_CLIENT:="/etc/clickhouse-client/config.xml"}

[ -x "${CLICKHOUSE_BINARY}-extract-from-config" ] && CLICKHOUSE_EXTRACT_CONFIG=${CLICKHOUSE_EXTRACT_CONFIG:="$CLICKHOUSE_BINARY-extract-from-config --config=$CLICKHOUSE_CONFIG"}
[ -x "${CLICKHOUSE_BINARY}" ] && CLICKHOUSE_EXTRACT_CONFIG=${CLICKHOUSE_EXTRACT_CONFIG:="$CLICKHOUSE_BINARY extract-from-config --config=$CLICKHOUSE_CONFIG"}
export CLICKHOUSE_EXTRACT_CONFIG=${CLICKHOUSE_EXTRACT_CONFIG:="$CLICKHOUSE_BINARY-extract-from-config --config=$CLICKHOUSE_CONFIG"}

[ -x "${CLICKHOUSE_BINARY}-format" ] && CLICKHOUSE_FORMAT=${CLICKHOUSE_FORMAT:="$CLICKHOUSE_BINARY-format"}
[ -x "${CLICKHOUSE_BINARY}" ] && CLICKHOUSE_FORMAT=${CLICKHOUSE_FORMAT:="$CLICKHOUSE_BINARY format"}
export CLICKHOUSE_FORMAT=${CLICKHOUSE_FORMAT:="$CLICKHOUSE_BINARY-format"}

export CLICKHOUSE_CONFIG_GREP=${CLICKHOUSE_CONFIG_GREP:="/etc/clickhouse-server/preprocessed/config.xml"}

export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:="localhost"}
export CLICKHOUSE_PORT_TCP=${CLICKHOUSE_PORT_TCP:=$(${CLICKHOUSE_EXTRACT_CONFIG} --try --key=tcp_port 2>/dev/null)} 2>/dev/null
export CLICKHOUSE_PORT_TCP=${CLICKHOUSE_PORT_TCP:="9000"}
export CLICKHOUSE_PORT_TCP_SECURE=${CLICKHOUSE_PORT_TCP_SECURE:=$(${CLICKHOUSE_EXTRACT_CONFIG} --try --key=tcp_port_secure 2>/dev/null)} 2>/dev/null
export CLICKHOUSE_PORT_TCP_SECURE=${CLICKHOUSE_PORT_TCP_SECURE:="9440"}
export CLICKHOUSE_PORT_HTTP=${CLICKHOUSE_PORT_HTTP:=$(${CLICKHOUSE_EXTRACT_CONFIG} --key=http_port 2>/dev/null)}
export CLICKHOUSE_PORT_HTTP=${CLICKHOUSE_PORT_HTTP:="8123"}
export CLICKHOUSE_PORT_HTTPS=${CLICKHOUSE_PORT_HTTPS:=$(${CLICKHOUSE_EXTRACT_CONFIG} --try --key=https_port 2>/dev/null)} 2>/dev/null
export CLICKHOUSE_PORT_HTTPS=${CLICKHOUSE_PORT_HTTPS:="8443"}
export CLICKHOUSE_PORT_HTTP_PROTO=${CLICKHOUSE_PORT_HTTP_PROTO:="http"}

# Add database to url params
if [ -n "${CLICKHOUSE_URL_PARAMS}" ]
then
  export CLICKHOUSE_URL_PARAMS="${CLICKHOUSE_URL_PARAMS}&database=${CLICKHOUSE_DATABASE}"
else
  export CLICKHOUSE_URL_PARAMS="database=${CLICKHOUSE_DATABASE}"
fi

export CLICKHOUSE_URL=${CLICKHOUSE_URL:="${CLICKHOUSE_PORT_HTTP_PROTO}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT_HTTP}/"}
export CLICKHOUSE_URL_HTTPS=${CLICKHOUSE_URL_HTTPS:="https://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT_HTTPS}/"}

# Add url params to url
if [ -n "${CLICKHOUSE_URL_PARAMS}" ]
then
  export CLICKHOUSE_URL="${CLICKHOUSE_URL}?${CLICKHOUSE_URL_PARAMS}"
  export CLICKHOUSE_URL_HTTPS="${CLICKHOUSE_URL_HTTPS}?${CLICKHOUSE_URL_PARAMS}"
fi

export CLICKHOUSE_PORT_INTERSERVER=${CLICKHOUSE_PORT_INTERSERVER:=$(${CLICKHOUSE_EXTRACT_CONFIG} --try --key=interserver_http_port 2>/dev/null)} 2>/dev/null
export CLICKHOUSE_PORT_INTERSERVER=${CLICKHOUSE_PORT_INTERSERVER:="9009"}
export CLICKHOUSE_URL_INTERSERVER=${CLICKHOUSE_URL_INTERSERVER:="${CLICKHOUSE_PORT_HTTP_PROTO}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT_INTERSERVER}/"}

export CLICKHOUSE_CURL_COMMAND=${CLICKHOUSE_CURL_COMMAND:="curl"}
export CLICKHOUSE_CURL_TIMEOUT=${CLICKHOUSE_CURL_TIMEOUT:="10"}
export CLICKHOUSE_CURL=${CLICKHOUSE_CURL:="${CLICKHOUSE_CURL_COMMAND} -q --max-time ${CLICKHOUSE_CURL_TIMEOUT}"}
export CLICKHOUSE_TMP=${CLICKHOUSE_TMP:="."}
mkdir -p ${CLICKHOUSE_TMP}

function clickhouse_client_removed_host_parameter()
{
    # removing only `--host=value` and `--host value` (removing '-hvalue' feels to dangerous) with python regex.
    # bash regex magic is arcane, but version dependant and weak; sed or awk are not really portable.
    $(echo "$CLICKHOUSE_CLIENT"  | python -c "import sys, re; print re.sub('--host(\s+|=)[^\s]+', '', sys.stdin.read())") "$@"
}
