#!/bin/bash

# script allows to install configs for clickhouse server and clients required
# for testing (stateless and stateful tests)

set -x -e

DEST_SERVER_PATH="${1:-/etc/clickhouse-server}"
DEST_CLIENT_PATH="${2:-/etc/clickhouse-client}"
SRC_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "Going to install test configs from $SRC_PATH into DEST_SERVER_PATH"

mkdir -p $DEST_SERVER_PATH/config.d/
mkdir -p $DEST_SERVER_PATH/users.d/
mkdir -p $DEST_CLIENT_PATH

ln -s $SRC_PATH/zookeeper.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/listen.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/part_log.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/text_log.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/metric_log.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/custom_settings_prefixes.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/log_queries.xml $DEST_SERVER_PATH/users.d/
ln -s $SRC_PATH/readonly.xml $DEST_SERVER_PATH/users.d/
ln -s $SRC_PATH/access_management.xml $DEST_SERVER_PATH/users.d/
ln -s $SRC_PATH/ints_dictionary.xml $DEST_SERVER_PATH/
ln -s $SRC_PATH/strings_dictionary.xml $DEST_SERVER_PATH/
ln -s $SRC_PATH/decimals_dictionary.xml $DEST_SERVER_PATH/
ln -s $SRC_PATH/executable_dictionary.xml $DEST_SERVER_PATH/
ln -s $SRC_PATH/macros.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/disks.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/secure_ports.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/clusters.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/graphite.xml $DEST_SERVER_PATH/config.d/
ln -s $SRC_PATH/server.key $DEST_SERVER_PATH/
ln -s $SRC_PATH/server.crt $DEST_SERVER_PATH/
ln -s $SRC_PATH/dhparam.pem $DEST_SERVER_PATH/

# Retain any pre-existing config and allow ClickHouse to load it if required
ln -s --backup=simple --suffix=_original.xml \
   $SRC_PATH/query_masking_rules.xml $DEST_SERVER_PATH/config.d/

if [[ -n "$USE_POLYMORPHIC_PARTS" ]] && [[ "$USE_POLYMORPHIC_PARTS" -eq 1 ]]; then
    ln -s $SRC_PATH/polymorphic_parts.xml $DEST_SERVER_PATH/config.d/
fi
if [[ -n "$USE_DATABASE_ATOMIC" ]] && [[ "$USE_DATABASE_ATOMIC" -eq 1 ]]; then
    ln -s $SRC_PATH/database_atomic_configd.xml $DEST_SERVER_PATH/config.d/
    ln -s $SRC_PATH/database_atomic_usersd.xml $DEST_SERVER_PATH/users.d/
fi

ln -sf $SRC_PATH/client_config.xml $DEST_CLIENT_PATH/config.xml
