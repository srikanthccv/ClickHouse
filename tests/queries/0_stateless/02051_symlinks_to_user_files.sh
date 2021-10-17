#!/usr/bin/env bash
# Tags: no-fasttest

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# See 01658_read_file_to_string_column.sh
user_files_path=$(clickhouse-client --query "select _path,_file from file('nonexist.txt', 'CSV', 'val1 char')" 2>&1 | grep Exception | awk '{gsub("/nonexist.txt","",$9); print $9}')

mkdir -p "${user_files_path}/"
chmod 777 "${user_files_path}"

export FILE="test_symlink_${CLICKHOUSE_DATABASE}"

symlink_path=${user_files_path}/${FILE}
file_path=$CUR_DIR/${FILE}

chmod +w ${file_path}

function cleanup()
{
    rm ${symlink_path} ${file_path}
}
trap cleanup EXIT

touch ${file_path}
ln -s ${file_path} ${symlink_path}

${CLICKHOUSE_CLIENT} --query="insert into table function file('${symlink_path}', 'Values', 'a String') select 'OK'";
${CLICKHOUSE_CLIENT} --query="select * from file('${symlink_path}', 'Values', 'a String')";

