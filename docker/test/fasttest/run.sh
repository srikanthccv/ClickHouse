#!/bin/bash
set -xeu
set -o pipefail
trap "exit" INT TERM
trap 'kill $(jobs -pr) ||:' EXIT

# This script is separated into two stages, cloning and everything else, so
# that we can run the "everything else" stage from the cloned source.
stage=${stage:-}

# A variable to pass additional flags to CMake.
# Here we explicitly default it to nothing so that bash doesn't complain about
# it being undefined. Also read it as array so that we can pass an empty list
# of additional variable to cmake properly, and it doesn't generate an extra
# empty parameter.
read -ra FASTTEST_CMAKE_FLAGS <<< "${FASTTEST_CMAKE_FLAGS:-}"

FASTTEST_WORKSPACE=$(readlink -f "${FASTTEST_WORKSPACE:-.}")
FASTTEST_SOURCE=$(readlink -f "${FASTTEST_SOURCE:-$FASTTEST_WORKSPACE/ch}")
FASTTEST_BUILD=$(readlink -f "${FASTTEST_BUILD:-${BUILD:-$FASTTEST_WORKSPACE/build}}")
FASTTEST_DATA=$(readlink -f "${FASTTEST_DATA:-$FASTTEST_WORKSPACE/db-fasttest}")
FASTTEST_OUTPUT=$(readlink -f "${FASTTEST_OUTPUT:-$FASTTEST_WORKSPACE}")
PATH="$FASTTEST_BUILD/programs:$FASTTEST_SOURCE/tests:$PATH"

# Export these variables, so that all subsequent invocations of the script
# use them, and not try to guess them anew, which leads to weird effects.
export FASTTEST_WORKSPACE
export FASTTEST_SOURCE
export FASTTEST_BUILD
export FASTTEST_DATA
export FASTTEST_OUT
export PATH

server_pid=none

function stop_server
{
    if ! kill -0 -- "$server_pid"
    then
        echo "ClickHouse server pid '$server_pid' is not running"
        return 0
    fi

    for _ in {1..60}
    do
        if ! pkill -f "clickhouse-server" && ! kill -- "$server_pid" ; then break ; fi
        sleep 1
    done

    if kill -0 -- "$server_pid"
    then
        pstree -apgT
        jobs
        echo "Failed to kill the ClickHouse server pid '$server_pid'"
        return 1
    fi

    server_pid=none
}

function start_server
{
    set -m # Spawn server in its own process groups
    clickhouse-server --config-file="$FASTTEST_DATA/config.xml" -- --path "$FASTTEST_DATA" --user_files_path "$FASTTEST_DATA/user_files" &>> "$FASTTEST_OUTPUT/server.log" &
    server_pid=$!
    set +m

    if [ "$server_pid" == "0" ]
    then
        echo "Failed to start ClickHouse server"
        # Avoid zero PID because `kill` treats it as our process group PID.
        server_pid="none"
        return 1
    fi

    for _ in {1..60}
    do
        if clickhouse-client --query "select 1" || ! kill -0 -- "$server_pid"
        then
            break
        fi
        sleep 1
    done

    if ! clickhouse-client --query "select 1"
    then
        echo "Failed to wait until ClickHouse server starts."
        server_pid="none"
        return 1
    fi

    if ! kill -0 -- "$server_pid"
    then
        echo "Wrong clickhouse server started: PID '$server_pid' we started is not running, but '$(pgrep -f clickhouse-server)' is running"
        server_pid="none"
        return 1
    fi

    echo "ClickHouse server pid '$server_pid' started and responded"
}

function clone_root
{
git clone https://github.com/ClickHouse/ClickHouse.git -- "$FASTTEST_SOURCE" | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/clone_log.txt"

(
cd "$FASTTEST_SOURCE"
if [ "$PULL_REQUEST_NUMBER" != "0" ]; then
    if git fetch origin "+refs/pull/$PULL_REQUEST_NUMBER/merge"; then
        git checkout FETCH_HEAD
        echo 'Clonned merge head'
    else
        git fetch
        git checkout "$COMMIT_SHA"
        echo 'Checked out to commit'
    fi
else
    if [ -v COMMIT_SHA ]; then
        git checkout "$COMMIT_SHA"
    fi
fi
)
}

function clone_submodules
{
(
cd "$FASTTEST_SOURCE"

SUBMODULES_TO_UPDATE=(contrib/boost contrib/zlib-ng contrib/libxml2 contrib/poco contrib/libunwind contrib/ryu contrib/fmtlib contrib/base64 contrib/cctz contrib/libcpuid contrib/double-conversion contrib/libcxx contrib/libcxxabi contrib/libc-headers contrib/lz4 contrib/zstd contrib/fastops contrib/rapidjson contrib/re2 contrib/sparsehash-c11)

git submodule sync
git submodule update --init --recursive "${SUBMODULES_TO_UPDATE[@]}"
git submodule foreach git reset --hard
git submodule foreach git checkout @ -f
git submodule foreach git clean -xfd
)
}

function run_cmake
{
CMAKE_LIBS_CONFIG=(
    "-DENABLE_LIBRARIES=0"
    "-DENABLE_TESTS=0"
    "-DENABLE_UTILS=0"
    "-DENABLE_EMBEDDED_COMPILER=0"
    "-DENABLE_THINLTO=0"
    "-DUSE_UNWIND=1"
)

# TODO remove this? we don't use ccache anyway. An option would be to download it
# from S3 simultaneously with cloning.
export CCACHE_DIR="$FASTTEST_WORKSPACE/ccache"
export CCACHE_BASEDIR="$FASTTEST_SOURCE"
export CCACHE_NOHASHDIR=true
export CCACHE_COMPILERCHECK=content
export CCACHE_MAXSIZE=15G

ccache --show-stats ||:
ccache --zero-stats ||:

mkdir "$FASTTEST_BUILD" ||:

(
cd "$FASTTEST_BUILD"
cmake "$FASTTEST_SOURCE" -DCMAKE_CXX_COMPILER=clang++-10 -DCMAKE_C_COMPILER=clang-10 "${CMAKE_LIBS_CONFIG[@]}" "${FASTTEST_CMAKE_FLAGS[@]}" | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/cmake_log.txt"
)
}

function build
{
(
cd "$FASTTEST_BUILD"
time ninja clickhouse-bundle | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/build_log.txt"
if [ "$COPY_CLICKHOUSE_BINARY_TO_OUTPUT" -eq "1" ]; then
    cp programs/clickhouse "$FASTTEST_OUTPUT/clickhouse"
fi
ccache --show-stats ||:
)
}

function configure
{
clickhouse-client --version
clickhouse-test --help

mkdir -p "$FASTTEST_DATA"{,/client-config}
cp -a "$FASTTEST_SOURCE/programs/server/"{config,users}.xml "$FASTTEST_DATA"
"$FASTTEST_SOURCE/tests/config/install.sh" "$FASTTEST_DATA" "$FASTTEST_DATA/client-config"
cp -a "$FASTTEST_SOURCE/programs/server/config.d/log_to_console.xml" "$FASTTEST_DATA/config.d"
# doesn't support SSL
rm -f "$FASTTEST_DATA/config.d/secure_ports.xml"
}

function run_tests
{
clickhouse-server --version
clickhouse-test --help

# Kill the server in case we are running locally and not in docker
stop_server ||:

start_server

TESTS_TO_SKIP=(
    00105_shard_collations
    00109_shard_totals_after_having
    00110_external_sort
    00302_http_compression
    00417_kill_query
    00436_convert_charset
    00490_special_line_separators_and_characters_outside_of_bmp
    00652_replicated_mutations_zookeeper
    00682_empty_parts_merge
    00701_rollup
    00834_cancel_http_readonly_queries_on_client_close
    00911_tautological_compare
    00926_multimatch
    00929_multi_match_edit_distance
    01031_mutations_interpreter_and_context
    01053_ssd_dictionary # this test mistakenly requires acces to /var/lib/clickhouse -- can't run this locally, disabled
    01083_expressions_in_engine_arguments
    01092_memory_profiler
    01098_msgpack_format
    01098_temporary_and_external_tables
    01103_check_cpu_instructions_at_startup # avoid dependency on qemu -- invonvenient when running locally
    01193_metadata_loading
    01238_http_memory_tracking              # max_memory_usage_for_user can interfere another queries running concurrently
    01251_dict_is_in_infinite_loop
    01259_dictionary_custom_settings_ddl
    01268_dictionary_direct_layout
    01280_ssd_complex_key_dictionary
    01281_group_by_limit_memory_tracking    # max_memory_usage_for_user can interfere another queries running concurrently
    01318_encrypt                           # Depends on OpenSSL
    01318_decrypt                           # Depends on OpenSSL
    01281_unsucceeded_insert_select_queries_counter
    01292_create_user
    01294_lazy_database_concurrent
    01305_replica_create_drop_zookeeper
    01354_order_by_tuple_collate_const
    01355_ilike
    01411_bayesian_ab_testing
    _orc_
    arrow
    avro
    base64
    brotli
    capnproto
    client
    ddl_dictionaries
    h3
    hashing
    hdfs
    java_hash
    json
    limit_memory
    live_view
    memory_leak
    memory_limit
    mysql
    odbc
    parallel_alter
    parquet
    protobuf
    secure
    sha256

    # Not sure why these two fail even in sequential mode. Disabled for now
    # to make some progress.
    00646_url_engine
    00974_query_profiler

    # Look at DistributedFilesToInsert, so cannot run in parallel.
    01457_DistributedFilesToInsert
)

time clickhouse-test -j 8 --order=random --no-long --testname --shard --zookeeper --skip "${TESTS_TO_SKIP[@]}" 2>&1 | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/test_log.txt"

# substr is to remove semicolon after test name
readarray -t FAILED_TESTS < <(awk '/FAIL|TIMEOUT|ERROR/ { print substr($3, 1, length($3)-1) }' "$FASTTEST_OUTPUT/test_log.txt" | tee "$FASTTEST_OUTPUT/failed-parallel-tests.txt")

# We will rerun sequentially any tests that have failed during parallel run.
# They might have failed because there was some interference from other tests
# running concurrently. If they fail even in seqential mode, we will report them.
# FIXME All tests that require exclusive access to the server must be
# explicitly marked as `sequential`, and `clickhouse-test` must detect them and
# run them in a separate group after all other tests. This is faster and also
# explicit instead of guessing.
if [[ -n "${FAILED_TESTS[*]}" ]]
then
    stop_server ||:

    # Clean the data so that there is no interference from the previous test run.
    rm -rf "$FASTTEST_DATA"/{{meta,}data,user_files} ||:

    start_server

    echo "Going to run again: ${FAILED_TESTS[*]}"

    clickhouse-test --order=random --no-long --testname --shard --zookeeper "${FAILED_TESTS[@]}" 2>&1 | ts '%Y-%m-%d %H:%M:%S' | tee -a "$FASTTEST_OUTPUT/test_log.txt"
else
    echo "No failed tests"
fi
}

case "$stage" in
"")
    ls -la
    ;&
"clone_root")
    clone_root

    # Pass control to the script from cloned sources, unless asked otherwise.
    if ! [ -v FASTTEST_LOCAL_SCRIPT ]
    then
        # 'run' stage is deprecated, used for compatibility with old scripts.
        # Replace with 'clone_submodules' after Nov 1, 2020.
        # cd and CLICKHOUSE_DIR are also a setup for old scripts, remove as well.
        # In modern script we undo it by changing back into workspace dir right
        # away, see below. Remove that as well.
        cd "$FASTTEST_SOURCE"
        CLICKHOUSE_DIR=$(pwd)
        export CLICKHOUSE_DIR
        stage=run "$FASTTEST_SOURCE/docker/test/fasttest/run.sh"
        exit $?
    fi
    ;&
"run")
    # A deprecated stage that is called by old script and equivalent to everything
    # after cloning root, starting with cloning submodules.
    ;&
"clone_submodules")
    # Recover after being called from the old script that changes into source directory.
    # See the compatibility hacks in `clone_root` stage above. Remove at the same time,
    # after Nov 1, 2020.
    cd "$FASTTEST_WORKSPACE"
    clone_submodules | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/submodule_log.txt"
    ;&
"run_cmake")
    run_cmake
    ;&
"build")
    build
    ;&
"configure")
    # The `install_log.txt` is also needed for compatibility with old CI task --
    # if there is no log, it will decide that build failed.
    configure | ts '%Y-%m-%d %H:%M:%S' | tee "$FASTTEST_OUTPUT/install_log.txt"
    ;&
"run_tests")
    run_tests
    ;;
*)
    echo "Unknown test stage '$stage'"
    exit 1
esac

pstree -apgT
jobs
