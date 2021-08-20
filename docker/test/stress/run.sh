#!/bin/bash
# shellcheck disable=SC2094
# shellcheck disable=SC2086

set -x

# Thread Fuzzer allows to check more permutations of possible thread scheduling
# and find more potential issues.

export THREAD_FUZZER_CPU_TIME_PERIOD_US=1000
export THREAD_FUZZER_SLEEP_PROBABILITY=0.1
export THREAD_FUZZER_SLEEP_TIME_US=100000

export THREAD_FUZZER_pthread_mutex_lock_BEFORE_MIGRATE_PROBABILITY=1
export THREAD_FUZZER_pthread_mutex_lock_AFTER_MIGRATE_PROBABILITY=1
export THREAD_FUZZER_pthread_mutex_unlock_BEFORE_MIGRATE_PROBABILITY=1
export THREAD_FUZZER_pthread_mutex_unlock_AFTER_MIGRATE_PROBABILITY=1

export THREAD_FUZZER_pthread_mutex_lock_BEFORE_SLEEP_PROBABILITY=0.001
export THREAD_FUZZER_pthread_mutex_lock_AFTER_SLEEP_PROBABILITY=0.001
export THREAD_FUZZER_pthread_mutex_unlock_BEFORE_SLEEP_PROBABILITY=0.001
export THREAD_FUZZER_pthread_mutex_unlock_AFTER_SLEEP_PROBABILITY=0.001
export THREAD_FUZZER_pthread_mutex_lock_BEFORE_SLEEP_TIME_US=10000
LONG
export THREAD_FUZZER_pthread_mutex_lock_AFTER_SLEEP_TIME_US=10000
export THREAD_FUZZER_pthread_mutex_unlock_BEFORE_SLEEP_TIME_US=10000
export THREAD_FUZZER_pthread_mutex_unlock_AFTER_SLEEP_TIME_US=10000


function install_packages()
{
    dpkg -i $1/clickhouse-common-static_*.deb
    dpkg -i $1/clickhouse-common-static-dbg_*.deb
    dpkg -i $1/clickhouse-server_*.deb
    dpkg -i $1/clickhouse-client_*.deb
    dpkg -i $1/clickhouse-test_*.deb
}

function configure()
{
    # install test configs
    /usr/share/clickhouse-test/config/install.sh

    # for clickhouse-server (via service)
    echo "ASAN_OPTIONS='malloc_context_size=10 verbosity=1 allocator_release_to_os_interval_ms=10000'" >> /etc/environment
    # for clickhouse-client
    export ASAN_OPTIONS='malloc_context_size=10 allocator_release_to_os_interval_ms=10000'

    # since we run clickhouse from root
    sudo chown root: /var/lib/clickhouse

    # Set more frequent update period of asynchronous metrics to more frequently update information about real memory usage (less chance of OOM).
    echo "<yandex><asynchronous_metrics_update_period_s>1</asynchronous_metrics_update_period_s></yandex>" \
        > /etc/clickhouse-server/config.d/asynchronous_metrics_update_period_s.xml

    # Set maximum memory usage as half of total memory (less chance of OOM).
    echo "<yandex><max_server_memory_usage_to_ram_ratio>0.5</max_server_memory_usage_to_ram_ratio></yandex>" \
        > /etc/clickhouse-server/config.d/max_server_memory_usage_to_ram_ratio.xml
}

function stop()
{
    clickhouse stop
}

function start()
{
    # Rename existing log file - it will be more convenient to read separate files for separate server runs.
    if [ -f '/var/log/clickhouse-server/clickhouse-server.log' ]
    then
        log_file_counter=1
        while [ -f "/var/log/clickhouse-server/clickhouse-server.log.${log_file_counter}" ]
        do
            log_file_counter=$((log_file_counter + 1))
        done
        mv '/var/log/clickhouse-server/clickhouse-server.log' "/var/log/clickhouse-server/clickhouse-server.log.${log_file_counter}"
    fi

    counter=0
    until clickhouse-client --query "SELECT 1"
    do
        if [ "$counter" -gt 240 ]
        then
            echo "Cannot start clickhouse-server"
            cat /var/log/clickhouse-server/stdout.log
            tail -n1000 /var/log/clickhouse-server/stderr.log
            tail -n100000 /var/log/clickhouse-server/clickhouse-server.log | grep -F -v -e '<Warning> RaftInstance:' -e '<Information> RaftInstance' | tail -n1000
            break
        fi
        # use root to match with current uid
        clickhouse start --user root >/var/log/clickhouse-server/stdout.log 2>>/var/log/clickhouse-server/stderr.log
        sleep 0.5
        counter=$((counter + 1))
    done

    echo "
set follow-fork-mode child
handle all noprint
handle SIGSEGV stop print
handle SIGBUS stop print
handle SIGABRT stop print
continue
thread apply all backtrace
detach
quit
" > script.gdb

    # FIXME Hung check may work incorrectly because of attached gdb
    # 1. False positives are possible
    # 2. We cannot attach another gdb to get stacktraces if some queries hung
    gdb -batch -command script.gdb -p "$(cat /var/run/clickhouse-server/clickhouse-server.pid)" >> /test_output/gdb.log &
}

install_packages package_folder

configure

start

# shellcheck disable=SC2086 # No quotes because I want to split it into words.
/s3downloader --dataset-names $DATASETS
chmod 777 -R /var/lib/clickhouse
clickhouse-client --query "ATTACH DATABASE IF NOT EXISTS datasets ENGINE = Ordinary"
clickhouse-client --query "CREATE DATABASE IF NOT EXISTS test"

stop
start

clickhouse-client --query "SHOW TABLES FROM datasets"
clickhouse-client --query "SHOW TABLES FROM test"
clickhouse-client --query "RENAME TABLE datasets.hits_v1 TO test.hits"
clickhouse-client --query "RENAME TABLE datasets.visits_v1 TO test.visits"
clickhouse-client --query "SHOW TABLES FROM test"

./stress --hung-check --drop-databases --output-folder test_output --skip-func-tests "$SKIP_TESTS_OPTION" \
    && echo -e 'Test script exit code\tOK' >> /test_output/test_results.tsv \
    || echo -e 'Test script failed\tFAIL' >> /test_output/test_results.tsv

stop
start

clickhouse-client --query "SELECT 'Server successfully started', 'OK'" >> /test_output/test_results.tsv \
                       || echo -e 'Server failed to start\tFAIL' >> /test_output/test_results.tsv

[ -f /var/log/clickhouse-server/clickhouse-server.log ] || echo -e "Server log does not exist\tFAIL"
[ -f /var/log/clickhouse-server/stderr.log ] || echo -e "Stderr log does not exist\tFAIL"

# Print Fatal log messages to stdout
zgrep -Fa " <Fatal> " /var/log/clickhouse-server/clickhouse-server.log*

# Grep logs for sanitizer asserts, crashes and other critical errors

# Sanitizer asserts
zgrep -Fa "==================" /var/log/clickhouse-server/stderr.log >> /test_output/tmp
zgrep -Fa "WARNING" /var/log/clickhouse-server/stderr.log >> /test_output/tmp
zgrep -Fav "ASan doesn't fully support makecontext/swapcontext functions" /test_output/tmp > /dev/null \
    && echo -e 'Sanitizer assert (in stderr.log)\tFAIL' >> /test_output/test_results.tsv \
    || echo -e 'No sanitizer asserts\tOK' >> /test_output/test_results.tsv
rm -f /test_output/tmp

# OOM
zgrep -Fa " <Fatal> Application: Child process was terminated by signal 9" /var/log/clickhouse-server/clickhouse-server.log* > /dev/null \
    && echo -e 'OOM killer (or signal 9) in clickhouse-server.log\tFAIL' >> /test_output/test_results.tsv \
    || echo -e 'No OOM messages in clickhouse-server.log\tOK' >> /test_output/test_results.tsv

# Logical errors
zgrep -Fa "Code: 49, e.displayText() = DB::Exception:" /var/log/clickhouse-server/clickhouse-server.log* > /dev/null \
    && echo -e 'Logical error thrown (see clickhouse-server.log)\tFAIL' >> /test_output/test_results.tsv \
    || echo -e 'No logical errors\tOK' >> /test_output/test_results.tsv

# Crash
zgrep -Fa "########################################" /var/log/clickhouse-server/clickhouse-server.log* > /dev/null \
    && echo -e 'Killed by signal (in clickhouse-server.log)\tFAIL' >> /test_output/test_results.tsv \
    || echo -e 'Not crashed\tOK' >> /test_output/test_results.tsv

# It also checks for crash without stacktrace (printed by watchdog)
zgrep -Fa " <Fatal> " /var/log/clickhouse-server/clickhouse-server.log* > /dev/null \
    && echo -e 'Fatal message in clickhouse-server.log\tFAIL' >> /test_output/test_results.tsv \
    || echo -e 'No fatal messages in clickhouse-server.log\tOK' >> /test_output/test_results.tsv

zgrep -Fa "########################################" /test_output/* > /dev/null \
    && echo -e 'Killed by signal (output files)\tFAIL' >> /test_output/test_results.tsv

echo -e "Backward compatibility check\n"

echo "Download previous release server"
clickhouse-client --query="SELECT version()" | ./download_previous_release && echo -e 'Download script exit code\tOK' >> /test_output/backward_compatibility_check_results.tsv \
    || echo -e 'Download script failed\tFAIL' >> /test_output/backward_compatibility_check_results.tsv

if [ "$(ls -A previous_release_package_folder/clickhouse-common-static_*.deb && ls -A previous_release_package_folder/clickhouse-server_*.deb)" ]
then
    echo -e "Successfully downloaded previous release packets\tOK" >> /test_output/backward_compatibility_check_results.tsv
    stop

    # Uninstall current packages
    dpkg --remove clickhouse-test
    dpkg --remove clickhouse-client
    dpkg --remove clickhouse-server
    dpkg --remove clickhouse-common-static-dbg
    dpkg --remove clickhouse-common-static

    # Install previous release packages
    install_packages previous_release_package_folder

    # Start server from previous release
    configure
    start

    clickhouse-client --query="SELECT 'Server version: ', version()"

    # Install new package before running stress test because we should use new clickhouse-client and new clickhouse-test
    install_packages package_folder

    mkdir tmp_stress_output
    
    ./stress --backward-compatibility-check --output-folder tmp_stress_output --global-time-limit=1800 \
        && echo -e 'Test script exit code\tOK' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'Test script failed\tFAIL' >> /test_output/backward_compatibility_check_results.tsv
    rm -rf tmp_stress_output

    clickhouse-client --query="SELECT 'Tables count:', count() FROM system.tables"
    stop
    
    # Start new server
    configure
    start
    clickhouse-client --query "SELECT 'Server successfully started', 'OK'" >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'Server failed to start\tFAIL' >> /test_output/backward_compatibility_check_results.tsv

    clickhouse-client --query="SELECT 'Server version: ', version()"

    # Let the server run for a while before checking log.
    sleep 60
    
    stop

    # Error messages (we ignore Cancelled merging parts, REPLICA_IS_ALREADY_ACTIVE and  errors)
    zgrep -Fav -e "Code: 236. DB::Exception: Cancelled merging parts" -e "REPLICA_IS_ALREADY_ACTIVE" -e "RaftInstance: failed to accept a rpc connection due to error 125" \
        /var/log/clickhouse-server/clickhouse-server.log | zgrep -Fa "<Error>" > /dev/null \
        && echo -e 'Error message in clickhouse-server.log\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'No Error messages in clickhouse-server.log\tOK' >> /test_output/backward_compatibility_check_results.tsv

    # Sanitizer asserts
    zgrep -Fa "==================" /var/log/clickhouse-server/stderr.log >> /test_output/tmp
    zgrep -Fa "WARNING" /var/log/clickhouse-server/stderr.log >> /test_output/tmp
    zgrep -Fav "ASan doesn't fully support makecontext/swapcontext functions" /test_output/tmp > /dev/null \
        && echo -e 'Sanitizer assert (in stderr.log)\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'No sanitizer asserts\tOK' >> /test_output/backward_compatibility_check_results.tsv
    rm -f /test_output/tmp

    # OOM
    zgrep -Fa " <Fatal> Application: Child process was terminated by signal 9" /var/log/clickhouse-server/clickhouse-server.log > /dev/null \
        && echo -e 'OOM killer (or signal 9) in clickhouse-server.log\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'No OOM messages in clickhouse-server.log\tOK' >> /test_output/backward_compatibility_check_results.tsv

    # Logical errors
    zgrep -Fa "Code: 49, e.displayText() = DB::Exception:" /var/log/clickhouse-server/clickhouse-server.log > /dev/null \
        && echo -e 'Logical error thrown (see clickhouse-server.log)\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'No logical errors\tOK' >> /test_output/backward_compatibility_check_results.tsv

    # Crash
    zgrep -Fa "########################################" /var/log/clickhouse-server/clickhouse-server.log > /dev/null \
        && echo -e 'Killed by signal (in clickhouse-server.log)\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'Not crashed\tOK' >> /test_output/backward_compatibility_check_results.tsv

    # It also checks for crash without stacktrace (printed by watchdog)
    zgrep -Fa " <Fatal> " /var/log/clickhouse-server/clickhouse-server.log > /dev/null \
        && echo -e 'Fatal message in clickhouse-server.log\tFAIL' >> /test_output/backward_compatibility_check_results.tsv \
        || echo -e 'No fatal messages in clickhouse-server.log\tOK' >> /test_output/backward_compatibility_check_results.tsv

else
    echo -e "Failed to download previous release packets\tFAIL" >> /test_output/backward_compatibility_check_results.tsv
fi

zgrep -Fa "FAIL" /test_output/backward_compatibility_check_results.tsv > /dev/null \
        && echo -e 'Backward compatibility check\tFAIL' >> /test_output/test_results.tsv \
        || echo -e 'Backward compatibility check\tOK' >> /test_output/test_results.tsv


# Put logs into /test_output/
for log_file in /var/log/clickhouse-server/clickhouse-server.log*
do
    pigz < "${log_file}" > /test_output/"$(basename ${log_file})".gz
done

tar -chf /test_output/coordination.tar /var/lib/clickhouse/coordination ||:
mv /var/log/clickhouse-server/stderr.log /test_output/
tar -chf /test_output/query_log_dump.tar /var/lib/clickhouse/data/system/query_log ||:
tar -chf /test_output/trace_log_dump.tar /var/lib/clickhouse/data/system/trace_log ||:

# Write check result into check_status.tsv
clickhouse-local --structure "test String, res String" -q "SELECT 'failure', test FROM table WHERE res != 'OK' order by (lower(test) like '%hung%') LIMIT 1" < /test_output/test_results.tsv > /test_output/check_status.tsv
[ -s /test_output/check_status.tsv ] || echo -e "success\tNo errors found" > /test_output/check_status.tsv
