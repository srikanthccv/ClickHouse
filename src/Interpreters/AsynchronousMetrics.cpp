#include <Interpreters/AsynchronousMetrics.h>
#include <Interpreters/AsynchronousMetricLog.h>
#include <Interpreters/ExpressionJIT.h>
#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/Context.h>
#include <Common/Exception.h>
#include <Common/setThreadName.h>
#include <Common/CurrentMetrics.h>
#include <Common/typeid_cast.h>
#include <Server/ProtocolServerAdapter.h>
#include <Storages/MarkCache.h>
#include <Storages/StorageMergeTree.h>
#include <Storages/StorageReplicatedMergeTree.h>
#include <IO/UncompressedCache.h>
#include <IO/MMappedFileCache.h>
#include <IO/ReadHelpers.h>
#include <Databases/IDatabase.h>
#include <chrono>


#if !defined(ARCADIA_BUILD)
#    include "config_core.h"
#endif

#if USE_JEMALLOC
#    include <jemalloc/jemalloc.h>
#endif


namespace CurrentMetrics
{
    extern const Metric MemoryTracking;
}


namespace DB
{

static void openFileIfExists(const char * filename, std::optional<ReadBufferFromFile> & out)
{
    static constexpr size_t small_buffer_size = 4096;

    /// Ignoring time of check is not time of use cases, as procfs/sysfs files are fairly persistent.

    std::error_code ec;
    if (std::filesystem::is_regular_file(filename, ec))
        out.emplace(filename, small_buffer_size);
}


AsynchronousMetrics::AsynchronousMetrics(
    ContextPtr global_context_,
    int update_period_seconds,
    std::shared_ptr<std::vector<ProtocolServerAdapter>> servers_to_start_before_tables_,
    std::shared_ptr<std::vector<ProtocolServerAdapter>> servers_)
    : WithContext(global_context_)
    , update_period(update_period_seconds)
    , servers_to_start_before_tables(servers_to_start_before_tables_)
    , servers(servers_)
{
#if defined(OS_LINUX)
    openFileIfExists("/proc/meminfo", meminfo);
    openFileIfExists("/proc/mounts", mounts);
    openFileIfExists("/proc/loadavg", loadavg);
    openFileIfExists("/proc/stat", proc_stat);
    openFileIfExists("/proc/cpuinfo", cpuinfo);
    openFileIfExists("/proc/schedstat", schedstat);
    openFileIfExists("/proc/sockstat", sockstat);
    openFileIfExists("/proc/netstat", netstat);
    openFileIfExists("/proc/sys/fs/file-nr", file_nr);
#endif
}

void AsynchronousMetrics::start()
{
    /// Update once right now, to make metrics available just after server start
    /// (without waiting for asynchronous_metrics_update_period_s).
    update();
    thread = std::make_unique<ThreadFromGlobalPool>([this] { run(); });
}

AsynchronousMetrics::~AsynchronousMetrics()
{
    try
    {
        {
            std::lock_guard lock{mutex};
            quit = true;
        }

        wait_cond.notify_one();
        if (thread)
            thread->join();
    }
    catch (...)
    {
        DB::tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}


AsynchronousMetricValues AsynchronousMetrics::getValues() const
{
    std::lock_guard lock{mutex};
    return values;
}

static auto get_next_update_time(std::chrono::seconds update_period)
{
    using namespace std::chrono;

    const auto now = time_point_cast<seconds>(system_clock::now());

    // Use seconds since the start of the hour, because we don't know when
    // the epoch started, maybe on some weird fractional time.
    const auto start_of_hour = time_point_cast<seconds>(time_point_cast<hours>(now));
    const auto seconds_passed = now - start_of_hour;

    // Rotate time forward by half a period -- e.g. if a period is a minute,
    // we'll collect metrics on start of minute + 30 seconds. This is to
    // achieve temporal separation with MetricTransmitter. Don't forget to
    // rotate it back.
    const auto rotation = update_period / 2;

    const auto periods_passed = (seconds_passed + rotation) / update_period;
    const auto seconds_next = (periods_passed + 1) * update_period - rotation;
    const auto time_next = start_of_hour + seconds_next;

    return time_next;
}

void AsynchronousMetrics::run()
{
    setThreadName("AsyncMetrics");

    while (true)
    {
        {
            // Wait first, so that the first metric collection is also on even time.
            std::unique_lock lock{mutex};
            if (wait_cond.wait_until(lock, get_next_update_time(update_period),
                [this] { return quit; }))
            {
                break;
            }
        }

        try
        {
            update();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}


template <typename Max, typename T>
static void calculateMax(Max & max, T x)
{
    if (Max(x) > max)
        max = x;
}

template <typename Max, typename Sum, typename T>
static void calculateMaxAndSum(Max & max, Sum & sum, T x)
{
    sum += x;
    if (Max(x) > max)
        max = x;
}

#if USE_JEMALLOC && JEMALLOC_VERSION_MAJOR >= 4
uint64_t updateJemallocEpoch()
{
    uint64_t value = 0;
    size_t size = sizeof(value);
    mallctl("epoch", &value, &size, &value, size);
    return value;
}

template <typename Value>
static void saveJemallocMetricImpl(AsynchronousMetricValues & values,
    const std::string & jemalloc_full_name,
    const std::string & clickhouse_full_name)
{
    Value value{};
    size_t size = sizeof(value);
    mallctl(jemalloc_full_name.c_str(), &value, &size, nullptr, 0);
    values[clickhouse_full_name] = value;
}

template<typename Value>
static void saveJemallocMetric(AsynchronousMetricValues & values,
    const std::string & metric_name)
{
    saveJemallocMetricImpl<Value>(values,
        fmt::format("stats.{}", metric_name),
        fmt::format("jemalloc.{}", metric_name));
}

template<typename Value>
static void saveAllArenasMetric(AsynchronousMetricValues & values,
    const std::string & metric_name)
{
    saveJemallocMetricImpl<Value>(values,
        fmt::format("stats.arenas.{}.{}", MALLCTL_ARENAS_ALL, metric_name),
        fmt::format("jemalloc.arenas.all.{}", metric_name));
}
#endif

void AsynchronousMetrics::update()
{
    AsynchronousMetricValues new_values;

    {
        if (auto mark_cache = getContext()->getMarkCache())
        {
            new_values["MarkCacheBytes"] = mark_cache->weight();
            new_values["MarkCacheFiles"] = mark_cache->count();
        }
    }

    {
        if (auto uncompressed_cache = getContext()->getUncompressedCache())
        {
            new_values["UncompressedCacheBytes"] = uncompressed_cache->weight();
            new_values["UncompressedCacheCells"] = uncompressed_cache->count();
        }
    }

    {
        if (auto mmap_cache = getContext()->getMMappedFileCache())
        {
            new_values["MMapCacheCells"] = mmap_cache->count();
        }
    }

#if USE_EMBEDDED_COMPILER
    {
        if (auto * compiled_expression_cache = CompiledExpressionCacheFactory::instance().tryGetCache())
        {
            new_values["CompiledExpressionCacheBytes"] = compiled_expression_cache->weight();
            new_values["CompiledExpressionCacheCount"]  = compiled_expression_cache->count();
        }
    }
#endif

    new_values["Uptime"] = getContext()->getUptimeSeconds();

    /// Process process memory usage according to OS
#if defined(OS_LINUX)
    {
        MemoryStatisticsOS::Data data = memory_stat.get();

        new_values["MemoryVirtual"] = data.virt;
        new_values["MemoryResident"] = data.resident;
        new_values["MemoryShared"] = data.shared;
        new_values["MemoryCode"] = data.code;
        new_values["MemoryDataAndStack"] = data.data_and_stack;

        /// We must update the value of total_memory_tracker periodically.
        /// Otherwise it might be calculated incorrectly - it can include a "drift" of memory amount.
        /// See https://github.com/ClickHouse/ClickHouse/issues/10293
        {
            Int64 amount = total_memory_tracker.get();
            Int64 peak = total_memory_tracker.getPeak();
            Int64 new_amount = data.resident;

            LOG_DEBUG(&Poco::Logger::get("AsynchronousMetrics"),
                "MemoryTracking: was {}, peak {}, will set to {} (RSS), difference: {}",
                ReadableSize(amount),
                ReadableSize(peak),
                ReadableSize(new_amount),
                ReadableSize(new_amount - amount)
            );

            total_memory_tracker.set(new_amount);
            CurrentMetrics::set(CurrentMetrics::MemoryTracking, new_amount);
        }
    }
#endif

#if defined(OS_LINUX)
    if (loadavg)
    {
        loadavg->rewind();

        Float64 loadavg1 = 0;
        Float64 loadavg5 = 0;
        Float64 loadavg15 = 0;
        UInt64 threads_runnable = 0;
        UInt64 threads_total = 0;

        readText(loadavg1, *loadavg);
        skipWhitespaceIfAny(*loadavg);
        readText(loadavg5, *loadavg);
        skipWhitespaceIfAny(*loadavg);
        readText(loadavg15, *loadavg);
        skipWhitespaceIfAny(*loadavg);
        readText(threads_runnable, *loadavg);
        assertChar('/', *loadavg);
        readText(threads_total, *loadavg);

        new_values["LoadAverage1"] = loadavg1;
        new_values["LoadAverage5"] = loadavg5;
        new_values["LoadAverage15"] = loadavg15;
        new_values["OSThreadsRunnable"] = threads_runnable;
        new_values["OSThreadsTotal"] = threads_total;
    }

    if (meminfo)
    {
        meminfo->rewind();


    }
#endif

    /// Process CPU usage according to OS
#if defined(OS_LINUX)
    {
        ProcessorStatisticsOS::Data data = proc_stat.get();

        new_values["LoadAvg1"] = data.loadavg.avg1;
        new_values["LoadAvg5"] = data.loadavg.avg5;
        new_values["LoadAvg15"] = data.loadavg.avg15;

        new_values["FreqMin"] = data.freq.min;
        new_values["FreqMax"] = data.freq.max;
        new_values["FreqAvg"] = data.freq.avg;

        new_values["TimeLoadUser"] = data.stload.user_time;
        new_values["TimeLoadNice"] = data.stload.nice_time;
        new_values["TimeLoadSystem"] = data.stload.system_time;
        new_values["TimeLoadIDLE"] = data.stload.idle_time;
        new_values["TimeLoadIowait"] = data.stload.iowait_time;
        new_values["TimeLoadSteal"] = data.stload.steal_time;
        new_values["TimeLoadGuest"] = data.stload.guest_time;
        new_values["TimeLoadGuestNice"] = data.stload.guest_nice_time;

        new_values["Processes"] = data.stload.processes;
        new_values["ProcessesRunning"] = data.stload.procs_running;
        new_values["ProcessesBlocked"] = data.stload.procs_blocked;
    }
#endif

    /// Process disk usage according to OS
#if defined(OS_LINUX)
    {
        DiskStatisticsOS::Data data = disk_stat.get();

        new_values["FilesystemsTotalBytes"] = data.total_bytes;
        new_values["FilesystemsUsedBytes"] = data.used_bytes;
        new_values["FilesystemsTotalINodes"] = data.total_inodes;
        new_values["FilesystemsUsedINodes"] = data.used_inodes;
    }
#endif

    {
        auto databases = DatabaseCatalog::instance().getDatabases();

        size_t max_queue_size = 0;
        size_t max_inserts_in_queue = 0;
        size_t max_merges_in_queue = 0;

        size_t sum_queue_size = 0;
        size_t sum_inserts_in_queue = 0;
        size_t sum_merges_in_queue = 0;

        size_t max_absolute_delay = 0;
        size_t max_relative_delay = 0;

        size_t max_part_count_for_partition = 0;

        size_t number_of_databases = databases.size();
        size_t total_number_of_tables = 0;

        size_t total_number_of_bytes = 0;
        size_t total_number_of_rows = 0;
        size_t total_number_of_parts = 0;

        for (const auto & db : databases)
        {
            /// Check if database can contain MergeTree tables
            if (!db.second->canContainMergeTreeTables())
                continue;
            for (auto iterator = db.second->getTablesIterator(getContext()); iterator->isValid(); iterator->next())
            {
                ++total_number_of_tables;
                const auto & table = iterator->table();
                if (!table)
                    continue;

                StorageMergeTree * table_merge_tree = dynamic_cast<StorageMergeTree *>(table.get());
                StorageReplicatedMergeTree * table_replicated_merge_tree = dynamic_cast<StorageReplicatedMergeTree *>(table.get());

                if (table_replicated_merge_tree)
                {
                    StorageReplicatedMergeTree::Status status;
                    table_replicated_merge_tree->getStatus(status, false);

                    calculateMaxAndSum(max_queue_size, sum_queue_size, status.queue.queue_size);
                    calculateMaxAndSum(max_inserts_in_queue, sum_inserts_in_queue, status.queue.inserts_in_queue);
                    calculateMaxAndSum(max_merges_in_queue, sum_merges_in_queue, status.queue.merges_in_queue);

                    if (!status.is_readonly)
                    {
                        try
                        {
                            time_t absolute_delay = 0;
                            time_t relative_delay = 0;
                            table_replicated_merge_tree->getReplicaDelays(absolute_delay, relative_delay);

                            calculateMax(max_absolute_delay, absolute_delay);
                            calculateMax(max_relative_delay, relative_delay);
                        }
                        catch (...)
                        {
                            tryLogCurrentException(__PRETTY_FUNCTION__,
                                "Cannot get replica delay for table: " + backQuoteIfNeed(db.first) + "." + backQuoteIfNeed(iterator->name()));
                        }
                    }

                    calculateMax(max_part_count_for_partition, table_replicated_merge_tree->getMaxPartsCountForPartition());
                }

                if (table_merge_tree)
                {
                    calculateMax(max_part_count_for_partition, table_merge_tree->getMaxPartsCountForPartition());
                    const auto & settings = getContext()->getSettingsRef();
                    total_number_of_bytes += table_merge_tree->totalBytes(settings).value();
                    total_number_of_rows += table_merge_tree->totalRows(settings).value();
                    total_number_of_parts += table_merge_tree->getPartsCount();
                }
                if (table_replicated_merge_tree)
                {
                    const auto & settings = getContext()->getSettingsRef();
                    total_number_of_bytes += table_replicated_merge_tree->totalBytes(settings).value();
                    total_number_of_rows += table_replicated_merge_tree->totalRows(settings).value();
                    total_number_of_parts += table_replicated_merge_tree->getPartsCount();
                }
            }
        }

        new_values["ReplicasMaxQueueSize"] = max_queue_size;
        new_values["ReplicasMaxInsertsInQueue"] = max_inserts_in_queue;
        new_values["ReplicasMaxMergesInQueue"] = max_merges_in_queue;

        new_values["ReplicasSumQueueSize"] = sum_queue_size;
        new_values["ReplicasSumInsertsInQueue"] = sum_inserts_in_queue;
        new_values["ReplicasSumMergesInQueue"] = sum_merges_in_queue;

        new_values["ReplicasMaxAbsoluteDelay"] = max_absolute_delay;
        new_values["ReplicasMaxRelativeDelay"] = max_relative_delay;

        new_values["MaxPartCountForPartition"] = max_part_count_for_partition;

        new_values["NumberOfDatabases"] = number_of_databases;
        new_values["NumberOfTables"] = total_number_of_tables;

        new_values["TotalBytesOfMergeTreeTables"] = total_number_of_bytes;
        new_values["TotalRowsOfMergeTreeTables"] = total_number_of_rows;
        new_values["TotalPartsOfMergeTreeTables"] = total_number_of_parts;

        auto get_metric_name = [](const String & name) -> const char *
        {
            static std::map<String, const char *> metric_map = {
                {"tcp_port", "TCPThreads"},
                {"tcp_port_secure", "TCPSecureThreads"},
                {"http_port", "HTTPThreads"},
                {"https_port", "HTTPSecureThreads"},
                {"interserver_http_port", "InterserverThreads"},
                {"interserver_https_port", "InterserverSecureThreads"},
                {"mysql_port", "MySQLThreads"},
                {"postgresql_port", "PostgreSQLThreads"},
                {"grpc_port", "GRPCThreads"},
                {"prometheus.port", "PrometheusThreads"}
            };
            auto it = metric_map.find(name);
            if (it == metric_map.end())
                return nullptr;
            else
                return it->second;
        };

        if (servers_to_start_before_tables)
        {
            for (const auto & server : *servers_to_start_before_tables)
            {
                if (const auto * name = get_metric_name(server.getPortName()))
                    new_values[name] = server.currentThreads();
            }
        }

        if (servers)
        {
            for (const auto & server : *servers)
            {
                if (const auto * name = get_metric_name(server.getPortName()))
                    new_values[name] = server.currentThreads();
            }
        }
    }

#if USE_JEMALLOC && JEMALLOC_VERSION_MAJOR >= 4
    // 'epoch' is a special mallctl -- it updates the statistics. Without it, all
    // the following calls will return stale values. It increments and returns
    // the current epoch number, which might be useful to log as a sanity check.
    auto epoch = updateJemallocEpoch();
    new_values["jemalloc.epoch"] = epoch;

    // Collect the statistics themselves.
    saveJemallocMetric<size_t>(new_values, "allocated");
    saveJemallocMetric<size_t>(new_values, "active");
    saveJemallocMetric<size_t>(new_values, "metadata");
    saveJemallocMetric<size_t>(new_values, "metadata_thp");
    saveJemallocMetric<size_t>(new_values, "resident");
    saveJemallocMetric<size_t>(new_values, "mapped");
    saveJemallocMetric<size_t>(new_values, "retained");
    saveJemallocMetric<size_t>(new_values, "background_thread.num_threads");
    saveJemallocMetric<uint64_t>(new_values, "background_thread.num_runs");
    saveJemallocMetric<uint64_t>(new_values, "background_thread.run_intervals");
    saveAllArenasMetric<size_t>(new_values, "pactive");
    saveAllArenasMetric<size_t>(new_values, "pdirty");
    saveAllArenasMetric<size_t>(new_values, "pmuzzy");
    saveAllArenasMetric<size_t>(new_values, "dirty_purged");
    saveAllArenasMetric<size_t>(new_values, "muzzy_purged");
#endif

#if defined(OS_LINUX)
    // Try to add processor frequencies, ignoring errors.
    try
    {
        ReadBufferFromFile buf("/proc/cpuinfo", 32768 /* buf_size */);

        // We need the following lines:
        // processor : 4
        // cpu MHz : 4052.941
        // They contain tabs and are interspersed with other info.
        int core_id = 0;
        while (!buf.eof())
        {
            std::string s;
            // We don't have any backslash escape sequences in /proc/cpuinfo, so
            // this function will read the line until EOL, which is exactly what
            // we need.
            readEscapedStringUntilEOL(s, buf);
            // It doesn't read the EOL itself.
            ++buf.position();

            if (s.rfind("processor", 0) == 0)
            {
                if (auto colon = s.find_first_of(':'))
                {
                    core_id = std::stoi(s.substr(colon + 2));
                }
            }
            else if (s.rfind("cpu MHz", 0) == 0)
            {
                if (auto colon = s.find_first_of(':'))
                {
                    auto mhz = std::stod(s.substr(colon + 2));
                    new_values[fmt::format("CPUFrequencyMHz_{}", core_id)] = mhz;
                }
            }
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
#endif

    /// Add more metrics as you wish.

    // Log the new metrics.
    if (auto log = getContext()->getAsynchronousMetricLog())
    {
        log->addValues(new_values);
    }

    // Finally, update the current metrics.
    std::lock_guard lock(mutex);
    values = new_values;
}

}
