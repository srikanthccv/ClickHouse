#include <Interpreters/Aggregator.h>
#include <Interpreters/AsynchronousMetrics.h>
#include <Interpreters/AsynchronousMetricLog.h>
#include <Interpreters/JIT/CompiledExpressionCache.h>
#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/Context.h>
#include <Coordination/Keeper4LWInfo.h>
#include <Coordination/KeeperDispatcher.h>
#include <Common/Exception.h>
#include <Common/setThreadName.h>
#include <Common/CurrentMetrics.h>
#include <Common/typeid_cast.h>
#include <Common/filesystemHelpers.h>
#include <Interpreters/Cache/FileCacheFactory.h>
#include <Common/getCurrentProcessFDCount.h>
#include <Common/getMaxFileDescriptorCount.h>
#include <Interpreters/Cache/FileCache.h>
#include <Storages/MarkCache.h>
#include <Storages/StorageMergeTree.h>
#include <Storages/StorageReplicatedMergeTree.h>
#include <Storages/MergeTree/MergeTreeMetadataCache.h>
#include <IO/UncompressedCache.h>
#include <IO/MMappedFileCache.h>
#include <IO/ReadHelpers.h>
#include <Databases/IDatabase.h>
#include <base/errnoToString.h>
#include <chrono>

#include "config.h"

#if USE_JEMALLOC
#    include <jemalloc/jemalloc.h>
#endif


namespace DB
{

namespace ErrorCodes
{
    extern const int CORRUPTED_DATA;
    extern const int CANNOT_SYSCONF;
}


#if defined(OS_LINUX)

static constexpr size_t small_buffer_size = 4096;

static void openFileIfExists(const char * filename, std::optional<ReadBufferFromFilePRead> & out)
{
    /// Ignoring time of check is not time of use cases, as procfs/sysfs files are fairly persistent.

    std::error_code ec;
    if (std::filesystem::is_regular_file(filename, ec))
        out.emplace(filename, small_buffer_size);
}

static std::unique_ptr<ReadBufferFromFilePRead> openFileIfExists(const std::string & filename)
{
    std::error_code ec;
    if (std::filesystem::is_regular_file(filename, ec))
        return std::make_unique<ReadBufferFromFilePRead>(filename, small_buffer_size);
    return {};
}

#endif


AsynchronousMetrics::AsynchronousMetrics(
    ContextPtr global_context_,
    int update_period_seconds,
    int heavy_metrics_update_period_seconds,
    const ProtocolServerMetricsFunc & protocol_server_metrics_func_)
    : WithContext(global_context_)
    , update_period(update_period_seconds)
    , heavy_metric_update_period(heavy_metrics_update_period_seconds)
    , protocol_server_metrics_func(protocol_server_metrics_func_)
    , log(&Poco::Logger::get("AsynchronousMetrics"))
{
#if defined(OS_LINUX)
    openFileIfExists("/proc/meminfo", meminfo);
    openFileIfExists("/proc/loadavg", loadavg);
    openFileIfExists("/proc/stat", proc_stat);
    openFileIfExists("/proc/cpuinfo", cpuinfo);
    openFileIfExists("/proc/sys/fs/file-nr", file_nr);
    openFileIfExists("/proc/uptime", uptime);
    openFileIfExists("/proc/net/dev", net_dev);

    openSensors();
    openBlockDevices();
    openEDAC();
    openSensorsChips();
#endif
}

#if defined(OS_LINUX)
void AsynchronousMetrics::openSensors()
{
    LOG_TRACE(log, "Scanning /sys/class/thermal");

    thermal.clear();

    for (size_t thermal_device_index = 0;; ++thermal_device_index)
    {
        std::unique_ptr<ReadBufferFromFilePRead> file = openFileIfExists(fmt::format("/sys/class/thermal/thermal_zone{}/temp", thermal_device_index));
        if (!file)
        {
            /// Sometimes indices are from zero sometimes from one.
            if (thermal_device_index == 0)
                continue;
            else
                break;
        }

        file->rewind();
        Int64 temperature = 0;
        try
        {
            readText(temperature, *file);
        }
        catch (const ErrnoException & e)
        {
            LOG_WARNING(
                &Poco::Logger::get("AsynchronousMetrics"),
                "Thermal monitor '{}' exists but could not be read: {}.",
                thermal_device_index,
                errnoToString(e.getErrno()));
            continue;
        }

        thermal.emplace_back(std::move(file));
    }
}

void AsynchronousMetrics::openBlockDevices()
{
    LOG_TRACE(log, "Scanning /sys/block");

    if (!std::filesystem::exists("/sys/block"))
        return;

    block_devices_rescan_delay.restart();

    block_devs.clear();

    for (const auto & device_dir : std::filesystem::directory_iterator("/sys/block"))
    {
        String device_name = device_dir.path().filename();

        /// We are not interested in loopback devices.
        if (device_name.starts_with("loop"))
            continue;

        std::unique_ptr<ReadBufferFromFilePRead> file = openFileIfExists(device_dir.path() / "stat");
        if (!file)
            continue;

        block_devs[device_name] = std::move(file);
    }
}

void AsynchronousMetrics::openEDAC()
{
    LOG_TRACE(log, "Scanning /sys/devices/system/edac");

    edac.clear();

    for (size_t edac_index = 0;; ++edac_index)
    {
        String edac_correctable_file = fmt::format("/sys/devices/system/edac/mc/mc{}/ce_count", edac_index);
        String edac_uncorrectable_file = fmt::format("/sys/devices/system/edac/mc/mc{}/ue_count", edac_index);

        bool edac_correctable_file_exists = std::filesystem::exists(edac_correctable_file);
        bool edac_uncorrectable_file_exists = std::filesystem::exists(edac_uncorrectable_file);

        if (!edac_correctable_file_exists && !edac_uncorrectable_file_exists)
        {
            if (edac_index == 0)
                continue;
            else
                break;
        }

        edac.emplace_back();

        if (edac_correctable_file_exists)
            edac.back().first = openFileIfExists(edac_correctable_file);
        if (edac_uncorrectable_file_exists)
            edac.back().second = openFileIfExists(edac_uncorrectable_file);
    }
}

void AsynchronousMetrics::openSensorsChips()
{
    LOG_TRACE(log, "Scanning /sys/class/hwmon");

    hwmon_devices.clear();

    for (size_t hwmon_index = 0;; ++hwmon_index)
    {
        String hwmon_name_file = fmt::format("/sys/class/hwmon/hwmon{}/name", hwmon_index);
        if (!std::filesystem::exists(hwmon_name_file))
        {
            if (hwmon_index == 0)
                continue;
            else
                break;
        }

        String hwmon_name;
        ReadBufferFromFilePRead hwmon_name_in(hwmon_name_file, small_buffer_size);
        readText(hwmon_name, hwmon_name_in);
        std::replace(hwmon_name.begin(), hwmon_name.end(), ' ', '_');

        for (size_t sensor_index = 0;; ++sensor_index)
        {
            String sensor_name_file = fmt::format("/sys/class/hwmon/hwmon{}/temp{}_label", hwmon_index, sensor_index);
            String sensor_value_file = fmt::format("/sys/class/hwmon/hwmon{}/temp{}_input", hwmon_index, sensor_index);

            bool sensor_name_file_exists = std::filesystem::exists(sensor_name_file);
            bool sensor_value_file_exists = std::filesystem::exists(sensor_value_file);

            /// Sometimes there are labels but there is no files with data or vice versa.
            if (!sensor_name_file_exists && !sensor_value_file_exists)
            {
                if (sensor_index == 0)
                    continue;
                else
                    break;
            }

            std::unique_ptr<ReadBufferFromFilePRead> file = openFileIfExists(sensor_value_file);
            if (!file)
                continue;

            String sensor_name;
            if (sensor_name_file_exists)
            {
                ReadBufferFromFilePRead sensor_name_in(sensor_name_file, small_buffer_size);
                readText(sensor_name, sensor_name_in);
                std::replace(sensor_name.begin(), sensor_name.end(), ' ', '_');
            }

            file->rewind();
            Int64 temperature = 0;
            try
            {
                readText(temperature, *file);
            }
            catch (const ErrnoException & e)
            {
                LOG_WARNING(
                    &Poco::Logger::get("AsynchronousMetrics"),
                    "Hardware monitor '{}', sensor '{}' exists but could not be read: {}.",
                    hwmon_name,
                    sensor_name,
                    errnoToString(e.getErrno()));
                continue;
            }

            hwmon_devices[hwmon_name][sensor_name] = std::move(file);
        }
    }
}
#endif

void AsynchronousMetrics::start()
{
    /// Update once right now, to make metrics available just after server start
    /// (without waiting for asynchronous_metrics_update_period_s).
    update(std::chrono::system_clock::now());
    thread = std::make_unique<ThreadFromGlobalPool>([this] { run(); });
}

void AsynchronousMetrics::stop()
{
    try
    {
        {
            std::lock_guard lock{mutex};
            quit = true;
        }

        wait_cond.notify_one();
        if (thread)
        {
            thread->join();
            thread.reset();
        }
    }
    catch (...)
    {
        DB::tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

AsynchronousMetrics::~AsynchronousMetrics()
{
    stop();
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
        auto next_update_time = get_next_update_time(update_period);

        {
            // Wait first, so that the first metric collection is also on even time.
            std::unique_lock lock{mutex};
            if (wait_cond.wait_until(lock, next_update_time,
                [this] { return quit; }))
            {
                break;
            }
        }

        try
        {
            update(next_update_time);
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

#if USE_JEMALLOC
uint64_t updateJemallocEpoch()
{
    uint64_t value = 0;
    size_t size = sizeof(value);
    mallctl("epoch", &value, &size, &value, size);
    return value;
}

template <typename Value>
static Value saveJemallocMetricImpl(
    AsynchronousMetricValues & values,
    const std::string & jemalloc_full_name,
    const std::string & clickhouse_full_name)
{
    Value value{};
    size_t size = sizeof(value);
    mallctl(jemalloc_full_name.c_str(), &value, &size, nullptr, 0);
    values[clickhouse_full_name] = AsynchronousMetricValue(value, "An internal metric of the low-level memory allocator (jemalloc). See https://jemalloc.net/jemalloc.3.html");
    return value;
}

template<typename Value>
static Value saveJemallocMetric(AsynchronousMetricValues & values,
    const std::string & metric_name)
{
    return saveJemallocMetricImpl<Value>(values,
        fmt::format("stats.{}", metric_name),
        fmt::format("jemalloc.{}", metric_name));
}

template<typename Value>
static Value saveAllArenasMetric(AsynchronousMetricValues & values,
    const std::string & metric_name)
{
    return saveJemallocMetricImpl<Value>(values,
        fmt::format("stats.arenas.{}.{}", MALLCTL_ARENAS_ALL, metric_name),
        fmt::format("jemalloc.arenas.all.{}", metric_name));
}
#endif


#if defined(OS_LINUX)

void AsynchronousMetrics::ProcStatValuesCPU::read(ReadBuffer & in)
{
    readText(user, in);
    skipWhitespaceIfAny(in, true);
    readText(nice, in);
    skipWhitespaceIfAny(in, true);
    readText(system, in);
    skipWhitespaceIfAny(in, true);
    readText(idle, in);
    skipWhitespaceIfAny(in, true);
    readText(iowait, in);
    skipWhitespaceIfAny(in, true);
    readText(irq, in);
    skipWhitespaceIfAny(in, true);
    readText(softirq, in);

    /// Just in case for old Linux kernels, we check if these values present.

    if (!checkChar('\n', in))
    {
        skipWhitespaceIfAny(in, true);
        readText(steal, in);
    }

    if (!checkChar('\n', in))
    {
        skipWhitespaceIfAny(in, true);
        readText(guest, in);
    }

    if (!checkChar('\n', in))
    {
        skipWhitespaceIfAny(in, true);
        readText(guest_nice, in);
    }

    skipToNextLineOrEOF(in);
}

AsynchronousMetrics::ProcStatValuesCPU
AsynchronousMetrics::ProcStatValuesCPU::operator-(const AsynchronousMetrics::ProcStatValuesCPU & other) const
{
    ProcStatValuesCPU res{};
    res.user = user - other.user;
    res.nice = nice - other.nice;
    res.system = system - other.system;
    res.idle = idle - other.idle;
    res.iowait = iowait - other.iowait;
    res.irq = irq - other.irq;
    res.softirq = softirq - other.softirq;
    res.steal = steal - other.steal;
    res.guest = guest - other.guest;
    res.guest_nice = guest_nice - other.guest_nice;
    return res;
}

AsynchronousMetrics::ProcStatValuesOther
AsynchronousMetrics::ProcStatValuesOther::operator-(const AsynchronousMetrics::ProcStatValuesOther & other) const
{
    ProcStatValuesOther res{};
    res.interrupts = interrupts - other.interrupts;
    res.context_switches = context_switches - other.context_switches;
    res.processes_created = processes_created - other.processes_created;
    return res;
}

void AsynchronousMetrics::BlockDeviceStatValues::read(ReadBuffer & in)
{
    skipWhitespaceIfAny(in, true);
    readText(read_ios, in);
    skipWhitespaceIfAny(in, true);
    readText(read_merges, in);
    skipWhitespaceIfAny(in, true);
    readText(read_sectors, in);
    skipWhitespaceIfAny(in, true);
    readText(read_ticks, in);
    skipWhitespaceIfAny(in, true);
    readText(write_ios, in);
    skipWhitespaceIfAny(in, true);
    readText(write_merges, in);
    skipWhitespaceIfAny(in, true);
    readText(write_sectors, in);
    skipWhitespaceIfAny(in, true);
    readText(write_ticks, in);
    skipWhitespaceIfAny(in, true);
    readText(in_flight_ios, in);
    skipWhitespaceIfAny(in, true);
    readText(io_ticks, in);
    skipWhitespaceIfAny(in, true);
    readText(time_in_queue, in);
    skipWhitespaceIfAny(in, true);
    readText(discard_ops, in);
    skipWhitespaceIfAny(in, true);
    readText(discard_merges, in);
    skipWhitespaceIfAny(in, true);
    readText(discard_sectors, in);
    skipWhitespaceIfAny(in, true);
    readText(discard_ticks, in);
}

AsynchronousMetrics::BlockDeviceStatValues
AsynchronousMetrics::BlockDeviceStatValues::operator-(const AsynchronousMetrics::BlockDeviceStatValues & other) const
{
    BlockDeviceStatValues res{};
    res.read_ios = read_ios - other.read_ios;
    res.read_merges = read_merges - other.read_merges;
    res.read_sectors = read_sectors - other.read_sectors;
    res.read_ticks = read_ticks - other.read_ticks;
    res.write_ios = write_ios - other.write_ios;
    res.write_merges = write_merges - other.write_merges;
    res.write_sectors = write_sectors - other.write_sectors;
    res.write_ticks = write_ticks - other.write_ticks;
    res.in_flight_ios = in_flight_ios; /// This is current value, not total.
    res.io_ticks = io_ticks - other.io_ticks;
    res.time_in_queue = time_in_queue - other.time_in_queue;
    res.discard_ops = discard_ops - other.discard_ops;
    res.discard_merges = discard_merges - other.discard_merges;
    res.discard_sectors = discard_sectors - other.discard_sectors;
    res.discard_ticks = discard_ticks - other.discard_ticks;
    return res;
}

AsynchronousMetrics::NetworkInterfaceStatValues
AsynchronousMetrics::NetworkInterfaceStatValues::operator-(const AsynchronousMetrics::NetworkInterfaceStatValues & other) const
{
    NetworkInterfaceStatValues res{};
    res.recv_bytes = recv_bytes - other.recv_bytes;
    res.recv_packets = recv_packets - other.recv_packets;
    res.recv_errors = recv_errors - other.recv_errors;
    res.recv_drop = recv_drop - other.recv_drop;
    res.send_bytes = send_bytes - other.send_bytes;
    res.send_packets = send_packets - other.send_packets;
    res.send_errors = send_errors - other.send_errors;
    res.send_drop = send_drop - other.send_drop;
    return res;
}

#endif


void AsynchronousMetrics::update(TimePoint update_time)
{
    Stopwatch watch;

    AsynchronousMetricValues new_values;

    auto current_time = std::chrono::system_clock::now();
    auto time_after_previous_update [[maybe_unused]] = current_time - previous_update_time;
    previous_update_time = update_time;

    /// This is also a good indicator of system responsiveness.
    new_values["Jitter"] = { std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - update_time).count() / 1e9,
        "The difference in time the thread for calculation of the asynchronous metrics was scheduled to wake up and the time it was in fact, woken up."
        " A proxy-indicator of overall system latency and responsiveness." };

    if (auto mark_cache = getContext()->getMarkCache())
    {
        new_values["MarkCacheBytes"] = { mark_cache->weight(), "Total size of mark cache in bytes" };
        new_values["MarkCacheFiles"] = { mark_cache->count(), "Total number of mark files cached in the mark cache" };
    }

    if (auto uncompressed_cache = getContext()->getUncompressedCache())
    {
        new_values["UncompressedCacheBytes"] = { uncompressed_cache->weight(),
            "Total size of uncompressed cache in bytes. Uncompressed cache does not usually improve the performance and should be mostly avoided." };
        new_values["UncompressedCacheCells"] = { uncompressed_cache->count(),
            "Total number of entries in the uncompressed cache. Each entry represents a decompressed block of data. Uncompressed cache does not usually improve performance and should be mostly avoided." };
    }

    if (auto index_mark_cache = getContext()->getIndexMarkCache())
    {
        new_values["IndexMarkCacheBytes"] = { index_mark_cache->weight(), "Total size of mark cache for secondary indices in bytes." };
        new_values["IndexMarkCacheFiles"] = { index_mark_cache->count(), "Total number of mark files cached in the mark cache for secondary indices." };
    }

    if (auto index_uncompressed_cache = getContext()->getIndexUncompressedCache())
    {
        new_values["IndexUncompressedCacheBytes"] = { index_uncompressed_cache->weight(),
            "Total size of uncompressed cache in bytes for secondary indices. Uncompressed cache does not usually improve the performance and should be mostly avoided." };
        new_values["IndexUncompressedCacheCells"] = { index_uncompressed_cache->count(),
            "Total number of entries in the uncompressed cache for secondary indices. Each entry represents a decompressed block of data. Uncompressed cache does not usually improve performance and should be mostly avoided." };
    }

    if (auto mmap_cache = getContext()->getMMappedFileCache())
    {
        new_values["MMapCacheCells"] = { mmap_cache->count(),
            "The number of files opened with `mmap` (mapped in memory)."
            " This is used for queries with the setting `local_filesystem_read_method` set to  `mmap`."
            " The files opened with `mmap` are kept in the cache to avoid costly TLB flushes."};
    }

    {
        auto caches = FileCacheFactory::instance().getAll();
        size_t total_bytes = 0;
        size_t total_files = 0;

        for (const auto & [_, cache_data] : caches)
        {
            total_bytes += cache_data->cache->getUsedCacheSize();
            total_files += cache_data->cache->getFileSegmentsNum();
        }

        new_values["FilesystemCacheBytes"] = { total_bytes,
            "Total bytes in the `cache` virtual filesystem. This cache is hold on disk." };
        new_values["FilesystemCacheFiles"] = { total_files,
            "Total number of cached file segments in the `cache` virtual filesystem. This cache is hold on disk." };
    }

#if USE_ROCKSDB
    if (auto metadata_cache = getContext()->tryGetMergeTreeMetadataCache())
    {
        new_values["MergeTreeMetadataCacheSize"] = { metadata_cache->getEstimateNumKeys(),
            "The size of the metadata cache for tables. This cache is experimental and not used in production." };
    }
#endif

#if USE_EMBEDDED_COMPILER
    if (auto * compiled_expression_cache = CompiledExpressionCacheFactory::instance().tryGetCache())
    {
        new_values["CompiledExpressionCacheBytes"] = { compiled_expression_cache->weight(),
            "Total bytes used for the cache of JIT-compiled code." };
        new_values["CompiledExpressionCacheCount"] = { compiled_expression_cache->count(),
            "Total entries in the cache of JIT-compiled code." };
    }
#endif

    new_values["Uptime"] = { getContext()->getUptimeSeconds(),
        "The server uptime in seconds. It includes the time spent for server initialization before accepting connections." };

    if (const auto stats = getHashTablesCacheStatistics())
    {
        new_values["HashTableStatsCacheEntries"] = { stats->entries,
            "The number of entries in the cache of hash table sizes."
            " The cache for hash table sizes is used for predictive optimization of GROUP BY." };
        new_values["HashTableStatsCacheHits"] = { stats->hits,
            "The number of times the prediction of a hash table size was correct." };
        new_values["HashTableStatsCacheMisses"] = { stats->misses,
            "The number of times the prediction of a hash table size was incorrect." };
    }

#if defined(OS_LINUX) || defined(OS_FREEBSD)
    MemoryStatisticsOS::Data memory_statistics_data = memory_stat.get();
#endif

#if USE_JEMALLOC
    // 'epoch' is a special mallctl -- it updates the statistics. Without it, all
    // the following calls will return stale values. It increments and returns
    // the current epoch number, which might be useful to log as a sanity check.
    auto epoch = updateJemallocEpoch();
    new_values["jemalloc.epoch"] = { epoch, "An internal incremental update number of the statistics of jemalloc (Jason Evans' memory allocator), used in all other `jemalloc` metrics." };

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
    [[maybe_unused]] size_t je_malloc_pdirty = saveAllArenasMetric<size_t>(new_values, "pdirty");
    [[maybe_unused]] size_t je_malloc_pmuzzy = saveAllArenasMetric<size_t>(new_values, "pmuzzy");
    saveAllArenasMetric<size_t>(new_values, "dirty_purged");
    saveAllArenasMetric<size_t>(new_values, "muzzy_purged");
#endif

    /// Process process memory usage according to OS
#if defined(OS_LINUX) || defined(OS_FREEBSD)
    {
        MemoryStatisticsOS::Data & data = memory_statistics_data;

        new_values["MemoryVirtual"] = { data.virt,
            "The size of the virtual address space allocated by the server process, in bytes."
            " The size of the virtual address space is usually much greater than the physical memory consumption, and should not be used as an estimate for the memory consumption."
            " The large values of this metric are totally normal, and makes only technical sense."};
        new_values["MemoryResident"] = { data.resident,
            "The amount of physical memory used by the server process, in bytes." };
#if !defined(OS_FREEBSD)
        new_values["MemoryShared"] = { data.shared,
            "The amount of memory used by the server process, that is also shared by another processes, in bytes."
            " ClickHouse does not use shared memory, but some memory can be labeled by OS as shared for its own reasons."
            " This metric does not make a lot of sense to watch, and it exists only for completeness reasons."};
#endif
        new_values["MemoryCode"] = { data.code,
            "The amount of virtual memory mapped for the pages of machine code of the server process, in bytes." };
        new_values["MemoryDataAndStack"] = { data.data_and_stack,
            "The amount of virtual memory mapped for the use of stack and for the allocated memory, in bytes."
            " It is unspecified whether it includes the per-thread stacks and most of the allocated memory, that is allocated with the 'mmap' system call."
            " This metric exists only for completeness reasons. I recommend to use the `MemoryResident` metric for monitoring."};

        /// We must update the value of total_memory_tracker periodically.
        /// Otherwise it might be calculated incorrectly - it can include a "drift" of memory amount.
        /// See https://github.com/ClickHouse/ClickHouse/issues/10293
        {
            Int64 amount = total_memory_tracker.get();
            Int64 peak = total_memory_tracker.getPeak();
            Int64 rss = data.resident;
            Int64 free_memory_in_allocator_arenas = 0;

#if USE_JEMALLOC
            /// According to jemalloc man, pdirty is:
            ///
            ///     Number of pages within unused extents that are potentially
            ///     dirty, and for which madvise() or similar has not been called.
            ///
            /// So they will be subtracted from RSS to make accounting more
            /// accurate, since those pages are not really RSS but a memory
            /// that can be used at anytime via jemalloc.
            free_memory_in_allocator_arenas = je_malloc_pdirty * getPageSize();
#endif

            Int64 difference = rss - amount;

            /// Log only if difference is high. This is for convenience. The threshold is arbitrary.
            if (difference >= 1048576 || difference <= -1048576)
                LOG_TRACE(log,
                    "MemoryTracking: was {}, peak {}, free memory in arenas {}, will set to {} (RSS), difference: {}",
                    ReadableSize(amount),
                    ReadableSize(peak),
                    ReadableSize(free_memory_in_allocator_arenas),
                    ReadableSize(rss),
                    ReadableSize(difference));

            total_memory_tracker.setRSS(rss, free_memory_in_allocator_arenas);
        }
    }
#endif

#if defined(OS_LINUX)
    if (loadavg)
    {
        try
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

#define LOAD_AVERAGE_DOCUMENTATION \
    " The load represents the number of threads across all the processes (the scheduling entities of the OS kernel)," \
    " that are currently running by CPU or waiting for IO, or ready to run but not being scheduled at this point of time." \
    " This number includes all the processes, not only clickhouse-server. The number can be greater than the number of CPU cores," \
    " if the system is overloaded, and many processes are ready to run but waiting for CPU or IO."

            new_values["LoadAverage1"] = { loadavg1,
                "The whole system load, averaged with exponential smoothing over 1 minute." LOAD_AVERAGE_DOCUMENTATION };
            new_values["LoadAverage5"] = { loadavg5,
                "The whole system load, averaged with exponential smoothing over 5 minutes." LOAD_AVERAGE_DOCUMENTATION };
            new_values["LoadAverage15"] = { loadavg15,
                "The whole system load, averaged with exponential smoothing over 15 minutes." LOAD_AVERAGE_DOCUMENTATION };
            new_values["OSThreadsRunnable"] = { threads_runnable,
                "The total number of 'runnable' threads, as the OS kernel scheduler seeing it." };
            new_values["OSThreadsTotal"] = { threads_total,
                "The total number of threads, as the OS kernel scheduler seeing it." };
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    if (uptime)
    {
        try
        {
            uptime->rewind();

            Float64 uptime_seconds = 0;
            readText(uptime_seconds, *uptime);

            new_values["OSUptime"] = { uptime_seconds, "The uptime of the host server (the machine where ClickHouse is running), in seconds." };
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    if (proc_stat)
    {
        try
        {
            proc_stat->rewind();

            int64_t hz = sysconf(_SC_CLK_TCK);
            if (-1 == hz)
                throwFromErrno("Cannot call 'sysconf' to obtain system HZ", ErrorCodes::CANNOT_SYSCONF);

            double multiplier = 1.0 / hz / (std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_previous_update).count() / 1e9);
            size_t num_cpus = 0;

            ProcStatValuesOther current_other_values{};
            ProcStatValuesCPU delta_values_all_cpus{};

            while (!proc_stat->eof())
            {
                String name;
                readStringUntilWhitespace(name, *proc_stat);
                skipWhitespaceIfAny(*proc_stat);

                if (name.starts_with("cpu"))
                {
                    String cpu_num_str = name.substr(strlen("cpu"));
                    UInt64 cpu_num = 0;
                    if (!cpu_num_str.empty())
                    {
                        cpu_num = parse<UInt64>(cpu_num_str);

                        if (cpu_num > 1000000) /// Safety check, arbitrary large number, suitable for supercomputing applications.
                            throw Exception(ErrorCodes::CORRUPTED_DATA, "Too many CPUs (at least {}) in '/proc/stat' file", cpu_num);

                        if (proc_stat_values_per_cpu.size() <= cpu_num)
                            proc_stat_values_per_cpu.resize(cpu_num + 1);
                    }

                    ProcStatValuesCPU current_values{};
                    current_values.read(*proc_stat);

                    ProcStatValuesCPU & prev_values = !cpu_num_str.empty() ? proc_stat_values_per_cpu[cpu_num] : proc_stat_values_all_cpus;

                    if (!first_run)
                    {
                        ProcStatValuesCPU delta_values = current_values - prev_values;

                        String cpu_suffix;
                        if (!cpu_num_str.empty())
                        {
                            cpu_suffix = "CPU" + cpu_num_str;
                            ++num_cpus;
                        }
                        else
                            delta_values_all_cpus = delta_values;

                        new_values["OSUserTime" + cpu_suffix] = { delta_values.user * multiplier,
                            "The ratio of time the CPU core was running userspace code. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " This includes also the time when the CPU was under-utilized due to the reasons internal to the CPU (memory loads, pipeline stalls, branch mispredictions, running another SMT core)."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSNiceTime" + cpu_suffix] = { delta_values.nice * multiplier,
                            "The ratio of time the CPU core was running userspace code with higher priority. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSSystemTime" + cpu_suffix] = { delta_values.system * multiplier,
                            "The ratio of time the CPU core was running OS kernel (system) code. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSIdleTime" + cpu_suffix] = { delta_values.idle * multiplier,
                            "The ratio of time the CPU core was idle (not even ready to run a process waiting for IO) from the OS kernel standpoint. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " This does not include the time when the CPU was under-utilized due to the reasons internal to the CPU (memory loads, pipeline stalls, branch mispredictions, running another SMT core)."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSIOWaitTime" + cpu_suffix] = { delta_values.iowait * multiplier,
                            "The ratio of time the CPU core was not running the code but when the OS kernel did not run any other process on this CPU as the processes were waiting for IO. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSIrqTime" + cpu_suffix] = { delta_values.irq * multiplier,
                            "The ratio of time spent for running hardware interrupt requests on the CPU. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " A high number of this metric may indicate hardware misconfiguration or a very high network load."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSSoftIrqTime" + cpu_suffix] = { delta_values.softirq * multiplier,
                            "The ratio of time spent for running software interrupt requests on the CPU. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " A high number of this metric may indicate inefficient software running on the system."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSStealTime" + cpu_suffix] = { delta_values.steal * multiplier,
                            "The ratio of time spent in other operating systems by the CPU when running in a virtualized environment. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " Not every virtualized environments present this metric, and most of them don't."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSGuestTime" + cpu_suffix] = { delta_values.guest * multiplier,
                            "The ratio of time spent running a virtual CPU for guest operating systems under the control of the Linux kernel (See `man procfs`). This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " This metric is irrelevant for ClickHouse, but still exists for completeness."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                        new_values["OSGuestNiceTime" + cpu_suffix] = { delta_values.guest_nice * multiplier,
                            "The ratio of time spent running a virtual CPU for guest operating systems under the control of the Linux kernel, when a guest was set to a higher priority (See `man procfs`). This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server."
                            " This metric is irrelevant for ClickHouse, but still exists for completeness."
                            " The value for a single CPU core will be in the interval [0..1]. The value for all CPU cores is calculated as a sum across them [0..num cores]."};
                    }

                    prev_values = current_values;
                }
                else if (name == "intr")
                {
                    readText(current_other_values.interrupts, *proc_stat);
                    skipToNextLineOrEOF(*proc_stat);
                }
                else if (name == "ctxt")
                {
                    readText(current_other_values.context_switches, *proc_stat);
                    skipToNextLineOrEOF(*proc_stat);
                }
                else if (name == "processes")
                {
                    readText(current_other_values.processes_created, *proc_stat);
                    skipToNextLineOrEOF(*proc_stat);
                }
                else if (name == "procs_running")
                {
                    UInt64 processes_running = 0;
                    readText(processes_running, *proc_stat);
                    skipToNextLineOrEOF(*proc_stat);
                    new_values["OSProcessesRunning"] = { processes_running,
                        "The number of runnable (running or ready to run) threads by the operating system."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else if (name == "procs_blocked")
                {
                    UInt64 processes_blocked = 0;
                    readText(processes_blocked, *proc_stat);
                    skipToNextLineOrEOF(*proc_stat);
                    new_values["OSProcessesBlocked"] = { processes_blocked,
                        "Number of threads blocked waiting for I/O to complete (`man procfs`)."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else
                    skipToNextLineOrEOF(*proc_stat);
            }

            if (!first_run)
            {
                ProcStatValuesOther delta_values = current_other_values - proc_stat_values_other;

                new_values["OSInterrupts"] = { delta_values.interrupts, "The number of interrupts on the host machine. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                new_values["OSContextSwitches"] = { delta_values.context_switches, "The number of context switches that the system underwent on the host machine. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                new_values["OSProcessesCreated"] = { delta_values.processes_created, "The number of processes created. This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };

                /// Also write values normalized to 0..1 by diving to the number of CPUs.
                /// These values are good to be averaged across the cluster of non-uniform servers.

                if (num_cpus)
                {
                    new_values["OSUserTimeNormalized"] = { delta_values_all_cpus.user * multiplier / num_cpus,
                        "The value is similar to `OSUserTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSNiceTimeNormalized"] = { delta_values_all_cpus.nice * multiplier / num_cpus,
                        "The value is similar to `OSNiceTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSSystemTimeNormalized"] = { delta_values_all_cpus.system * multiplier / num_cpus,
                        "The value is similar to `OSSystemTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSIdleTimeNormalized"] = { delta_values_all_cpus.idle * multiplier / num_cpus,
                        "The value is similar to `OSIdleTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSIOWaitTimeNormalized"] = { delta_values_all_cpus.iowait * multiplier / num_cpus,
                        "The value is similar to `OSIOWaitTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSIrqTimeNormalized"] = { delta_values_all_cpus.irq * multiplier / num_cpus,
                        "The value is similar to `OSIrqTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSSoftIrqTimeNormalized"] = { delta_values_all_cpus.softirq * multiplier / num_cpus,
                        "The value is similar to `OSSoftIrqTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSStealTimeNormalized"] = { delta_values_all_cpus.steal * multiplier / num_cpus,
                        "The value is similar to `OSStealTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSGuestTimeNormalized"] = { delta_values_all_cpus.guest * multiplier / num_cpus,
                        "The value is similar to `OSGuestTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                    new_values["OSGuestNiceTimeNormalized"] = { delta_values_all_cpus.guest_nice * multiplier / num_cpus,
                        "The value is similar to `OSGuestNiceTime` but divided to the number of CPU cores to be measured in the [0..1] interval regardless of the number of cores."
                        " This allows you to average the values of this metric across multiple servers in a cluster even if the number of cores is non-uniform, and still get the average resource utilization metric."};
                }
            }

            proc_stat_values_other = current_other_values;
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    if (meminfo)
    {
        try
        {
            meminfo->rewind();

            uint64_t free_plus_cached_bytes = 0;

            while (!meminfo->eof())
            {
                String name;
                readStringUntilWhitespace(name, *meminfo);
                skipWhitespaceIfAny(*meminfo, true);

                uint64_t kb = 0;
                readText(kb, *meminfo);

                if (!kb)
                {
                    skipToNextLineOrEOF(*meminfo);
                    continue;
                }

                skipWhitespaceIfAny(*meminfo, true);

                /**
                 * Not all entries in /proc/meminfo contain the kB suffix, e.g.
                 * HugePages_Total:       0
                 * HugePages_Free:        0
                 * We simply skip such entries as they're not needed
                 */
                if (*meminfo->position() == '\n')
                {
                    skipToNextLineOrEOF(*meminfo);
                    continue;
                }

                assertString("kB", *meminfo);

                uint64_t bytes = kb * 1024;

                if (name == "MemTotal:")
                {
                    new_values["OSMemoryTotal"] = { bytes, "The total amount of memory on the host system, in bytes." };
                }
                else if (name == "MemFree:")
                {
                    free_plus_cached_bytes += bytes;
                    new_values["OSMemoryFreeWithoutCached"] = { bytes,
                        "The amount of free memory on the host system, in bytes."
                        " This does not include the memory used by the OS page cache memory, in bytes."
                        " The page cache memory is also available for usage by programs, so the value of this metric can be confusing."
                        " See the `OSMemoryAvailable` metric instead."
                        " For convenience we also provide the `OSMemoryFreePlusCached` metric, that should be somewhat similar to OSMemoryAvailable."
                        " See also https://www.linuxatemyram.com/."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else if (name == "MemAvailable:")
                {
                    new_values["OSMemoryAvailable"] = { bytes, "The amount of memory available to be used by programs, in bytes. This is very similar to the `OSMemoryFreePlusCached` metric."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else if (name == "Buffers:")
                {
                    new_values["OSMemoryBuffers"] = { bytes, "The amount of memory used by OS kernel buffers, in bytes. This should be typically small, and large values may indicate a misconfiguration of the OS."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else if (name == "Cached:")
                {
                    free_plus_cached_bytes += bytes;
                    new_values["OSMemoryCached"] = { bytes, "The amount of memory used by the OS page cache, in bytes. Typically, almost all available memory is used by the OS page cache - high values of this metric are normal and expected."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
                else if (name == "SwapCached:")
                {
                    new_values["OSMemorySwapCached"] = { bytes, "The amount of memory in swap that was also loaded in RAM. Swap should be disabled on production systems. If the value of this metric is large, it indicates a misconfiguration."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }

                skipToNextLineOrEOF(*meminfo);
            }

            new_values["OSMemoryFreePlusCached"] = { free_plus_cached_bytes, "The amount of free memory plus OS page cache memory on the host system, in bytes. This memory is available to be used by programs. The value should be very similar to `OSMemoryAvailable`."
                " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    // Try to add processor frequencies, ignoring errors.
    if (cpuinfo)
    {
        try
        {
            cpuinfo->rewind();

            // We need the following lines:
            // processor : 4
            // cpu MHz : 4052.941
            // They contain tabs and are interspersed with other info.

            int core_id = 0;
            while (!cpuinfo->eof())
            {
                std::string s;
                // We don't have any backslash escape sequences in /proc/cpuinfo, so
                // this function will read the line until EOL, which is exactly what
                // we need.
                readEscapedStringUntilEOL(s, *cpuinfo);
                // It doesn't read the EOL itself.
                ++cpuinfo->position();

                if (s.rfind("processor", 0) == 0)
                {
                    /// s390x example: processor 0: version = FF, identification = 039C88, machine = 3906
                    /// non s390x example: processor : 0
                    if (auto colon = s.find_first_of(':'))
                    {
#ifdef __s390x__
                        core_id = std::stoi(s.substr(10)); /// 10: length of "processor" plus 1
#else
                        core_id = std::stoi(s.substr(colon + 2));
#endif
                    }
                }
                else if (s.rfind("cpu MHz", 0) == 0)
                {
                    if (auto colon = s.find_first_of(':'))
                    {
                        auto mhz = std::stod(s.substr(colon + 2));
                        new_values[fmt::format("CPUFrequencyMHz_{}", core_id)] = { mhz, "The current frequency of the CPU, in MHz. Most of the modern CPUs adjust the frequency dynamically for power saving and Turbo Boosting." };
                    }
                }
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    if (file_nr)
    {
        try
        {
            file_nr->rewind();

            uint64_t open_files = 0;
            readText(open_files, *file_nr);
            new_values["OSOpenFiles"] = { open_files, "The total number of opened files on the host machine."
                " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    /// Update list of block devices periodically
    /// (i.e. someone may add new disk to RAID array)
    if (block_devices_rescan_delay.elapsedSeconds() >= 300)
        openBlockDevices();

    try
    {
        for (auto & [name, device] : block_devs)
        {
            device->rewind();

            BlockDeviceStatValues current_values{};
            BlockDeviceStatValues & prev_values = block_device_stats[name];

            try
            {
                current_values.read(*device);
            }
            catch (const ErrnoException & e)
            {
                LOG_DEBUG(log, "Cannot read statistics about the block device '{}': {}.",
                    name, errnoToString(e.getErrno()));
                continue;
            }

            BlockDeviceStatValues delta_values = current_values - prev_values;
            prev_values = current_values;

            if (first_run)
                continue;

            /// Always 512 according to the docs.
            static constexpr size_t sector_size = 512;

            /// Always in milliseconds according to the docs.
            static constexpr double time_multiplier = 1e-6;

#define BLOCK_DEVICE_EXPLANATION \
    " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." \
    " See https://www.kernel.org/doc/Documentation/block/stat.txt"

            new_values["BlockReadOps_" + name] = { delta_values.read_ios,
                "Number of read operations requested from the block device."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockWriteOps_" + name] = { delta_values.write_ios,
                "Number of write operations requested from the block device."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockDiscardOps_" + name] = { delta_values.discard_ops,
                "Number of discard operations requested from the block device. These operations are relevant for SSD."
                " Discard operations are not used by ClickHouse, but can be used by other processes on the system."
                BLOCK_DEVICE_EXPLANATION };

            new_values["BlockReadMerges_" + name] = { delta_values.read_merges,
                "Number of read operations requested from the block device and merged together by the OS IO scheduler."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockWriteMerges_" + name] = { delta_values.write_merges,
                "Number of write operations requested from the block device and merged together by the OS IO scheduler."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockDiscardMerges_" + name] = { delta_values.discard_merges,
                "Number of discard operations requested from the block device and merged together by the OS IO scheduler."
                " These operations are relevant for SSD. Discard operations are not used by ClickHouse, but can be used by other processes on the system."
                BLOCK_DEVICE_EXPLANATION };

            new_values["BlockReadBytes_" + name] = { delta_values.read_sectors * sector_size,
                "Number of bytes read from the block device."
                " It can be lower than the number of bytes read from the filesystem due to the usage of the OS page cache, that saves IO."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockWriteBytes_" + name] = { delta_values.write_sectors * sector_size,
                "Number of bytes written to the block device."
                " It can be lower than the number of bytes written to the filesystem due to the usage of the OS page cache, that saves IO."
                " A write to the block device may happen later than the corresponding write to the filesystem due to write-through caching."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockDiscardBytes_" + name] = { delta_values.discard_sectors * sector_size,
                "Number of discarded bytes on the block device."
                " These operations are relevant for SSD. Discard operations are not used by ClickHouse, but can be used by other processes on the system."
                BLOCK_DEVICE_EXPLANATION };

            new_values["BlockReadTime_" + name] = { delta_values.read_ticks * time_multiplier,
                "Time in seconds spend in read operations requested from the block device, summed across all the operations."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockWriteTime_" + name] = { delta_values.write_ticks * time_multiplier,
                "Time in seconds spend in write operations requested from the block device, summed across all the operations."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockDiscardTime_" + name] = { delta_values.discard_ticks * time_multiplier,
                "Time in seconds spend in discard operations requested from the block device, summed across all the operations."
                " These operations are relevant for SSD. Discard operations are not used by ClickHouse, but can be used by other processes on the system."
                BLOCK_DEVICE_EXPLANATION };

            new_values["BlockInFlightOps_" + name] = { delta_values.in_flight_ios,
                "This value counts the number of I/O requests that have been issued to"
                " the device driver but have not yet completed. It does not include IO"
                " requests that are in the queue but not yet issued to the device driver."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockActiveTime_" + name] = { delta_values.io_ticks * time_multiplier,
                "Time in seconds the block device had the IO requests queued."
                BLOCK_DEVICE_EXPLANATION };
            new_values["BlockQueueTime_" + name] = { delta_values.time_in_queue * time_multiplier,
                "This value counts the number of milliseconds that IO requests have waited"
                " on this block device. If there are multiple IO requests waiting, this"
                " value will increase as the product of the number of milliseconds times the"
                " number of requests waiting."
                BLOCK_DEVICE_EXPLANATION };

            if (delta_values.in_flight_ios)
            {
                /// TODO Check if these values are meaningful.

                new_values["BlockActiveTimePerOp_" + name] = { delta_values.io_ticks * time_multiplier / delta_values.in_flight_ios,
                    "Similar to the `BlockActiveTime` metrics, but the value is divided to the number of IO operations to count the per-operation time." };
                new_values["BlockQueueTimePerOp_" + name] = { delta_values.time_in_queue * time_multiplier / delta_values.in_flight_ios,
                    "Similar to the `BlockQueueTime` metrics, but the value is divided to the number of IO operations to count the per-operation time." };
            }
        }
    }
    catch (...)
    {
        LOG_DEBUG(log, "Cannot read statistics from block devices: {}", getCurrentExceptionMessage(false));

        /// Try to reopen block devices in case of error
        /// (i.e. ENOENT or ENODEV means that some disk had been replaced, and it may appear with a new name)
        try
        {
            openBlockDevices();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    if (net_dev)
    {
        try
        {
            net_dev->rewind();

            /// Skip first two lines:
            /// Inter-|   Receive                                                |  Transmit
            ///  face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed

            skipToNextLineOrEOF(*net_dev);
            skipToNextLineOrEOF(*net_dev);

            while (!net_dev->eof())
            {
                skipWhitespaceIfAny(*net_dev, true);
                String interface_name;
                readStringUntilWhitespace(interface_name, *net_dev);

                /// We are not interested in loopback devices.
                if (!interface_name.ends_with(':') || interface_name == "lo:" || interface_name.size() <= 1)
                {
                    skipToNextLineOrEOF(*net_dev);
                    continue;
                }

                interface_name.pop_back();

                NetworkInterfaceStatValues current_values{};
                uint64_t unused;

                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.recv_bytes, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.recv_packets, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.recv_errors, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.recv_drop, *net_dev);

                /// NOTE We should pay more attention to the number of fields.

                skipWhitespaceIfAny(*net_dev, true);
                readText(unused, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(unused, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(unused, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(unused, *net_dev);

                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.send_bytes, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.send_packets, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.send_errors, *net_dev);
                skipWhitespaceIfAny(*net_dev, true);
                readText(current_values.send_drop, *net_dev);

                skipToNextLineOrEOF(*net_dev);

                NetworkInterfaceStatValues & prev_values = network_interface_stats[interface_name];
                NetworkInterfaceStatValues delta_values = current_values - prev_values;
                prev_values = current_values;

                if (!first_run)
                {
                    new_values["NetworkReceiveBytes_" + interface_name] = { delta_values.recv_bytes,
                        " Number of bytes received via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkReceivePackets_" + interface_name] = { delta_values.recv_packets,
                        " Number of network packets received via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkReceiveErrors_" + interface_name] = { delta_values.recv_errors,
                        " Number of times error happened receiving via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkReceiveDrop_" + interface_name] = { delta_values.recv_drop,
                        " Number of bytes a packet was dropped while received via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };

                    new_values["NetworkSendBytes_" + interface_name] = { delta_values.send_bytes,
                        " Number of bytes sent via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkSendPackets_" + interface_name] = { delta_values.send_packets,
                        " Number of network packets sent via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkSendErrors_" + interface_name] = { delta_values.send_errors,
                        " Number of times error (e.g. TCP retransmit) happened while sending via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                    new_values["NetworkSendDrop_" + interface_name] = { delta_values.send_drop,
                        " Number of times a packed was dropped while sending via the network interface."
                        " This is a system-wide metric, it includes all the processes on the host machine, not just clickhouse-server." };
                }
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    try
    {
        for (size_t i = 0, size = thermal.size(); i < size; ++i)
        {
            ReadBufferFromFilePRead & in = *thermal[i];

            in.rewind();
            Int64 temperature = 0;
            readText(temperature, in);
            new_values[fmt::format("Temperature{}", i)] = { temperature * 0.001,
                "The temperature of the corresponding device in ℃. A sensor can return an unrealistic value." };
        }
    }
    catch (...)
    {
        if (errno != ENODATA)   /// Ok for thermal sensors.
            tryLogCurrentException(__PRETTY_FUNCTION__);

        /// Files maybe re-created on module load/unload
        try
        {
            openSensors();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    try
    {
        for (const auto & [hwmon_name, sensors] : hwmon_devices)
        {
            for (const auto & [sensor_name, sensor_file] : sensors)
            {
                sensor_file->rewind();
                Int64 temperature = 0;
                try
                {
                    readText(temperature, *sensor_file);
                }
                catch (const ErrnoException & e)
                {
                    LOG_DEBUG(log, "Hardware monitor '{}', sensor '{}' exists but could not be read: {}.",
                        hwmon_name, sensor_name, errnoToString(e.getErrno()));
                    continue;
                }

                if (sensor_name.empty())
                    new_values[fmt::format("Temperature_{}", hwmon_name)] = { temperature * 0.001,
                        "The temperature reported by the corresponding hardware monitor in ℃. A sensor can return an unrealistic value." };
                else
                    new_values[fmt::format("Temperature_{}_{}", hwmon_name, sensor_name)] = { temperature * 0.001,
                        "The temperature reported by the corresponding hardware monitor and the corresponding sensor in ℃. A sensor can return an unrealistic value." };
            }
        }
    }
    catch (...)
    {
        if (errno != ENODATA)   /// Ok for thermal sensors.
            tryLogCurrentException(__PRETTY_FUNCTION__);

        /// Files can be re-created on:
        /// - module load/unload
        /// - suspend/resume cycle
        /// So file descriptors should be reopened.
        try
        {
            openSensorsChips();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    try
    {
        for (size_t i = 0, size = edac.size(); i < size; ++i)
        {
            /// NOTE maybe we need to take difference with previous values.
            /// But these metrics should be exceptionally rare, so it's ok to keep them accumulated.

            if (edac[i].first)
            {
                ReadBufferFromFilePRead & in = *edac[i].first;
                in.rewind();
                uint64_t errors = 0;
                readText(errors, in);
                new_values[fmt::format("EDAC{}_Correctable", i)] = { errors,
                    "The number of correctable ECC memory errors."
                    " A high number of this value indicates bad RAM which has to be immediately replaced,"
                    " because in presence of a high number of corrected errors, a number of silent errors may happen as well, leading to data corruption." };
            }

            if (edac[i].second)
            {
                ReadBufferFromFilePRead & in = *edac[i].second;
                in.rewind();
                uint64_t errors = 0;
                readText(errors, in);
                new_values[fmt::format("EDAC{}_Uncorrectable", i)] = { errors,
                    "The number of uncorrectable ECC memory errors."
                    " A non-zero number of this value indicates bad RAM which has to be immediately replaced,"
                    " because it indicates potential data corruption." };
            }
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);

        /// EDAC files can be re-created on module load/unload
        try
        {
            openEDAC();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
#endif

    /// Free space in filesystems at data path and logs path.
    {
        auto stat = getStatVFS(getContext()->getPath());

        new_values["FilesystemMainPathTotalBytes"] = { stat.f_blocks * stat.f_frsize,
            "The size of the volume where the main ClickHouse path is mounted, in bytes." };
        new_values["FilesystemMainPathAvailableBytes"] = { stat.f_bavail * stat.f_frsize,
            "Available bytes on the volume where the main ClickHouse path is mounted." };
        new_values["FilesystemMainPathUsedBytes"] = { (stat.f_blocks - stat.f_bavail) * stat.f_frsize,
            "Used bytes on the volume where the main ClickHouse path is mounted." };
        new_values["FilesystemMainPathTotalINodes"] = { stat.f_files,
            "The total number of inodes on the volume where the main ClickHouse path is mounted. If it is less than 25 million, it indicates a misconfiguration." };
        new_values["FilesystemMainPathAvailableINodes"] = { stat.f_favail,
            "The number of available inodes on the volume where the main ClickHouse path is mounted. If it is close to zero, it indicates a misconfiguration, and you will get 'no space left on device' even when the disk is not full." };
        new_values["FilesystemMainPathUsedINodes"] = { stat.f_files - stat.f_favail,
            "The number of used inodes on the volume where the main ClickHouse path is mounted. This value mostly corresponds to the number of files." };
    }

    {
        /// Current working directory of the server is the directory with logs.
        auto stat = getStatVFS(".");

        new_values["FilesystemLogsPathTotalBytes"] = { stat.f_blocks * stat.f_frsize,
            "The size of the volume where ClickHouse logs path is mounted, in bytes. It's recommended to have at least 10 GB for logs." };
        new_values["FilesystemLogsPathAvailableBytes"] = { stat.f_bavail * stat.f_frsize,
            "Available bytes on the volume where ClickHouse logs path is mounted. If this value approaches zero, you should tune the log rotation in the configuration file." };
        new_values["FilesystemLogsPathUsedBytes"] = { (stat.f_blocks - stat.f_bavail) * stat.f_frsize,
            "Used bytes on the volume where ClickHouse logs path is mounted." };
        new_values["FilesystemLogsPathTotalINodes"] = { stat.f_files,
            "The total number of inodes on the volume where ClickHouse logs path is mounted." };
        new_values["FilesystemLogsPathAvailableINodes"] = { stat.f_favail,
            "The number of available inodes on the volume where ClickHouse logs path is mounted." };
        new_values["FilesystemLogsPathUsedINodes"] = { stat.f_files - stat.f_favail,
            "The number of used inodes on the volume where ClickHouse logs path is mounted." };
    }

    /// Free and total space on every configured disk.
    {
        DisksMap disks_map = getContext()->getDisksMap();
        for (const auto & [name, disk] : disks_map)
        {
            auto total = disk->getTotalSpace();

            /// Some disks don't support information about the space.
            if (!total)
                continue;

            auto available = disk->getAvailableSpace();
            auto unreserved = disk->getUnreservedSpace();

            new_values[fmt::format("DiskTotal_{}", name)] = { total,
                "The total size in bytes of the disk (virtual filesystem). Remote filesystems can show a large value like 16 EiB." };
            new_values[fmt::format("DiskUsed_{}", name)] = { total - available,
                "Used bytes on the disk (virtual filesystem). Remote filesystems not always provide this information." };
            new_values[fmt::format("DiskAvailable_{}", name)] = { available,
                "Available bytes on the disk (virtual filesystem). Remote filesystems can show a large value like 16 EiB." };
            new_values[fmt::format("DiskUnreserved_{}", name)] = { unreserved,
                "Available bytes on the disk (virtual filesystem) without the reservations for merges, fetches, and moves. Remote filesystems can show a large value like 16 EiB." };
        }
    }

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

                if (MergeTreeData * table_merge_tree = dynamic_cast<MergeTreeData *>(table.get()))
                {
                    const auto & settings = getContext()->getSettingsRef();

                    calculateMax(max_part_count_for_partition, table_merge_tree->getMaxPartsCountAndSizeForPartition().first);
                    total_number_of_bytes += table_merge_tree->totalBytes(settings).value();
                    total_number_of_rows += table_merge_tree->totalRows(settings).value();
                    total_number_of_parts += table_merge_tree->getPartsCount();
                }

                if (StorageReplicatedMergeTree * table_replicated_merge_tree = typeid_cast<StorageReplicatedMergeTree *>(table.get()))
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
                }
            }
        }

        new_values["ReplicasMaxQueueSize"] = { max_queue_size, "Maximum queue size (in the number of operations like get, merge) across Replicated tables." };
        new_values["ReplicasMaxInsertsInQueue"] = { max_inserts_in_queue, "Maximum number of INSERT operations in the queue (still to be replicated) across Replicated tables." };
        new_values["ReplicasMaxMergesInQueue"] = { max_merges_in_queue, "Maximum number of merge operations in the queue (still to be applied) across Replicated tables." };

        new_values["ReplicasSumQueueSize"] = { sum_queue_size, "Sum queue size (in the number of operations like get, merge) across Replicated tables." };
        new_values["ReplicasSumInsertsInQueue"] = { sum_inserts_in_queue, "Sum of INSERT operations in the queue (still to be replicated) across Replicated tables." };
        new_values["ReplicasSumMergesInQueue"] = { sum_merges_in_queue, "Sum of merge operations in the queue (still to be applied) across Replicated tables." };

        new_values["ReplicasMaxAbsoluteDelay"] = { max_absolute_delay, "Maximum difference in seconds between the most fresh replicated part and the most fresh data part still to be replicated, across Replicated tables. A very high value indicates a replica with no data." };
        new_values["ReplicasMaxRelativeDelay"] = { max_relative_delay, "Maximum difference between the replica delay and the delay of the most up-to-date replica of the same table, across Replicated tables." };

        new_values["MaxPartCountForPartition"] = { max_part_count_for_partition, "Maximum number of parts per partition across all partitions of all tables of MergeTree family. Values larger than 300 indicates misconfiguration, overload, or massive data loading." };

        new_values["NumberOfDatabases"] = { number_of_databases, "Total number of databases on the server." };
        new_values["NumberOfTables"] = { total_number_of_tables, "Total number of tables summed across the databases on the server, excluding the databases that cannot contain MergeTree tables."
            " The excluded database engines are those who generate the set of tables on the fly, like `Lazy`, `MySQL`, `PostgreSQL`, `SQlite`."};

        new_values["TotalBytesOfMergeTreeTables"] = { total_number_of_bytes, "Total amount of bytes (compressed, including data and indices) stored in all tables of MergeTree family." };
        new_values["TotalRowsOfMergeTreeTables"] = { total_number_of_rows, "Total amount of rows (records) stored in all tables of MergeTree family." };
        new_values["TotalPartsOfMergeTreeTables"] = { total_number_of_parts, "Total amount of data parts in all tables of MergeTree family."
            " Numbers larger than 10 000 will negatively affect the server startup time and it may indicate unreasonable choice of the partition key." };

        auto get_metric_name_doc = [](const String & name) -> std::pair<const char *, const char *>
        {
            static std::map<String, std::pair<const char *, const char *>> metric_map =
            {
                {"tcp_port", {"TCPThreads", "Number of threads in the server of the TCP protocol (without TLS)."}},
                {"tcp_port_secure", {"TCPSecureThreads", "Number of threads in the server of the TCP protocol (with TLS)."}},
                {"http_port", {"HTTPThreads", "Number of threads in the server of the HTTP interface (without TLS)."}},
                {"https_port", {"HTTPSecureThreads", "Number of threads in the server of the HTTPS interface."}},
                {"interserver_http_port", {"InterserverThreads", "Number of threads in the server of the replicas communication protocol (without TLS)."}},
                {"interserver_https_port", {"InterserverSecureThreads", "Number of threads in the server of the replicas communication protocol (with TLS)."}},
                {"mysql_port", {"MySQLThreads", "Number of threads in the server of the MySQL compatibility protocol."}},
                {"postgresql_port", {"PostgreSQLThreads", "Number of threads in the server of the PostgreSQL compatibility protocol."}},
                {"grpc_port", {"GRPCThreads", "Number of threads in the server of the GRPC protocol."}},
                {"prometheus.port", {"PrometheusThreads", "Number of threads in the server of the Prometheus endpoint. Note: prometheus endpoints can be also used via the usual HTTP/HTTPs ports."}}
            };
            auto it = metric_map.find(name);
            if (it == metric_map.end())
                return { nullptr, nullptr };
            else
                return it->second;
        };

        const auto server_metrics = protocol_server_metrics_func();
        for (const auto & server_metric : server_metrics)
        {
            if (auto name_doc = get_metric_name_doc(server_metric.port_name); name_doc.first != nullptr)
                new_values[name_doc.first] = { server_metric.current_threads, name_doc.second };
        }
    }
#if USE_NURAFT
    {
        auto keeper_dispatcher = getContext()->tryGetKeeperDispatcher();
        if (keeper_dispatcher)
        {
            size_t is_leader = 0;
            size_t is_follower = 0;
            size_t is_observer = 0;
            size_t is_standalone = 0;
            size_t znode_count = 0;
            size_t watch_count = 0;
            size_t ephemerals_count = 0;
            size_t approximate_data_size = 0;
            size_t key_arena_size = 0;
            size_t latest_snapshot_size = 0;
            size_t open_file_descriptor_count = 0;
            size_t max_file_descriptor_count = 0;
            size_t followers = 0;
            size_t synced_followers = 0;
            size_t zxid = 0;
            size_t session_with_watches = 0;
            size_t paths_watched = 0;
            size_t snapshot_dir_size = 0;
            size_t log_dir_size = 0;

            if (keeper_dispatcher->isServerActive())
            {
                auto keeper_info = keeper_dispatcher -> getKeeper4LWInfo();
                is_standalone = static_cast<size_t>(keeper_info.is_standalone);
                is_leader = static_cast<size_t>(keeper_info.is_leader);
                is_observer = static_cast<size_t>(keeper_info.is_observer);
                is_follower = static_cast<size_t>(keeper_info.is_follower);

                zxid = keeper_info.last_zxid;
                const auto & state_machine = keeper_dispatcher->getStateMachine();
                znode_count = state_machine.getNodesCount();
                watch_count = state_machine.getTotalWatchesCount();
                ephemerals_count = state_machine.getTotalEphemeralNodesCount();
                approximate_data_size = state_machine.getApproximateDataSize();
                key_arena_size = state_machine.getKeyArenaSize();
                latest_snapshot_size = state_machine.getLatestSnapshotBufSize();
                session_with_watches = state_machine.getSessionsWithWatchesCount();
                paths_watched = state_machine.getWatchedPathsCount();
                snapshot_dir_size = keeper_dispatcher->getSnapDirSize();
                log_dir_size = keeper_dispatcher->getLogDirSize();

                #if defined(__linux__) || defined(__APPLE__)
                    open_file_descriptor_count = getCurrentProcessFDCount();
                    max_file_descriptor_count = getMaxFileDescriptorCount();
                #endif

                if (keeper_info.is_leader)
                {
                    followers = keeper_info.follower_count;
                    synced_followers = keeper_info.synced_follower_count;
                }
            }

            new_values["KeeperIsLeader"] = { is_leader, "1 if ClickHouse Keeper is a leader, 0 otherwise." };
            new_values["KeeperIsFollower"] = { is_follower, "1 if ClickHouse Keeper is a follower, 0 otherwise." };
            new_values["KeeperIsObserver"] = { is_observer, "1 if ClickHouse Keeper is an observer, 0 otherwise." };
            new_values["KeeperIsStandalone"] = { is_standalone, "1 if ClickHouse Keeper is in a standalone mode, 0 otherwise." };

            new_values["KeeperZnodeCount"] = { znode_count, "The number of nodes (data entries) in ClickHouse Keeper." };
            new_values["KeeperWatchCount"] = { watch_count, "The number of watches in ClickHouse Keeper." };
            new_values["KeeperEphemeralsCount"] = { ephemerals_count, "The number of ephemeral nodes in ClickHouse Keeper." };

            new_values["KeeperApproximateDataSize"] = { approximate_data_size, "The approximate data size of ClickHouse Keeper, in bytes." };
            new_values["KeeperKeyArenaSize"] = { key_arena_size, "The size in bytes of the memory arena for keys in ClickHouse Keeper." };
            new_values["KeeperLatestSnapshotSize"] = { latest_snapshot_size, "The uncompressed size in bytes of the latest snapshot created by ClickHouse Keeper." };

            new_values["KeeperOpenFileDescriptorCount"] = { open_file_descriptor_count, "The number of open file descriptors in ClickHouse Keeper." };
            new_values["KeeperMaxFileDescriptorCount"] = { max_file_descriptor_count, "The maximum number of open file descriptors in ClickHouse Keeper." };

            new_values["KeeperFollowers"] = { followers, "The number of followers of ClickHouse Keeper." };
            new_values["KeeperSyncedFollowers"] = { synced_followers, "The number of followers of ClickHouse Keeper who are also in-sync." };
            new_values["KeeperZxid"] = { zxid, "The current transaction id number (zxid) in ClickHouse Keeper." };
            new_values["KeeperSessionWithWatches"] = { session_with_watches, "The number of client sessions of ClickHouse Keeper having watches." };
            new_values["KeeperPathsWatched"] = { paths_watched, "The number of different paths watched by the clients of ClickHouse Keeper." };
            new_values["KeeperSnapshotDirSize"] = { snapshot_dir_size, "The size of the snapshots directory of ClickHouse Keeper, in bytes." };
            new_values["KeeperLogDirSize"] = { log_dir_size, "The size of the logs directory of ClickHouse Keeper, in bytes." };
        }
    }
#endif

    updateHeavyMetricsIfNeeded(current_time, update_time, new_values);

    /// Add more metrics as you wish.

    new_values["AsynchronousMetricsCalculationTimeSpent"] = { watch.elapsedSeconds(), "Time in seconds spent for calculation of asynchronous metrics (this is the overhead of asynchronous metrics)." };

    /// Log the new metrics.
    if (auto asynchronous_metric_log = getContext()->getAsynchronousMetricLog())
    {
        asynchronous_metric_log->addValues(new_values);
    }

    first_run = false;

    // Finally, update the current metrics.
    std::lock_guard lock(mutex);
    values = new_values;
}

void AsynchronousMetrics::updateDetachedPartsStats()
{
    DetachedPartsStats current_values{};

    for (const auto & db : DatabaseCatalog::instance().getDatabases())
    {
        if (!db.second->canContainMergeTreeTables())
            continue;

        for (auto iterator = db.second->getTablesIterator(getContext()); iterator->isValid(); iterator->next())
        {
            const auto & table = iterator->table();
            if (!table)
                continue;

            if (MergeTreeData * table_merge_tree = dynamic_cast<MergeTreeData *>(table.get()))
            {
                for (const auto & detached_part: table_merge_tree->getDetachedParts())
                {
                    if (!detached_part.valid_name)
                        continue;

                    if (detached_part.prefix.empty())
                        ++current_values.detached_by_user;

                    ++current_values.count;
                }
            }
        }
    }

    detached_parts_stats = current_values;
}

void AsynchronousMetrics::updateHeavyMetricsIfNeeded(TimePoint current_time, TimePoint update_time, AsynchronousMetricValues & new_values)
{
    const auto time_after_previous_update = current_time - heavy_metric_previous_update_time;
    const bool update_heavy_metric = time_after_previous_update >= heavy_metric_update_period || first_run;

    if (update_heavy_metric)
    {
        heavy_metric_previous_update_time = update_time;

        Stopwatch watch;

        /// Test shows that listing 100000 entries consuming around 0.15 sec.
        updateDetachedPartsStats();

        watch.stop();

        /// Normally heavy metrics don't delay the rest of the metrics calculation
        /// otherwise log the warning message
        auto log_level = std::make_pair(DB::LogsLevel::trace, Poco::Message::PRIO_TRACE);
        if (watch.elapsedSeconds() > (update_period.count() / 2.))
            log_level = std::make_pair(DB::LogsLevel::debug, Poco::Message::PRIO_DEBUG);
        else if (watch.elapsedSeconds() > (update_period.count() / 4. * 3))
            log_level = std::make_pair(DB::LogsLevel::warning, Poco::Message::PRIO_WARNING);
        LOG_IMPL(log, log_level.first, log_level.second,
                 "Update heavy metrics. "
                 "Update period {} sec. "
                 "Update heavy metrics period {} sec. "
                 "Heavy metrics calculation elapsed: {} sec.",
                 update_period.count(),
                 heavy_metric_update_period.count(),
                 watch.elapsedSeconds());
    }

    new_values["NumberOfDetachedParts"] = { detached_parts_stats.count, "The total number of parts detached from MergeTree tables. A part can be detached by a user with the `ALTER TABLE DETACH` query or by the server itself it the part is broken, unexpected or unneeded. The server does not care about detached parts and they can be removed." };
    new_values["NumberOfDetachedByUserParts"] = { detached_parts_stats.detached_by_user, "The total number of parts detached from MergeTree tables by users with the `ALTER TABLE DETACH` query (as opposed to unexpected, broken or ignored parts). The server does not care about detached parts and they can be removed." };
}

}
