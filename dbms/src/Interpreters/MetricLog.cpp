#include <Interpreters/MetricLog.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>

namespace DB
{

Block MetricLogElement::createBlock()
{
    ColumnsWithTypeAndName columns_with_type_and_name;

    columns_with_type_and_name.emplace_back(DataTypeDate(),       "event_date");
    columns_with_type_and_name.emplace_back(DataTypeDateTime(),   "event_time");

    //ProfileEvents
    for (size_t i = 0, end = ProfileEvents::end(); i < end; ++i)
    {
        std::string name;
        name += "ProfileEvent_";
        name += ProfileEvents::getName(ProfileEvents::Event(i));
        columns_with_type_and_name.emplace_back(DataTypeUInt64(), name);
    }

    //CurrentMetrics
    for (size_t i = 0, end = CurrentMetrics::end(); i < end; ++i)
    {
        std::string name;
        name += "CurrentMetric_";
        name += CurrentMetrics::getName(ProfileEvents::Event(i));
        columns_with_type_and_name.emplace_back(DataTypeInt64(), name);
    }

    return Block(columns_with_type_and_name);
}

void MetricLogElement::appendToBlock(Block & block) const
{
    MutableColumns columns = block.mutateColumns();

    size_t iter = 0;

    columns[iter++]->insert(DateLUT::instance().toDayNum(event_time));
    columns[iter++]->insert(event_time);

    //ProfileEvents
    for (size_t i = 0, end = ProfileEvents::end(); i < end; ++i)
    {
        const UInt64 value = ProfileEvents::global_counters[i].load(std::memory_order_relaxed);
        columns[iter++]->insert(value);
    }

    //CurrentMetrics
    for (size_t i = 0, end = CurrentMetrics::end(); i < end; ++i)
    {
        const UInt64 value = CurrentMetrics::values[i];
        columns[iter++]->insert(value);
    }
}

void MetricLog::startCollectMetric(size_t collect_interval_milliseconds_)
{
    collect_interval_milliseconds = collect_interval_milliseconds_;
    is_shutdown_metric_thread = false;
    metric_flush_thread = ThreadFromGlobalPool([this] { metricThreadFunction(); });
}

void MetricLog::stopCollectMetric()
{
    bool old_val = false;
    if (!is_shutdown_metric_thread.compare_exchange_strong(old_val, true))
        return;
    metric_flush_thread.join();
}

void MetricLog::metricThreadFunction()
{
    auto desired_timepoint = std::chrono::system_clock::now();
    while (true)
    {
        try
        {
            const auto prev_timepoint = desired_timepoint;

            if (is_shutdown_metric_thread)
                break;

            MetricLogElement elem;
            elem.event_time = std::chrono::system_clock::to_time_t(prev_timepoint);
            this->add(elem);

            desired_timepoint = prev_timepoint + std::chrono::milliseconds(collect_interval_milliseconds);
            std::this_thread::sleep_until(desired_timepoint);
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }

    }
}

}
