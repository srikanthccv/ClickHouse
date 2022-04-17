#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/AsynchronousMetricLog.h>
#include <Interpreters/AsynchronousMetrics.h>


namespace DB
{

NamesAndTypesList AsynchronousMetricLogElement::getNamesAndTypes()
{
    return
    {
        {"event_date", std::make_shared<DataTypeDate>()},
        {"event_time", std::make_shared<DataTypeDateTime>()},
        {"metric", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())},
        {"value", std::make_shared<DataTypeFloat64>(),}
    };
}

void AsynchronousMetricLogElement::appendToBlock(MutableColumns & columns) const
{
    size_t column_idx = 0;

    columns[column_idx++]->insert(event_date);
    columns[column_idx++]->insert(event_time);
    columns[column_idx++]->insert(metric_name);
    columns[column_idx++]->insert(value);
}


static inline UInt64 time_in_seconds(std::chrono::time_point<std::chrono::system_clock> timepoint)
{
    return std::chrono::duration_cast<std::chrono::seconds>(timepoint.time_since_epoch()).count();
}

void AsynchronousMetricLog::addValues(const AsynchronousMetricValues & values)
{
    AsynchronousMetricLogElement element;

    const auto now = std::chrono::system_clock::now();
    element.event_time = time_in_seconds(now);
    element.event_date = DateLUT::instance().toDayNum(element.event_time);

    for (const auto & [key, value] : values)
    {
        element.metric_name = key;
        element.value = value;

        add(element);
    }
}

}
