#pragma once

#include <string>

#include <Interpreters/AsynchronousMetrics.h>

#include <IO/WriteBuffer.h>

#include <Poco/Util/AbstractConfiguration.h>

namespace DB
{

/// Write metrics in Prometheus format
class PrometheusMetricsWriter
{
public:
    PrometheusMetricsWriter(
        const Poco::Util::AbstractConfiguration & config, const std::string & config_name,
        const AsynchronousMetrics & async_metrics_);

    void write(WriteBuffer & wb) const;

private:
    const AsynchronousMetrics & async_metrics;

    const bool send_events;
    const bool send_metrics;
    const bool send_asynchronous_metrics;

    static inline constexpr auto profile_events_prefix = "ClickHouseProfileEvents";
    static inline constexpr auto current_metrics_prefix = "ClickHouseMetrics";
    static inline constexpr auto asynchronous_metrics_prefix = "ClickHouseAsyncMetrics";
};

}
