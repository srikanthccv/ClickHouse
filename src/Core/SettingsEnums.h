#pragma once

#include <Core/SettingsFields.h>
#include <QueryPipeline/SizeLimits.h>
#include <Formats/FormatSettings.h>


namespace DB
{
enum class LoadBalancing
{
    /// among replicas with a minimum number of errors selected randomly
    RANDOM = 0,
    /// a replica is selected among the replicas with the minimum number of errors
    /// with the minimum number of distinguished characters in the replica name and local hostname
    NEAREST_HOSTNAME,
    // replicas with the same number of errors are accessed in the same order
    // as they are specified in the configuration.
    IN_ORDER,
    /// if first replica one has higher number of errors,
    ///   pick a random one from replicas with minimum number of errors
    FIRST_OR_RANDOM,
    // round robin across replicas with the same number of errors.
    ROUND_ROBIN,
};

DECLARE_SETTING_ENUM(LoadBalancing)


enum class JoinStrictness
{
    Unspecified = 0, /// Query JOIN without strictness will throw Exception.
    ALL, /// Query JOIN without strictness -> ALL JOIN ...
    ANY, /// Query JOIN without strictness -> ANY JOIN ...
};

DECLARE_SETTING_ENUM(JoinStrictness)

enum class JoinAlgorithm
{
    AUTO = 0,
    HASH,
    PARTIAL_MERGE,
    PREFER_PARTIAL_MERGE,
};

DECLARE_SETTING_ENUM(JoinAlgorithm)


/// Which rows should be included in TOTALS.
enum class TotalsMode
{
    BEFORE_HAVING            = 0, /// Count HAVING for all read rows;
                                  ///  including those not in max_rows_to_group_by
                                  ///  and have not passed HAVING after grouping.
    AFTER_HAVING_INCLUSIVE    = 1, /// Count on all rows except those that have not passed HAVING;
                                   ///  that is, to include in TOTALS all the rows that did not pass max_rows_to_group_by.
    AFTER_HAVING_EXCLUSIVE    = 2, /// Include only the rows that passed and max_rows_to_group_by, and HAVING.
    AFTER_HAVING_AUTO         = 3, /// Automatically select between INCLUSIVE and EXCLUSIVE,
};

DECLARE_SETTING_ENUM(TotalsMode)


/// The settings keeps OverflowMode which cannot be OverflowMode::ANY.
DECLARE_SETTING_ENUM(OverflowMode)

/// The settings keeps OverflowMode which can be OverflowMode::ANY.
DECLARE_SETTING_ENUM_WITH_RENAME(OverflowModeGroupBy, OverflowMode)


/// The setting for executing distributed subqueries inside IN or JOIN sections.
enum class DistributedProductMode
{
    DENY = 0,    /// Disable
    LOCAL,       /// Convert to local query
    GLOBAL,      /// Convert to global query
    ALLOW        /// Enable
};

DECLARE_SETTING_ENUM(DistributedProductMode)


DECLARE_SETTING_ENUM_WITH_RENAME(DateTimeInputFormat, FormatSettings::DateTimeInputFormat)

DECLARE_SETTING_ENUM_WITH_RENAME(DateTimeOutputFormat, FormatSettings::DateTimeOutputFormat)

enum class LogsLevel
{
    none = 0,    /// Disable
    fatal,
    error,
    warning,
    information,
    debug,
    trace,
    test,
};

DECLARE_SETTING_ENUM(LogsLevel)


// Make it signed for compatibility with DataTypeEnum8
enum QueryLogElementType : int8_t
{
    QUERY_START = 1,
    QUERY_FINISH = 2,
    EXCEPTION_BEFORE_START = 3,
    EXCEPTION_WHILE_PROCESSING = 4,
};

DECLARE_SETTING_ENUM_WITH_RENAME(LogQueriesType, QueryLogElementType)


enum class DefaultDatabaseEngine
{
    Ordinary,
    Atomic,
};

DECLARE_SETTING_ENUM(DefaultDatabaseEngine)


enum class MySQLDataTypesSupport
{
    DECIMAL, // convert MySQL's decimal and number to ClickHouse Decimal when applicable
    DATETIME64, // convert MySQL's DATETIME and TIMESTAMP and ClickHouse DateTime64 if precision is > 0 or range is greater that for DateTime.
    // ENUM
};

DECLARE_SETTING_MULTI_ENUM(MySQLDataTypesSupport)

enum class UnionMode
{
    Unspecified = 0, // Query UNION without UnionMode will throw exception
    ALL, // Query UNION without UnionMode -> SELECT ... UNION ALL SELECT ...
    DISTINCT // Query UNION without UnionMode -> SELECT ... UNION DISTINCT SELECT ...
};

DECLARE_SETTING_ENUM(UnionMode)

enum class DistributedDDLOutputMode
{
    NONE,
    THROW,
    NULL_STATUS_ON_TIMEOUT,
    NEVER_THROW,
};

DECLARE_SETTING_ENUM(DistributedDDLOutputMode)

enum class HandleKafkaErrorMode
{
    DEFAULT = 0, // Ignore errors whit threshold.
    STREAM, // Put errors to stream in the virtual column named ``_error.
    /*FIXED_SYSTEM_TABLE, Put errors to in a fixed system table likey system.kafka_errors. This is not implemented now.  */
    /*CUSTOM_SYSTEM_TABLE, Put errors to in a custom system table. This is not implemented now.  */
};

DECLARE_SETTING_ENUM(HandleKafkaErrorMode)

enum class ShortCircuitFunctionEvaluation
{
    ENABLE, // Use short-circuit function evaluation for functions that are suitable for it.
    FORCE_ENABLE, // Use short-circuit function evaluation for all functions.
    DISABLE, // Disable short-circuit function evaluation.
};

DECLARE_SETTING_ENUM(ShortCircuitFunctionEvaluation)

enum class ZooKeeperLoadBalancing
{
    /// Randomly select one from the zookeeper nodes.
    RANDOM = 0,
    /// Choose one from the zookeeper node that has the least
    /// number of characters different from the hostname of the local host
    NEAREST_HOSTNAME,
    /// Select one from the zookeeper node configuration in order.
    IN_ORDER,
    /// If the first node cannot be connected,
    /// one will be randomly selected from other nodes.
    FIRST_OR_RANDOM,
    /// Round robin from the node configured by zookeeper.
    ROUND_ROBIN,
};

DECLARE_SETTING_ENUM(ZooKeeperLoadBalancing)

}
