#include <Core/SettingsEnums.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int UNKNOWN_LOAD_BALANCING;
    extern const int UNKNOWN_OVERFLOW_MODE;
    extern const int UNKNOWN_TOTALS_MODE;
    extern const int UNKNOWN_DISTRIBUTED_PRODUCT_MODE;
    extern const int UNKNOWN_JOIN;
    extern const int BAD_ARGUMENTS;
    extern const int UNKNOWN_MYSQL_DATATYPES_SUPPORT_LEVEL;
    extern const int UNKNOWN_UNION;
}


IMPLEMENT_SETTING_ENUM(LoadBalancing, ErrorCodes::UNKNOWN_LOAD_BALANCING,
    {{"random",           LoadBalancing::RANDOM},
     {"nearest_hostname", LoadBalancing::NEAREST_HOSTNAME},
     {"in_order",         LoadBalancing::IN_ORDER},
     {"first_or_random",  LoadBalancing::FIRST_OR_RANDOM},
     {"round_robin",      LoadBalancing::ROUND_ROBIN}})


IMPLEMENT_SETTING_ENUM(JoinStrictness, ErrorCodes::UNKNOWN_JOIN,
    {{"",    JoinStrictness::Unspecified},
     {"ALL", JoinStrictness::ALL},
     {"ANY", JoinStrictness::ANY}})


IMPLEMENT_SETTING_ENUM(JoinAlgorithm, ErrorCodes::UNKNOWN_JOIN,
    {{"auto",                 JoinAlgorithm::AUTO},
     {"hash",                 JoinAlgorithm::HASH},
     {"partial_merge",        JoinAlgorithm::PARTIAL_MERGE},
     {"prefer_partial_merge", JoinAlgorithm::PREFER_PARTIAL_MERGE}})


IMPLEMENT_SETTING_ENUM(TotalsMode, ErrorCodes::UNKNOWN_TOTALS_MODE,
    {{"before_having",          TotalsMode::BEFORE_HAVING},
     {"after_having_exclusive", TotalsMode::AFTER_HAVING_EXCLUSIVE},
     {"after_having_inclusive", TotalsMode::AFTER_HAVING_INCLUSIVE},
     {"after_having_auto",      TotalsMode::AFTER_HAVING_AUTO}})


IMPLEMENT_SETTING_ENUM(OverflowMode, ErrorCodes::UNKNOWN_OVERFLOW_MODE,
    {{"throw", OverflowMode::THROW},
     {"break", OverflowMode::BREAK}})


IMPLEMENT_SETTING_ENUM_WITH_RENAME(OverflowModeGroupBy, ErrorCodes::UNKNOWN_OVERFLOW_MODE,
    {{"throw", OverflowMode::THROW},
     {"break", OverflowMode::BREAK},
     {"any", OverflowMode::ANY}})


IMPLEMENT_SETTING_ENUM(DistributedProductMode, ErrorCodes::UNKNOWN_DISTRIBUTED_PRODUCT_MODE,
    {{"deny",   DistributedProductMode::DENY},
     {"local",  DistributedProductMode::LOCAL},
     {"global", DistributedProductMode::GLOBAL},
     {"allow",  DistributedProductMode::ALLOW}})


IMPLEMENT_SETTING_ENUM_WITH_RENAME(DateTimeInputFormat, ErrorCodes::BAD_ARGUMENTS,
    {{"basic",       FormatSettings::DateTimeInputFormat::Basic},
     {"best_effort", FormatSettings::DateTimeInputFormat::BestEffort}})


IMPLEMENT_SETTING_ENUM_WITH_RENAME(DateTimeOutputFormat, ErrorCodes::BAD_ARGUMENTS,
    {{"simple",         FormatSettings::DateTimeOutputFormat::Simple},
     {"iso",            FormatSettings::DateTimeOutputFormat::ISO},
     {"unix_timestamp", FormatSettings::DateTimeOutputFormat::UnixTimestamp}})

IMPLEMENT_SETTING_ENUM(LogsLevel, ErrorCodes::BAD_ARGUMENTS,
    {{"none",        LogsLevel::none},
     {"fatal",       LogsLevel::fatal},
     {"error",       LogsLevel::error},
     {"warning",     LogsLevel::warning},
     {"information", LogsLevel::information},
     {"debug",       LogsLevel::debug},
     {"trace",       LogsLevel::trace},
     {"test",        LogsLevel::test}})

IMPLEMENT_SETTING_ENUM_WITH_RENAME(LogQueriesType, ErrorCodes::BAD_ARGUMENTS,
    {{"QUERY_START",                QUERY_START},
     {"QUERY_FINISH",               QUERY_FINISH},
     {"EXCEPTION_BEFORE_START",     EXCEPTION_BEFORE_START},
     {"EXCEPTION_WHILE_PROCESSING", EXCEPTION_WHILE_PROCESSING}})


IMPLEMENT_SETTING_ENUM_WITH_RENAME(DefaultDatabaseEngine, ErrorCodes::BAD_ARGUMENTS,
    {{"Ordinary", DefaultDatabaseEngine::Ordinary},
     {"Atomic",   DefaultDatabaseEngine::Atomic}})

IMPLEMENT_SETTING_MULTI_ENUM(MySQLDataTypesSupport, ErrorCodes::UNKNOWN_MYSQL_DATATYPES_SUPPORT_LEVEL,
    {{"decimal",    MySQLDataTypesSupport::DECIMAL},
     {"datetime64", MySQLDataTypesSupport::DATETIME64}})

IMPLEMENT_SETTING_ENUM(UnionMode, ErrorCodes::UNKNOWN_UNION,
    {{"",         UnionMode::Unspecified},
     {"ALL",      UnionMode::ALL},
     {"DISTINCT", UnionMode::DISTINCT}})

IMPLEMENT_SETTING_ENUM(DistributedDDLOutputMode, ErrorCodes::BAD_ARGUMENTS,
    {{"none",         DistributedDDLOutputMode::NONE},
     {"throw",    DistributedDDLOutputMode::THROW},
     {"null_status_on_timeout", DistributedDDLOutputMode::NULL_STATUS_ON_TIMEOUT},
     {"never_throw", DistributedDDLOutputMode::NEVER_THROW}})

IMPLEMENT_SETTING_ENUM(HandleKafkaErrorMode, ErrorCodes::BAD_ARGUMENTS,
    {{"default",      HandleKafkaErrorMode::DEFAULT},
     {"stream",       HandleKafkaErrorMode::STREAM}})

IMPLEMENT_SETTING_ENUM(ShortCircuitFunctionEvaluation, ErrorCodes::BAD_ARGUMENTS,
    {{"enable",          ShortCircuitFunctionEvaluation::ENABLE},
     {"force_enable",    ShortCircuitFunctionEvaluation::FORCE_ENABLE},
     {"disable",         ShortCircuitFunctionEvaluation::DISABLE}})

IMPLEMENT_SETTING_ENUM(EnumComparingMode, ErrorCodes::BAD_ARGUMENTS,
    {{"by_names",   FormatSettings::EnumComparingMode::BY_NAMES},
     {"by_values",  FormatSettings::EnumComparingMode::BY_VALUES},
     {"by_names_case_insensitive", FormatSettings::EnumComparingMode::BY_NAMES_CASE_INSENSITIVE}})

IMPLEMENT_SETTING_ENUM(BoolFormat, ErrorCodes::BAD_ARGUMENTS,
    {{"true_false_lower_case",  FormatSettings::BoolFormat::true_false_lower_case},
     {"true_false_camel_case",  FormatSettings::BoolFormat::true_false_camel_case},
     {"T_F",                    FormatSettings::BoolFormat::T_F},
     {"Yes_No",                 FormatSettings::BoolFormat::Yes_No},
     {"Y_N",                    FormatSettings::BoolFormat::Y_N},
     {"On_Off",                 FormatSettings::BoolFormat::On_Off}})

}
