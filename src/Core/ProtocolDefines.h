#pragma once

#define DBMS_MIN_REVISION_WITH_CLIENT_INFO 54032
#define DBMS_MIN_REVISION_WITH_SERVER_TIMEZONE 54058
#define DBMS_MIN_REVISION_WITH_QUOTA_KEY_IN_CLIENT_INFO 54060
#define DBMS_MIN_REVISION_WITH_TABLES_STATUS 54226
#define DBMS_MIN_REVISION_WITH_TIME_ZONE_PARAMETER_IN_DATETIME_DATA_TYPE 54337
#define DBMS_MIN_REVISION_WITH_SERVER_DISPLAY_NAME 54372
#define DBMS_MIN_REVISION_WITH_VERSION_PATCH 54401
#define DBMS_MIN_REVISION_WITH_SERVER_LOGS 54406
/// Minimum revision with exactly the same set of aggregation methods and rules to select them.
/// Two-level (bucketed) aggregation is incompatible if servers are inconsistent in these rules
/// (keys will be placed in different buckets and result will not be fully aggregated).
#define DBMS_MIN_REVISION_WITH_CURRENT_AGGREGATION_VARIANT_SELECTION_METHOD 54448
#define DBMS_MIN_MAJOR_VERSION_WITH_CURRENT_AGGREGATION_VARIANT_SELECTION_METHOD 21
#define DBMS_MIN_MINOR_VERSION_WITH_CURRENT_AGGREGATION_VARIANT_SELECTION_METHOD 4
#define DBMS_MIN_REVISION_WITH_COLUMN_DEFAULTS_METADATA 54410

#define DBMS_MIN_REVISION_WITH_LOW_CARDINALITY_TYPE 54405
#define DBMS_MIN_REVISION_WITH_CLIENT_WRITE_INFO 54420

/// Minimum revision supporting SettingsBinaryFormat::STRINGS.
#define DBMS_MIN_REVISION_WITH_SETTINGS_SERIALIZED_AS_STRINGS 54429
#define DBMS_MIN_REVISION_WITH_SCALARS 54429

/// Minimum revision supporting OpenTelemetry
#define DBMS_MIN_REVISION_WITH_OPENTELEMETRY 54442

#define DBMS_MIN_REVISION_WITH_AGGREGATE_FUNCTIONS_VERSIONING 54452

#define DBMS_CLUSTER_PROCESSING_PROTOCOL_VERSION 1

#define DBMS_PARALLEL_REPLICAS_PROTOCOL_VERSION 1
#define DBMS_MIN_REVISION_WITH_PARALLEL_REPLICAS 54453

/// Minimum revision supporting interserver secret.
#define DBMS_MIN_REVISION_WITH_INTERSERVER_SECRET 54441

#define DBMS_MIN_REVISION_WITH_X_FORWARDED_FOR_IN_CLIENT_INFO 54443
#define DBMS_MIN_REVISION_WITH_REFERER_IN_CLIENT_INFO 54447

#define DBMS_MIN_PROTOCOL_VERSION_WITH_DISTRIBUTED_DEPTH 54448

#define DBMS_MIN_PROTOCOL_VERSION_WITH_INCREMENTAL_PROFILE_EVENTS 54451

#define DBMS_MIN_REVISION_WITH_CUSTOM_SERIALIZATION 54454

/// Version of ClickHouse TCP protocol.
///
/// Should be incremented manually on protocol changes.
///
/// NOTE: DBMS_TCP_PROTOCOL_VERSION has nothing common with VERSION_REVISION,
/// later is just a number for server version (one number instead of commit SHA)
/// for simplicity (sometimes it may be more convenient in some use cases).
#define DBMS_TCP_PROTOCOL_VERSION 54458

#define DBMS_MIN_PROTOCOL_VERSION_WITH_INITIAL_QUERY_START_TIME 54449

#define DBMS_MIN_PROTOCOL_VERSION_WITH_PROFILE_EVENTS_IN_INSERT 54456

#define DBMS_MIN_PROTOCOL_VERSION_WITH_VIEW_IF_PERMITTED 54457

/// The server will send query elapsed run time in the Progress packet.
#define DBMS_MIN_PROTOCOL_VERSION_WITH_SERVER_QUERY_TIME_IN_PROGRESS 54458
