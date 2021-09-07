#pragma once

#include <Parsers/ASTIdentifier.h>
#include <Interpreters/Context.h>
#include <Poco/Util/AbstractConfiguration.h>


namespace DB
{

struct ExternalDataSourceConfiguration
{
    String host;
    UInt16 port = 0;
    String username;
    String password;
    String database;
    String table;
    String schema;

    std::vector<std::pair<String, UInt16>> addresses; /// Failover replicas.

    String toString() const;
};

using ExternalDataSourceConfigurationPtr = std::shared_ptr<ExternalDataSourceConfiguration>;


struct StoragePostgreSQLConfiguration : ExternalDataSourceConfiguration
{
    explicit StoragePostgreSQLConfiguration(const ExternalDataSourceConfiguration & common_configuration)
        : ExternalDataSourceConfiguration(common_configuration) {}

    String on_conflict;
};


struct StorageMySQLConfiguration : ExternalDataSourceConfiguration
{
    explicit StorageMySQLConfiguration(const ExternalDataSourceConfiguration & common_configuration)
        : ExternalDataSourceConfiguration(common_configuration) {}

    bool replace_query = false;
    String on_duplicate_clause;
};

struct StorageMongoDBConfiguration : ExternalDataSourceConfiguration
{
    explicit StorageMongoDBConfiguration(const ExternalDataSourceConfiguration & common_configuration)
        : ExternalDataSourceConfiguration(common_configuration) {}

    String collection;
    String options;
};


using EngineArgs = std::vector<std::pair<String, DB::Field>>;

/* If storage engine's configuration was define via named_collections,
 * return all options in ExternalDataSource::Configuration struct.
 *
 * Also check if engine arguments have key-value defined configuration options:
 * ENGINE = PostgreSQL(postgresql_configuration, database = 'postgres_database');
 * In this case they will override values defined in config.
 *
 * If there are key-value arguments apart from common: `host`, `port`, `username`, `password`, `database`,
 * i.e. storage-specific arguments, then return them back in a set: ExternalDataSource::EngineArgs.
 */
std::tuple<ExternalDataSourceConfiguration, EngineArgs, bool>
getExternalDataSourceConfiguration(const ASTs & args, ContextPtr context, bool is_database_engine = false);

ExternalDataSourceConfiguration getExternalDataSourceConfiguration(
    const Poco::Util::AbstractConfiguration & dict_config, const String & dict_config_prefix, ContextPtr context);


/// Highest priority is 0, the bigger the number in map, the less the priority.
using ExternalDataSourcesConfigurationByPriority = std::map<size_t, std::vector<ExternalDataSourceConfiguration>>;

struct ExternalDataSourcesByPriority
{
    String database;
    String table;
    String schema;
    ExternalDataSourcesConfigurationByPriority replicas_configurations;
};

ExternalDataSourcesByPriority
getExternalDataSourceConfigurationByPriority(const Poco::Util::AbstractConfiguration & dict_config, const String & dict_config_prefix, ContextPtr context);


struct URLBasedDataSourceConfiguration
{
    String url;
    String format;
    String compression_method = "auto";
    String structure;

    std::vector<std::pair<String, Field>> headers;
};

struct StorageS3Configuration : URLBasedDataSourceConfiguration
{
    explicit StorageS3Configuration(const URLBasedDataSourceConfiguration & common_configuration)
        : URLBasedDataSourceConfiguration(common_configuration) {}

    String access_key_id;
    String secret_access_key;
};

std::tuple<URLBasedDataSourceConfiguration, EngineArgs, bool>
getURLBasedDataSourceConfiguration(const ASTs & args, ContextPtr context);

}
