#include "CassandraDictionarySource.h"
#include "DictionarySourceFactory.h"
#include "DictionaryStructure.h"
#include "ExternalQueryBuilder.h"
#include <common/logger_useful.h>

namespace DB
{
    namespace ErrorCodes
    {
        extern const int SUPPORT_IS_DISABLED;
    }

    void registerDictionarySourceCassandra(DictionarySourceFactory & factory)
    {
        auto create_table_source = [=](const DictionaryStructure & dict_struct,
                                     const Poco::Util::AbstractConfiguration & config,
                                     const std::string & config_prefix,
                                     Block & sample_block,
                                     const Context & /* context */,
                                     bool /*check_config*/) -> DictionarySourcePtr {
#if USE_CASSANDRA
        return std::make_unique<CassandraDictionarySource>(dict_struct, config, config_prefix + ".cassandra", sample_block);
#else
        (void)dict_struct;
        (void)config;
        (void)config_prefix;
        (void)sample_block;
        throw Exception{"Dictionary source of type `cassandra` is disabled because library was built without cassandra support.",
                        ErrorCodes::SUPPORT_IS_DISABLED};
#endif
        };
        factory.registerSource("cassandra", create_table_source);
    }

}

#if USE_CASSANDRA

#    include <cassandra.h>
#    include <IO/WriteHelpers.h>
#    include "CassandraBlockInputStream.h"

namespace DB
{
namespace ErrorCodes
{
    extern const int UNSUPPORTED_METHOD;
    extern const int WRONG_PASSWORD;
}

static const size_t max_block_size = 8192;

CassandraDictionarySource::CassandraDictionarySource(
    const DB::DictionaryStructure & dict_struct_,
    const String & host_,
    UInt16 port_,
    const String & user_,
    const String & password_,
    //const std::string & method_,
    const String & db_,
    const String & table_,
    const DB::Block & sample_block_)
    : log(&Logger::get("CassandraDictionarySource"))
    , dict_struct(dict_struct_)
    , host(host_)
    , port(port_)
    , user(user_)
    , password(password_)
    //, method(method_)
    , db(db_)
    , table(table_)
    , sample_block(sample_block_)
    , cluster(cass_cluster_new())   //FIXME will not be freed in case of exception
    , session(cass_session_new())
{
    cassandraCheck(cass_cluster_set_contact_points(cluster, host.c_str()));
    if (port)
        cassandraCheck(cass_cluster_set_port(cluster, port));
    cass_cluster_set_credentials(cluster, user.c_str(), password.c_str());
    cassandraWaitAndCheck(cass_session_connect_keyspace(session, cluster, db.c_str()));
}

CassandraDictionarySource::CassandraDictionarySource(
    const DB::DictionaryStructure & dict_struct_,
    const Poco::Util::AbstractConfiguration & config,
    const std::string & config_prefix,
    DB::Block & sample_block_)
    : CassandraDictionarySource(
        dict_struct_,
        config.getString(config_prefix + ".host"),
        config.getUInt(config_prefix + ".port", 0),
        config.getString(config_prefix + ".user", ""),
        config.getString(config_prefix + ".password", ""),
        //config.getString(config_prefix + ".method", ""),
        config.getString(config_prefix + ".keyspace", ""),
        config.getString(config_prefix + ".column_family"),
        sample_block_)
{
}

CassandraDictionarySource::CassandraDictionarySource(const CassandraDictionarySource & other)
    : CassandraDictionarySource{other.dict_struct,
                                other.host,
                                other.port,
                                other.user,
                                other.password,
                                //other.method,
                                other.db,
                                other.table,
                                other.sample_block}
{
}

CassandraDictionarySource::~CassandraDictionarySource() {
    cass_session_free(session);
    cass_cluster_free(cluster);
}

//std::string CassandraDictionarySource::toConnectionString(const std::string &host, const UInt16 port) {
//    return host + (port != 0 ? ":" + std::to_string(port) : "");
//}

BlockInputStreamPtr CassandraDictionarySource::loadAll()
{
    ExternalQueryBuilder builder{dict_struct, db, table, "", IdentifierQuotingStyle::DoubleQuotes};
    String query = builder.composeLoadAllQuery();
    query.pop_back();
    query += " ALLOW FILTERING;";
    LOG_INFO(log, "Loading all using query: " << query);
    return std::make_shared<CassandraBlockInputStream>(session, query, sample_block, max_block_size);
}

std::string CassandraDictionarySource::toString() const {
    return "Cassandra: " + /*db + '.' + collection + ',' + (user.empty() ? " " : " " + user + '@') + */ host + ':' + DB::toString(port);
}

BlockInputStreamPtr CassandraDictionarySource::loadIds(const std::vector<UInt64> & ids)
{
    ExternalQueryBuilder builder{dict_struct, db, table, "", IdentifierQuotingStyle::DoubleQuotes};
    String query = builder.composeLoadIdsQuery(ids);
    query.pop_back();
    query += " ALLOW FILTERING;";
    LOG_INFO(log, "Loading ids using query: " << query);
    return std::make_shared<CassandraBlockInputStream>(session, query, sample_block, max_block_size);
}

BlockInputStreamPtr CassandraDictionarySource::loadKeys(const Columns & key_columns, const std::vector<size_t> & requested_rows)
{
    //FIXME split conditions on partition key and clustering key
    ExternalQueryBuilder builder{dict_struct, db, table, "", IdentifierQuotingStyle::DoubleQuotes};
    String query = builder.composeLoadKeysQuery(key_columns, requested_rows, ExternalQueryBuilder::IN_WITH_TUPLES);
    query.pop_back();
    query += " ALLOW FILTERING;";
    LOG_INFO(log, "Loading keys using query: " << query);
    return std::make_shared<CassandraBlockInputStream>(session, query, sample_block, max_block_size);
}


}

#endif
