#include "RedisCommon.h"
#include <Common/Exception.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int INVALID_REDIS_TABLE_STRUCTURE;
    extern const int INTERNAL_REDIS_ERROR;
    extern const int TIMEOUT_EXCEEDED;
    extern const int INVALID_REDIS_STORAGE_TYPE;
}

RedisColumnTypes REDIS_HASH_MAP_COLUMN_TYPES = {RedisColumnType::KEY, RedisColumnType::FIELD, RedisColumnType::VALUE};
RedisColumnTypes REDIS_SIMPLE_COLUMN_TYPES = {RedisColumnType::KEY, RedisColumnType::VALUE};

RedisConnection::RedisConnection(RedisPoolPtr pool_, RedisClientPtr client_)
    : pool(std::move(pool_)), client(std::move(client_))
{
}

RedisConnection::~RedisConnection()
{
    pool->returnObject(std::move(client));
}

String serializeStorageType(RedisStorageType storage_type)
{
    switch (storage_type)
    {
        case RedisStorageType::SIMPLE:
            return "simple";
        case RedisStorageType::HASH_MAP:
            return "hash_map";
        default:
            return "none";
    }
}

RedisStorageType parseStorageType(const String & storage_type_str)
{
    if (storage_type_str == "hash_map")
        return RedisStorageType::HASH_MAP;
    else if (!storage_type_str.empty() && storage_type_str != "simple")
        throw Exception(ErrorCodes::INVALID_REDIS_STORAGE_TYPE, "Unknown storage type {} for Redis dictionary", storage_type_str);

    return RedisStorageType::SIMPLE;
}

RedisConnectionPtr getRedisConnection(RedisPoolPtr pool, const RedisConfiguration & configuration)
{
    RedisClientPtr client;
    bool ok = pool->tryBorrowObject(client,
        [] { return std::make_unique<Poco::Redis::Client>(); },
        REDIS_LOCK_ACQUIRE_TIMEOUT_MS);

    if (!ok)
        throw Exception(ErrorCodes::TIMEOUT_EXCEEDED,
                        "Could not get connection from pool, timeout exceeded {} seconds",
                        REDIS_LOCK_ACQUIRE_TIMEOUT_MS);

    if (!client->isConnected())
    {
        try
        {
            client->connect(configuration.host, configuration.port);

            if (!configuration.password.empty())
            {
                RedisCommand command("AUTH");
                command << configuration.password;
                String reply = client->execute<String>(command);
                if (reply != "OK")
                    throw Exception(ErrorCodes::INTERNAL_REDIS_ERROR,
                                    "Authentication failed with reason {}", reply);
            }

            if (configuration.db_index != 0)
            {
                RedisCommand command("SELECT");
                command << std::to_string(configuration.db_index);
                String reply = client->execute<String>(command);
                if (reply != "OK")
                    throw Exception(ErrorCodes::INTERNAL_REDIS_ERROR,
                                    "Selecting database with index {} failed with reason {}",
                                    configuration.db_index, reply);
            }
        }
        catch (...)
        {
            if (client->isConnected())
                client->disconnect();

            pool->returnObject(std::move(client));
            throw;
        }
    }

    return std::make_unique<RedisConnection>(pool, std::move(client));
}


RedisArrayPtr getRedisHashMapKeys(const RedisConnectionPtr & connection, RedisArray & keys)
{
    RedisArrayPtr hkeys = std::make_shared<RedisArray>();
    for (const auto & key : keys)
    {
        RedisCommand command_for_secondary_keys("HKEYS");
        command_for_secondary_keys.addRedisType(key);

        auto secondary_keys = connection->client->execute<RedisArray>(command_for_secondary_keys);
        if (secondary_keys.isNull())
            continue;

        RedisArray primary_with_secondary;
        primary_with_secondary.addRedisType(key);
        for (const auto & secondary_key : secondary_keys)
        {
            primary_with_secondary.addRedisType(secondary_key);
            /// Do not store more than max_block_size values for one request.
            if (primary_with_secondary.size() == REDIS_MAX_BLOCK_SIZE + 1)
            {
                hkeys->add(primary_with_secondary);
                primary_with_secondary.clear();
                primary_with_secondary.addRedisType(key);
            }
        }

        if (primary_with_secondary.size() > 1)
            hkeys->add(primary_with_secondary);
    }

    return hkeys;
}

RedisColumnType getRedisColumnType(RedisStorageType storage_type, const Names & all_columns, const String & column)
{
    String redis_col_key = all_columns.at(0);
    if (column == redis_col_key)
        return RedisColumnType::KEY;

    if (storage_type == RedisStorageType::HASH_MAP)
    {
        String redis_col_field = all_columns.at(1);
        if (column == redis_col_field)
            return RedisColumnType::FIELD;
        else
            return RedisColumnType::VALUE;
    }
    else
    {
        return RedisColumnType::VALUE;
    }
}

void checkRedisTableStructure(const ColumnsDescription & columns, const RedisConfiguration & configuration)
{
    /// TODO check data type
    if (configuration.storage_type == RedisStorageType::HASH_MAP && columns.size() != 3)
        throw Exception(ErrorCodes::INVALID_REDIS_TABLE_STRUCTURE,
                        "Redis hash table must have 3 columns, but found {}", columns.size());

    if (configuration.storage_type == RedisStorageType::SIMPLE && columns.size() != 2)
        throw Exception(ErrorCodes::INVALID_REDIS_TABLE_STRUCTURE,
                        "Redis string table must have 2 columns, but found {}", columns.size());
}

}
