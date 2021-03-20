#include "MaterializePostgreSQLConsumer.h"

#if USE_LIBPQXX
#include "StorageMaterializePostgreSQL.h"

#include <Columns/ColumnNullable.h>
#include <Common/hex.h>
#include <DataStreams/copyData.h>
#include <DataStreams/OneBlockInputStream.h>
#include <DataTypes/DataTypeNullable.h>
#include <ext/range.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterInsertQuery.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int UNKNOWN_TABLE;
}

MaterializePostgreSQLConsumer::MaterializePostgreSQLConsumer(
    const Context & context_,
    PostgreSQLConnectionPtr connection_,
    const std::string & replication_slot_name_,
    const std::string & publication_name_,
    const std::string & metadata_path,
    const std::string & start_lsn,
    const size_t max_block_size_,
    bool allow_minimal_ddl_,
    bool is_postgresql_replica_database_engine_,
    Storages storages_)
    : log(&Poco::Logger::get("PostgreSQLReaplicaConsumer"))
    , context(context_)
    , replication_slot_name(replication_slot_name_)
    , publication_name(publication_name_)
    , metadata(metadata_path)
    , connection(std::move(connection_))
    , current_lsn(start_lsn)
    , max_block_size(max_block_size_)
    , allow_minimal_ddl(allow_minimal_ddl_)
    , is_postgresql_replica_database_engine(is_postgresql_replica_database_engine_)
    , storages(storages_)
{
    for (const auto & [table_name, storage] : storages)
    {
        buffers.emplace(table_name, Buffer(storage));
    }
}


void MaterializePostgreSQLConsumer::Buffer::fillBuffer(StoragePtr storage)
{
    const auto storage_metadata = storage->getInMemoryMetadataPtr();
    description.init(storage_metadata->getSampleBlock());

    columns = description.sample_block.cloneEmptyColumns();
    const auto & storage_columns = storage_metadata->getColumns().getAllPhysical();
    auto insert_columns = std::make_shared<ASTExpressionList>();

    assert(description.sample_block.columns() == storage_columns.size());
    size_t idx = 0;

    for (const auto & column : storage_columns)
    {
        if (description.types[idx].first == ExternalResultDescription::ValueType::vtArray)
            preparePostgreSQLArrayInfo(array_info, idx, description.sample_block.getByPosition(idx).type);
        idx++;

        insert_columns->children.emplace_back(std::make_shared<ASTIdentifier>(column.name));
    }

    columnsAST = std::move(insert_columns);
}


void MaterializePostgreSQLConsumer::readMetadata()
{
    try
    {
        metadata.readMetadata();

        if (!metadata.lsn().empty())
        {
            auto tx = std::make_shared<pqxx::nontransaction>(*connection->conn());
            final_lsn = metadata.lsn();
            final_lsn = advanceLSN(tx);
            tx->commit();
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}


void MaterializePostgreSQLConsumer::insertValue(Buffer & buffer, const std::string & value, size_t column_idx)
{
    const auto & sample = buffer.description.sample_block.getByPosition(column_idx);
    bool is_nullable = buffer.description.types[column_idx].second;

    if (is_nullable)
    {
        ColumnNullable & column_nullable = assert_cast<ColumnNullable &>(*buffer.columns[column_idx]);
        const auto & data_type = assert_cast<const DataTypeNullable &>(*sample.type);

        insertPostgreSQLValue(
                column_nullable.getNestedColumn(), value,
                buffer.description.types[column_idx].first, data_type.getNestedType(), buffer.array_info, column_idx);

        column_nullable.getNullMapData().emplace_back(0);
    }
    else
    {
        insertPostgreSQLValue(
                *buffer.columns[column_idx], value,
                buffer.description.types[column_idx].first, sample.type,
                buffer.array_info, column_idx);
    }
}


void MaterializePostgreSQLConsumer::insertDefaultValue(Buffer & buffer, size_t column_idx)
{
    const auto & sample = buffer.description.sample_block.getByPosition(column_idx);
    insertDefaultPostgreSQLValue(*buffer.columns[column_idx], *sample.column);
}


void MaterializePostgreSQLConsumer::readString(const char * message, size_t & pos, size_t size, String & result)
{
    assert(size > pos + 2);
    char current = unhex2(message + pos);
    pos += 2;
    while (pos < size && current != '\0')
    {
        result += current;
        current = unhex2(message + pos);
        pos += 2;
    }
}


Int32 MaterializePostgreSQLConsumer::readInt32(const char * message, size_t & pos, [[maybe_unused]] size_t size)
{
    assert(size > pos + 8);
    Int32 result = (UInt32(unhex2(message + pos)) << 24)
                | (UInt32(unhex2(message + pos + 2)) << 16)
                | (UInt32(unhex2(message + pos + 4)) << 8)
                | (UInt32(unhex2(message + pos + 6)));
    pos += 8;
    return result;
}


Int16 MaterializePostgreSQLConsumer::readInt16(const char * message, size_t & pos, [[maybe_unused]] size_t size)
{
    assert(size > pos + 4);
    Int16 result = (UInt32(unhex2(message + pos)) << 8)
                | (UInt32(unhex2(message + pos + 2)));
    pos += 4;
    return result;
}


Int8 MaterializePostgreSQLConsumer::readInt8(const char * message, size_t & pos, [[maybe_unused]] size_t size)
{
    assert(size > pos + 2);
    Int8 result = unhex2(message + pos);
    pos += 2;
    return result;
}


Int64 MaterializePostgreSQLConsumer::readInt64(const char * message, size_t & pos, [[maybe_unused]] size_t size)
{
    assert(size > pos + 16);
    Int64 result = (UInt64(unhex4(message + pos)) << 48)
                | (UInt64(unhex4(message + pos + 4)) << 32)
                | (UInt64(unhex4(message + pos + 8)) << 16)
                | (UInt64(unhex4(message + pos + 12)));
    pos += 16;
    return result;
}


void MaterializePostgreSQLConsumer::readTupleData(
        Buffer & buffer, const char * message, size_t & pos, [[maybe_unused]] size_t size, PostgreSQLQuery type, bool old_value)
{
    Int16 num_columns = readInt16(message, pos, size);

    auto proccess_column_value = [&](Int8 identifier, Int16 column_idx)
    {
        switch (identifier)
        {
            case 'n': /// NULL
            {
                insertDefaultValue(buffer, column_idx);
                break;
            }
            case 't': /// Text formatted value
            {
                Int32 col_len = readInt32(message, pos, size);
                String value;

                for (Int32 i = 0; i < col_len; ++i)
                    value += readInt8(message, pos, size);

                insertValue(buffer, value, column_idx);
                break;
            }
            case 'u': /// TOAST value && unchanged at the same time. Actual value is not sent.
                /// TOAST values are not supported. (TOAST values are values that are considered in postgres
                /// to be too large to be stored directly)
                insertDefaultValue(buffer, column_idx);
                break;
        }
    };

    for (int column_idx = 0; column_idx < num_columns; ++column_idx)
        proccess_column_value(readInt8(message, pos, size), column_idx);

    switch (type)
    {
        case PostgreSQLQuery::INSERT:
        {
            buffer.columns[num_columns]->insert(Int8(1));
            buffer.columns[num_columns + 1]->insert(UInt64(metadata.version()));

            break;
        }
        case PostgreSQLQuery::DELETE:
        {
            buffer.columns[num_columns]->insert(Int8(-1));
            buffer.columns[num_columns + 1]->insert(UInt64(metadata.version()));

            break;
        }
        case PostgreSQLQuery::UPDATE:
        {
            /// Process old value in case changed value is a primary key.
            if (old_value)
                buffer.columns[num_columns]->insert(Int8(-1));
            else
                buffer.columns[num_columns]->insert(Int8(1));

            buffer.columns[num_columns + 1]->insert(UInt64(metadata.version()));

            break;
        }
    }
}


/// https://www.postgresql.org/docs/13/protocol-logicalrep-message-formats.html
void MaterializePostgreSQLConsumer::processReplicationMessage(const char * replication_message, size_t size)
{
    /// Skip '\x'
    size_t pos = 2;
    char type = readInt8(replication_message, pos, size);

    switch (type)
    {
        case 'B': // Begin
        {
            readInt64(replication_message, pos, size); /// Int64 transaction end lsn
            readInt64(replication_message, pos, size); /// Int64 transaction commit timestamp
            break;
        }
        case 'I': // Insert
        {
            Int32 relation_id = readInt32(replication_message, pos, size);

            if (!isSyncAllowed(relation_id))
                return;

            Int8 new_tuple = readInt8(replication_message, pos, size);
            const auto & table_name = relation_id_to_name[relation_id];
            auto buffer = buffers.find(table_name);

            if (buffer == buffers.end())
                throw Exception(ErrorCodes::UNKNOWN_TABLE, "Buffer for table {} does not exist", table_name);

            if (new_tuple)
                readTupleData(buffer->second, replication_message, pos, size, PostgreSQLQuery::INSERT);

            break;
        }
        case 'U': // Update
        {
            Int32 relation_id = readInt32(replication_message, pos, size);

            if (!isSyncAllowed(relation_id))
                return;

            const auto & table_name = relation_id_to_name[relation_id];
            auto buffer = buffers.find(table_name);

            auto proccess_identifier = [&](Int8 identifier) -> bool
            {
                bool read_next = true;
                switch (identifier)
                {
                    case 'K':
                    {
                        /// Only if changed column(s) are part of replica identity index (for now it can be only
                        /// be primary key - default values for replica identity index). In this case, first comes a tuple
                        /// with old replica identity indexes and all other values will come as nulls. Then comes a full new row.
                        readTupleData(buffer->second, replication_message, pos, size, PostgreSQLQuery::UPDATE, true);
                        break;
                    }
                    case 'O':
                    {
                        /// Old row. Only if replica identity is set to full. (For the case when a table does not have any
                        /// primary key, for now not supported, requires to find suitable order by key(s) for nested table.)
                        readTupleData(buffer->second, replication_message, pos, size, PostgreSQLQuery::UPDATE, true);
                        break;
                    }
                    case 'N':
                    {
                        readTupleData(buffer->second, replication_message, pos, size, PostgreSQLQuery::UPDATE);
                        read_next = false;
                        break;
                    }
                }

                return read_next;
            };

            /// Read either 'K' or 'O'. Never both of them. Also possible not to get both of them.
            bool read_next = proccess_identifier(readInt8(replication_message, pos, size));

            /// 'N'. Always present, but could come in place of 'K' and 'O'.
            if (read_next)
                proccess_identifier(readInt8(replication_message, pos, size));

            break;
        }
        case 'D': // Delete
        {
            Int32 relation_id = readInt32(replication_message, pos, size);

            if (!isSyncAllowed(relation_id))
                return;

             /// 0 or 1 if replica identity is set to full. For now only default replica identity is supported (with primary keys).
            readInt8(replication_message, pos, size);

            const auto & table_name = relation_id_to_name[relation_id];
            auto buffer = buffers.find(table_name);
            readTupleData(buffer->second, replication_message, pos, size, PostgreSQLQuery::DELETE);

            break;
        }
        case 'C': // Commit
        {
            constexpr size_t unused_flags_len = 1;
            constexpr size_t commit_lsn_len = 8;
            constexpr size_t transaction_end_lsn_len = 8;
            constexpr size_t transaction_commit_timestamp_len = 8;
            pos += unused_flags_len + commit_lsn_len + transaction_end_lsn_len + transaction_commit_timestamp_len;

            final_lsn = current_lsn;
            LOG_DEBUG(log, "Commit lsn: {}", getLSNValue(current_lsn)); /// Will be removed

            break;
        }
        case 'R': // Relation
        {
            Int32 relation_id = readInt32(replication_message, pos, size);

            String relation_namespace, relation_name;

            readString(replication_message, pos, size, relation_namespace);
            readString(replication_message, pos, size, relation_name);

            if (!isSyncAllowed(relation_id))
                return;

            /// 'd' - default (primary key if any)
            /// 'n' - nothing
            /// 'f' - all columns (set replica identity full)
            /// 'i' - user defined index with indisreplident set
            /// For database engine now supported only 'd', for table engine 'f' is also allowed.
            char replica_identity = readInt8(replication_message, pos, size);

            if (replica_identity != 'd' && (replica_identity != 'f' || is_postgresql_replica_database_engine))
            {
                LOG_WARNING(log,
                        "Table has replica identity {} - not supported. "
                        "For database engine only default (with primary keys) replica identity is supported."
                        "For table engine full replica identity is also supported. Table will be skipped.");
                markTableAsSkipped(relation_id, relation_name);
                return;
            }

            Int16 num_columns = readInt16(replication_message, pos, size);

            Int32 data_type_id;
            Int32 type_modifier; /// For example, n in varchar(n)

            bool new_relation_definition = false;
            if (schema_data.find(relation_id) == schema_data.end())
            {
                relation_id_to_name[relation_id] = relation_name;
                schema_data.emplace(relation_id, SchemaData(num_columns));
                new_relation_definition = true;
            }

            auto & current_schema_data = schema_data.find(relation_id)->second;

            if (current_schema_data.number_of_columns != num_columns)
            {
                markTableAsSkipped(relation_id, relation_name);
                return;
            }

            for (uint16_t i = 0; i < num_columns; ++i)
            {
                String column_name;
                readInt8(replication_message, pos, size); /// Marks column as part of replica identity index
                readString(replication_message, pos, size, column_name);

                data_type_id = readInt32(replication_message, pos, size);
                type_modifier = readInt32(replication_message, pos, size);

                if (new_relation_definition)
                {
                    current_schema_data.column_identifiers.emplace_back(std::make_tuple(data_type_id, type_modifier));
                }
                else
                {
                    if (current_schema_data.column_identifiers[i].first != data_type_id
                            || current_schema_data.column_identifiers[i].second != type_modifier)
                    {
                        markTableAsSkipped(relation_id, relation_name);
                        return;
                    }
                }
            }

            if (storages.find(relation_name) == storages.end())
            {
                throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Storage for table {} does not exist, but is included in replication stream", relation_name);
            }

            [[maybe_unused]] auto buffer_iter = buffers.find(relation_name);
            assert(buffer_iter != buffers.end());

            tables_to_sync.insert(relation_name);

            break;
        }
        case 'O': // Origin
            break;
        case 'Y': // Type
            break;
        case 'T': // Truncate
            break;
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR,
                    "Unexpected byte1 value {} while parsing replication message", type);
    }
}


void MaterializePostgreSQLConsumer::syncTables(std::shared_ptr<pqxx::nontransaction> tx)
{
    for (const auto & table_name : tables_to_sync)
    {
        try
        {
            auto & buffer = buffers.find(table_name)->second;
            Block result_rows = buffer.description.sample_block.cloneWithColumns(std::move(buffer.columns));

            if (result_rows.rows())
            {
                metadata.commitMetadata(final_lsn, [&]()
                {
                    auto storage = storages[table_name];

                    auto insert = std::make_shared<ASTInsertQuery>();
                    insert->table_id = storage->getStorageID();
                    insert->columns = buffer.columnsAST;

                    auto insert_context(context);
                    insert_context.makeQueryContext();
                    insert_context.addQueryFactoriesInfo(Context::QueryLogFactories::Storage, "ReplacingMergeTree");

                    InterpreterInsertQuery interpreter(insert, insert_context, true);
                    auto block_io = interpreter.execute();
                    OneBlockInputStream input(result_rows);

                    assertBlocksHaveEqualStructure(input.getHeader(), block_io.out->getHeader(), "postgresql replica table sync");
                    copyData(input, *block_io.out);

                    auto actual_lsn = advanceLSN(tx);
                    buffer.columns = buffer.description.sample_block.cloneEmptyColumns();

                    return actual_lsn;
                });
            }
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    LOG_DEBUG(log, "Table sync end for {} tables", tables_to_sync.size());
    tables_to_sync.clear();
    tx->commit();
}


String MaterializePostgreSQLConsumer::advanceLSN(std::shared_ptr<pqxx::nontransaction> tx)
{
    std::string query_str = fmt::format("SELECT end_lsn FROM pg_replication_slot_advance('{}', '{}')", replication_slot_name, final_lsn);
    pqxx::result result{tx->exec(query_str)};

    if (!result.empty())
        return result[0][0].as<std::string>();

    return final_lsn;
}


bool MaterializePostgreSQLConsumer::isSyncAllowed(Int32 relation_id)
{
    auto table_with_lsn = skip_list.find(relation_id);
    if (table_with_lsn == skip_list.end())
        return true;

    const auto & table_start_lsn = table_with_lsn->second;
    if (table_start_lsn.empty())
        return false;

    if (getLSNValue(current_lsn) >= getLSNValue(table_start_lsn))
    {
        LOG_TRACE(log, "Synchronization is resumed for table: {} (start_lsn: {})",
                relation_id_to_name[relation_id], table_start_lsn);

        skip_list.erase(table_with_lsn);

        return true;
    }

    return false;
}


void MaterializePostgreSQLConsumer::markTableAsSkipped(Int32 relation_id, const String & relation_name)
{
    skip_list.insert({relation_id, ""});
    schema_data.erase(relation_id);
    auto & buffer = buffers.find(relation_name)->second;
    buffer.columns = buffer.description.sample_block.cloneEmptyColumns();
    if (!allow_minimal_ddl)
        LOG_WARNING(log, "Table {} is skipped, because table schema has changed", relation_name);
    else
        LOG_TRACE(log, "Table {} is skipped temporarily. ID: {}", relation_name, relation_id);
}


/// Read binary changes from replication slot via COPY command (starting from current lsn in a slot).
bool MaterializePostgreSQLConsumer::readFromReplicationSlot()
{
    std::shared_ptr<pqxx::nontransaction> tx;
    bool slot_empty = true;

    try
    {
        tx = std::make_shared<pqxx::nontransaction>(*connection->conn());

        /// Read up to max_block_size rows changes (upto_n_changes parameter). It might return larger number as the limit
        /// is checked only after each transaction block.
        /// Returns less than max_block_changes, if reached end of wal. Sync to table in this case.

        std::string query_str = fmt::format(
                "select lsn, data FROM pg_logical_slot_peek_binary_changes("
                "'{}', NULL, {}, 'publication_names', '{}', 'proto_version', '1')",
                replication_slot_name, max_block_size, publication_name);

        pqxx::stream_from stream(*tx, pqxx::from_query, std::string_view(query_str));

        while (true)
        {
            const std::vector<pqxx::zview> * row{stream.read_row()};

            if (!row)
            {
                stream.complete();

                if (slot_empty)
                {
                    tx->commit();
                    return false;
                }

                break;
            }

            slot_empty = false;
            current_lsn = (*row)[0];

            processReplicationMessage((*row)[1].c_str(), (*row)[1].size());
        }
    }
    catch (const pqxx::sql_error & e)
    {
        /// For now sql replication interface is used and it has the problem that it registers relcache
        /// callbacks on each pg_logical_slot_get_changes and there is no way to invalidate them:
        /// https://github.com/postgres/postgres/blob/master/src/backend/replication/pgoutput/pgoutput.c#L1128
        /// So at some point will get out of limit and then they will be cleaned.
        std::string error_message = e.what();
        if (error_message.find("out of relcache_callback_list slots") == std::string::npos)
            tryLogCurrentException(__PRETTY_FUNCTION__);

        return false;
    }
    catch (const Exception & e)
    {
        if (e.code() == ErrorCodes::UNKNOWN_TABLE)
            throw;

        tryLogCurrentException(__PRETTY_FUNCTION__);
        return false;
    }

    if (!tables_to_sync.empty())
    {
        syncTables(tx);
    }

    return true;
}


bool MaterializePostgreSQLConsumer::consume(std::vector<std::pair<Int32, String>> & skipped_tables)
{
    if (!readFromReplicationSlot())
    {
        if (allow_minimal_ddl && !skip_list.empty())
        {
            for (const auto & [relation_id, lsn] : skip_list)
            {
                if (lsn.empty())
                    skipped_tables.emplace_back(std::make_pair(relation_id, relation_id_to_name[relation_id]));
            }
        }

        return false;
    }

    return true;
}


void MaterializePostgreSQLConsumer::updateNested(const String & table_name, StoragePtr nested_storage)
{
    storages[table_name] = nested_storage;
    auto & buffer = buffers.find(table_name)->second;
    buffer.fillBuffer(nested_storage);
}


void MaterializePostgreSQLConsumer::updateSkipList(const std::unordered_map<Int32, String> & tables_with_lsn)
{
    for (const auto & [relation_id, lsn] : tables_with_lsn)
    {
        if (!lsn.empty())
            skip_list[relation_id] = lsn; /// start_lsn
    }
}

}

#endif
