#pragma once

#if !defined(ARCADIA_BUILD)
#include "config_core.h"
#endif

#if USE_LIBPQXX

#include <Storages/PostgreSQL/PostgreSQLReplicationHandler.h>
#include <Storages/PostgreSQL/MaterializedPostgreSQLSettings.h>

#include <Databases/DatabasesCommon.h>
#include <Core/BackgroundSchedulePool.h>
#include <Parsers/ASTCreateQuery.h>
#include <Databases/IDatabase.h>
#include <Databases/DatabaseOnDisk.h>
#include <Databases/DatabaseAtomic.h>


namespace DB
{

class PostgreSQLConnection;
using PostgreSQLConnectionPtr = std::shared_ptr<PostgreSQLConnection>;


class DatabaseMaterializedPostgreSQL : public DatabaseAtomic
{

public:
    DatabaseMaterializedPostgreSQL(
        ContextPtr context_,
        const String & metadata_path_,
        UUID uuid_,
        ASTPtr storage_def_,
        bool is_attach_,
        const String & database_name_,
        const String & postgres_database_name,
        const postgres::ConnectionInfo & connection_info,
        std::unique_ptr<MaterializedPostgreSQLSettings> settings_);

    String getEngineName() const override { return "MaterializedPostgreSQL"; }

    String getMetadataPath() const override { return metadata_path; }

    void loadStoredObjects(ContextMutablePtr, bool, bool force_attach) override;

    DatabaseTablesIteratorPtr getTablesIterator(
            ContextPtr context, const DatabaseOnDisk::FilterByNameFunction & filter_by_table_name) const override;

    StoragePtr tryGetTable(const String & name, ContextPtr context) const override;

    void createTable(ContextPtr context, const String & name, const StoragePtr & table, const ASTPtr & query) override;

    void attachTable(const String & name, const StoragePtr & table, const String & relative_table_path) override;

    void dropTable(ContextPtr local_context, const String & name, bool no_delay) override;

    void drop(ContextPtr local_context) override;

    void stopReplication();

    void checkAlterIsPossible(const AlterCommands & commands, ContextPtr context) const override;

    void applySettings(const SettingsChanges & settings_changes, ContextPtr context) override;

    void shutdown() override;

    String getPostgreSQLDatabaseName() const { return remote_database_name; }

protected:
    ASTPtr getCreateTableQueryImpl(const String & table_name, ContextPtr local_context, bool throw_on_error) const override;

private:
    void startSynchronization();

    String getTablesList() const;

    bool is_attach;
    String remote_database_name;
    postgres::ConnectionInfo connection_info;
    std::unique_ptr<MaterializedPostgreSQLSettings> settings;

    std::shared_ptr<PostgreSQLReplicationHandler> replication_handler;
    std::map<std::string, StoragePtr> materialized_tables;
    mutable std::mutex tables_mutex;
};

}

#endif
