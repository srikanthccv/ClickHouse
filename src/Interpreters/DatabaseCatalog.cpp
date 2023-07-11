#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/Context.h>
#include <Interpreters/loadMetadata.h>
#include <Interpreters/executeQuery.h>
#include <Interpreters/InterpreterCreateQuery.h>
#include <Storages/IStorage.h>
#include <Databases/IDatabase.h>
#include <Databases/DatabaseMemory.h>
#include <Databases/DatabaseOnDisk.h>
#include <Disks/IDisk.h>
#include <Storages/StorageMemory.h>
#include <Core/BackgroundSchedulePool.h>
#include <Parsers/formatAST.h>
#include <IO/ReadHelpers.h>
#include <Poco/DirectoryIterator.h>
#include <Poco/Util/AbstractConfiguration.h>
#include <Common/quoteString.h>
#include <Common/atomicRename.h>
#include <Common/CurrentMetrics.h>
#include <Common/logger_useful.h>
#include <Common/ThreadPool.h>
#include <Common/filesystemHelpers.h>
#include <Common/noexcept_scope.h>
#include <Common/checkStackSize.h>

#include "config.h"

#if USE_MYSQL
#    include <Databases/MySQL/MaterializedMySQLSyncThread.h>
#    include <Storages/StorageMaterializedMySQL.h>
#endif

#if USE_LIBPQXX
#    include <Databases/PostgreSQL/DatabaseMaterializedPostgreSQL.h>
#    include <Storages/PostgreSQL/StorageMaterializedPostgreSQL.h>
#endif


namespace CurrentMetrics
{
    extern const Metric TablesToDropQueueSize;
    extern const Metric DatabaseCatalogThreads;
    extern const Metric DatabaseCatalogThreadsActive;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int UNKNOWN_DATABASE;
    extern const int UNKNOWN_TABLE;
    extern const int TABLE_ALREADY_EXISTS;
    extern const int DATABASE_ALREADY_EXISTS;
    extern const int DATABASE_NOT_EMPTY;
    extern const int DATABASE_ACCESS_DENIED;
    extern const int LOGICAL_ERROR;
    extern const int HAVE_DEPENDENT_OBJECTS;
    extern const int UNFINISHED;
}

TemporaryTableHolder::TemporaryTableHolder(ContextPtr context_, const TemporaryTableHolder::Creator & creator, const ASTPtr & query)
    : WithContext(context_->getGlobalContext())
    , temporary_tables(DatabaseCatalog::instance().getDatabaseForTemporaryTables().get())
{
    ASTPtr original_create;
    ASTCreateQuery * create = dynamic_cast<ASTCreateQuery *>(query.get());
    String global_name;
    if (create)
    {
        original_create = create->clone();
        if (create->uuid == UUIDHelpers::Nil)
            create->uuid = UUIDHelpers::generateV4();
        id = create->uuid;
        create->setTable("_tmp_" + toString(id));
        global_name = create->getTable();
        create->setDatabase(DatabaseCatalog::TEMPORARY_DATABASE);
    }
    else
    {
        id = UUIDHelpers::generateV4();
        global_name = "_tmp_" + toString(id);
    }
    auto table_id = StorageID(DatabaseCatalog::TEMPORARY_DATABASE, global_name, id);
    auto table = creator(table_id);
    DatabaseCatalog::instance().addUUIDMapping(id);
    temporary_tables->createTable(getContext(), global_name, table, original_create);
    table->startup();
}


TemporaryTableHolder::TemporaryTableHolder(
    ContextPtr context_,
    const ColumnsDescription & columns,
    const ConstraintsDescription & constraints,
    const ASTPtr & query,
    bool create_for_global_subquery)
    : TemporaryTableHolder(
        context_,
        [&](const StorageID & table_id)
        {
            auto storage = std::make_shared<StorageMemory>(table_id, ColumnsDescription{columns}, ConstraintsDescription{constraints}, String{});

            if (create_for_global_subquery)
                storage->delayReadForGlobalSubqueries();

            return storage;
        },
        query)
{
}

TemporaryTableHolder::TemporaryTableHolder(TemporaryTableHolder && rhs) noexcept
        : WithContext(rhs.context), temporary_tables(rhs.temporary_tables), id(rhs.id), future_set(std::move(rhs.future_set))
{
    rhs.id = UUIDHelpers::Nil;
}

TemporaryTableHolder & TemporaryTableHolder::operator=(TemporaryTableHolder && rhs) noexcept
{
    id = rhs.id;
    rhs.id = UUIDHelpers::Nil;
    return *this;
}

TemporaryTableHolder::~TemporaryTableHolder()
{
    if (id != UUIDHelpers::Nil)
    {
        try
        {
            auto table = getTable();
            table->flushAndShutdown();
            temporary_tables->dropTable(getContext(), "_tmp_" + toString(id));
        }
        catch (...)
        {
            tryLogCurrentException("TemporaryTableHolder");
        }
    }
}

StorageID TemporaryTableHolder::getGlobalTableID() const
{
    return StorageID{DatabaseCatalog::TEMPORARY_DATABASE, "_tmp_" + toString(id), id};
}

StoragePtr TemporaryTableHolder::getTable() const
{
    auto table = temporary_tables->tryGetTable("_tmp_" + toString(id), getContext());
    if (!table)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Temporary table {} not found", getGlobalTableID().getNameForLogs());
    return table;
}

void DatabaseCatalog::initializeAndLoadTemporaryDatabase()
{
    drop_delay_sec = getContext()->getConfigRef().getInt("database_atomic_delay_before_drop_table_sec", default_drop_delay_sec);
    unused_dir_hide_timeout_sec = getContext()->getConfigRef().getInt64("database_catalog_unused_dir_hide_timeout_sec", unused_dir_hide_timeout_sec);
    unused_dir_rm_timeout_sec = getContext()->getConfigRef().getInt64("database_catalog_unused_dir_rm_timeout_sec", unused_dir_rm_timeout_sec);
    unused_dir_cleanup_period_sec = getContext()->getConfigRef().getInt64("database_catalog_unused_dir_cleanup_period_sec", unused_dir_cleanup_period_sec);
    drop_error_cooldown_sec = getContext()->getConfigRef().getInt64("database_catalog_drop_error_cooldown_sec", drop_error_cooldown_sec);

    auto db_for_temporary_and_external_tables = std::make_shared<DatabaseMemory>(TEMPORARY_DATABASE, getContext());
    attachDatabase(TEMPORARY_DATABASE, db_for_temporary_and_external_tables);
}

void DatabaseCatalog::createBackgroundTasks()
{
    /// It has to be done before databases are loaded (to avoid a race condition on initialization)
    if (Context::getGlobalContextInstance()->getApplicationType() == Context::ApplicationType::SERVER && unused_dir_cleanup_period_sec)
    {
        auto cleanup_task_holder
            = getContext()->getSchedulePool().createTask("DatabaseCatalog", [this]() { this->cleanupStoreDirectoryTask(); });
        cleanup_task = std::make_unique<BackgroundSchedulePoolTaskHolder>(std::move(cleanup_task_holder));
    }

    auto task_holder = getContext()->getSchedulePool().createTask("DatabaseCatalog", [this](){ this->dropTableDataTask(); });
    drop_task = std::make_unique<BackgroundSchedulePoolTaskHolder>(std::move(task_holder));
}

void DatabaseCatalog::startupBackgroundCleanup()
{
    /// And it has to be done after all databases are loaded, otherwise cleanup_task may remove something that should not be removed
    if (cleanup_task)
    {
        (*cleanup_task)->activate();
        /// Do not start task immediately on server startup, it's not urgent.
        (*cleanup_task)->scheduleAfter(unused_dir_hide_timeout_sec * 1000);
    }

    (*drop_task)->activate();
    std::lock_guard lock{tables_marked_dropped_mutex};
    if (!tables_marked_dropped.empty())
        (*drop_task)->schedule();
}

void DatabaseCatalog::shutdownImpl()
{
    is_shutting_down = true;
    wait_table_finally_dropped.notify_all();

    if (cleanup_task)
        (*cleanup_task)->deactivate();

    if (drop_task)
        (*drop_task)->deactivate();

    /** At this point, some tables may have threads that block our mutex.
      * To shutdown them correctly, we will copy the current list of tables,
      *  and ask them all to finish their work.
      * Then delete all objects with tables.
      */

    Databases current_databases;
    {
        std::lock_guard lock(databases_mutex);
        current_databases = databases;
    }

    /// We still hold "databases" (instead of std::move) for Buffer tables to flush data correctly.

    /// Delay shutdown of temporary and system databases. They will be shutdown last.
    /// Because some databases might use them until their shutdown is called, but calling shutdown
    /// on temporary database means clearing its set of tables, which will lead to unnecessary errors like "table not found".
    std::vector<DatabasePtr> databases_with_delayed_shutdown;
    for (auto & database : current_databases)
    {
        if (database.first == TEMPORARY_DATABASE || database.first == SYSTEM_DATABASE)
        {
            databases_with_delayed_shutdown.push_back(database.second);
            continue;
        }
        LOG_TRACE(log, "Shutting down database {}", database.first);
        database.second->shutdown();
    }

    LOG_TRACE(log, "Shutting down system databases");
    for (auto & database : databases_with_delayed_shutdown)
    {
        database->shutdown();
    }

    {
        std::lock_guard lock(tables_marked_dropped_mutex);
        tables_marked_dropped.clear();
    }

    std::lock_guard lock(databases_mutex);
    for (const auto & db : databases)
    {
        UUID db_uuid = db.second->getUUID();
        if (db_uuid != UUIDHelpers::Nil)
            removeUUIDMapping(db_uuid);
    }
    assert(std::find_if(uuid_map.begin(), uuid_map.end(), [](const auto & elem)
    {
        /// Ensure that all UUID mappings are empty (i.e. all mappings contain nullptr instead of a pointer to storage)
        const auto & not_empty_mapping = [] (const auto & mapping)
        {
            auto & db = mapping.second.first;
            auto & table = mapping.second.second;
            return db || table;
        };
        std::lock_guard map_lock{elem.mutex};
        auto it = std::find_if(elem.map.begin(), elem.map.end(), not_empty_mapping);
        return it != elem.map.end();
    }) == uuid_map.end());
    databases.clear();
    referential_dependencies.clear();
    loading_dependencies.clear();
    view_dependencies.clear();
}

bool DatabaseCatalog::isPredefinedDatabase(std::string_view database_name)
{
    return database_name == TEMPORARY_DATABASE || database_name == SYSTEM_DATABASE || database_name == INFORMATION_SCHEMA
        || database_name == INFORMATION_SCHEMA_UPPERCASE;
}


DatabaseAndTable DatabaseCatalog::tryGetByUUID(const UUID & uuid) const
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    const UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    auto it = map_part.map.find(uuid);
    if (it == map_part.map.end())
        return {};
    return it->second;
}


DatabaseAndTable DatabaseCatalog::getTableImpl(
    const StorageID & table_id,
    ContextPtr context_,
    std::optional<Exception> * exception) const
{
    checkStackSize();

    if (!table_id)
    {
        if (exception)
            exception->emplace(ErrorCodes::UNKNOWN_TABLE, "Cannot find table: StorageID is empty");
        return {};
    }

    if (table_id.hasUUID())
    {
        /// Shortcut for tables which have persistent UUID
        auto db_and_table = tryGetByUUID(table_id.uuid);
        if (!db_and_table.first || !db_and_table.second)
        {
            assert(!db_and_table.first && !db_and_table.second);
            if (exception)
                exception->emplace(Exception(ErrorCodes::UNKNOWN_TABLE, "Table {} doesn't exist", table_id.getNameForLogs()));
            return {};
        }

#if USE_LIBPQXX
        if (!context_->isInternalQuery() && (db_and_table.first->getEngineName() == "MaterializedPostgreSQL"))
        {
            db_and_table.second = std::make_shared<StorageMaterializedPostgreSQL>(std::move(db_and_table.second), getContext(),
                                        assert_cast<const DatabaseMaterializedPostgreSQL *>(db_and_table.first.get())->getPostgreSQLDatabaseName(),
                                        db_and_table.second->getStorageID().table_name);
        }
#endif

#if USE_MYSQL
        /// It's definitely not the best place for this logic, but behaviour must be consistent with DatabaseMaterializedMySQL::tryGetTable(...)
        if (!context_->isInternalQuery() && db_and_table.first->getEngineName() == "MaterializedMySQL")
        {
            db_and_table.second = std::make_shared<StorageMaterializedMySQL>(std::move(db_and_table.second), db_and_table.first.get());
        }
#endif
        return db_and_table;
    }


    if (table_id.database_name == TEMPORARY_DATABASE)
    {
        /// For temporary tables UUIDs are set in Context::resolveStorageID(...).
        /// If table_id has no UUID, then the name of database was specified by user and table_id was not resolved through context.
        /// Do not allow access to TEMPORARY_DATABASE because it contains all temporary tables of all contexts and users.
        if (exception)
            exception->emplace(Exception(ErrorCodes::DATABASE_ACCESS_DENIED, "Direct access to `{}` database is not allowed", TEMPORARY_DATABASE));
        return {};
    }

    DatabasePtr database;
    {
        std::lock_guard lock{databases_mutex};
        auto it = databases.find(table_id.getDatabaseName());
        if (databases.end() == it)
        {
            if (exception)
                exception->emplace(Exception(ErrorCodes::UNKNOWN_DATABASE, "Database {} doesn't exist", backQuoteIfNeed(table_id.getDatabaseName())));
            return {};
        }
        database = it->second;
    }

    auto table = database->tryGetTable(table_id.table_name, context_);
    if (!table && exception)
        exception->emplace(Exception(ErrorCodes::UNKNOWN_TABLE, "Table {} doesn't exist", table_id.getNameForLogs()));

    if (!table)
        database = nullptr;

    return {database, table};
}

bool DatabaseCatalog::isPredefinedTable(const StorageID & table_id) const
{
    static const char * information_schema_views[] = {"schemata", "tables", "views", "columns"};
    static const char * information_schema_views_uppercase[] = {"SCHEMATA", "TABLES", "VIEWS", "COLUMNS"};

    auto check_database_and_table_name = [&](const String & database_name, const String & table_name)
    {
        if (database_name == SYSTEM_DATABASE)
        {
            auto storage = getSystemDatabase()->tryGetTable(table_name, getContext());
            return storage && storage->isSystemStorage();
        }
        if (database_name == INFORMATION_SCHEMA)
        {
            return std::find(std::begin(information_schema_views), std::end(information_schema_views), table_name)
                != std::end(information_schema_views);
        }
        if (database_name == INFORMATION_SCHEMA_UPPERCASE)
        {
            return std::find(std::begin(information_schema_views_uppercase), std::end(information_schema_views_uppercase), table_name)
                != std::end(information_schema_views_uppercase);
        }
        return false;
    };

    if (table_id.hasUUID())
    {
        if (auto storage = tryGetByUUID(table_id.uuid).second)
        {
            if (storage->isSystemStorage())
                return true;
            auto res_id = storage->getStorageID();
            String database_name = res_id.getDatabaseName();
            if (database_name != SYSTEM_DATABASE) /// If (database_name == SYSTEM_DATABASE) then we have already checked it (see isSystemStorage() above).
                return check_database_and_table_name(database_name, res_id.getTableName());
        }
        return false;
    }

    return check_database_and_table_name(table_id.getDatabaseName(), table_id.getTableName());
}

void DatabaseCatalog::assertDatabaseExists(const String & database_name) const
{
    std::lock_guard lock{databases_mutex};
    assertDatabaseExistsUnlocked(database_name);
}

void DatabaseCatalog::assertDatabaseDoesntExist(const String & database_name) const
{
    std::lock_guard lock{databases_mutex};
    assertDatabaseDoesntExistUnlocked(database_name);
}

void DatabaseCatalog::assertDatabaseExistsUnlocked(const String & database_name) const
{
    assert(!database_name.empty());
    if (databases.end() == databases.find(database_name))
        throw Exception(ErrorCodes::UNKNOWN_DATABASE, "Database {} doesn't exist", backQuoteIfNeed(database_name));
}


void DatabaseCatalog::assertDatabaseDoesntExistUnlocked(const String & database_name) const
{
    assert(!database_name.empty());
    if (databases.end() != databases.find(database_name))
        throw Exception(ErrorCodes::DATABASE_ALREADY_EXISTS, "Database {} already exists.", backQuoteIfNeed(database_name));
}

void DatabaseCatalog::attachDatabase(const String & database_name, const DatabasePtr & database)
{
    std::lock_guard lock{databases_mutex};
    assertDatabaseDoesntExistUnlocked(database_name);
    databases.emplace(database_name, database);
    NOEXCEPT_SCOPE({
        UUID db_uuid = database->getUUID();
        if (db_uuid != UUIDHelpers::Nil)
            addUUIDMapping(db_uuid, database, nullptr);
    });
}


DatabasePtr DatabaseCatalog::detachDatabase(ContextPtr local_context, const String & database_name, bool drop, bool check_empty)
{
    if (database_name == TEMPORARY_DATABASE)
        throw Exception(ErrorCodes::DATABASE_ACCESS_DENIED, "Cannot detach database with temporary tables.");

    DatabasePtr db;
    {
        std::lock_guard lock{databases_mutex};
        assertDatabaseExistsUnlocked(database_name);
        db = databases.find(database_name)->second;
        UUID db_uuid = db->getUUID();
        if (db_uuid != UUIDHelpers::Nil)
            removeUUIDMapping(db_uuid);
        databases.erase(database_name);
    }

    if (check_empty)
    {
        try
        {
            if (!db->empty())
                throw Exception(ErrorCodes::DATABASE_NOT_EMPTY, "New table appeared in database being dropped or detached. Try again.");
            if (!drop)
                db->assertCanBeDetached(false);
        }
        catch (...)
        {
            attachDatabase(database_name, db);
            throw;
        }
    }

    db->shutdown();

    if (drop)
    {
        UUID db_uuid = db->getUUID();

        /// Delete the database.
        db->drop(local_context);

        /// Old ClickHouse versions did not store database.sql files
        /// Remove metadata dir (if exists) to avoid recreation of .sql file on server startup
        fs::path database_metadata_dir = fs::path(getContext()->getPath()) / "metadata" / escapeForFileName(database_name);
        fs::remove(database_metadata_dir);
        fs::path database_metadata_file = fs::path(getContext()->getPath()) / "metadata" / (escapeForFileName(database_name) + ".sql");
        fs::remove(database_metadata_file);

        if (db_uuid != UUIDHelpers::Nil)
            removeUUIDMappingFinally(db_uuid);
    }

    return db;
}

void DatabaseCatalog::updateDatabaseName(const String & old_name, const String & new_name, const Strings & tables_in_database)
{
    std::lock_guard lock{databases_mutex};
    assert(databases.find(new_name) == databases.end());
    auto it = databases.find(old_name);
    assert(it != databases.end());
    auto db = it->second;
    databases.erase(it);
    databases.emplace(new_name, db);

    /// Update dependencies.
    for (const auto & table_name : tables_in_database)
    {
        auto removed_ref_deps = referential_dependencies.removeDependencies(StorageID{old_name, table_name}, /* remove_isolated_tables= */ true);
        auto removed_loading_deps = loading_dependencies.removeDependencies(StorageID{old_name, table_name}, /* remove_isolated_tables= */ true);
        referential_dependencies.addDependencies(StorageID{new_name, table_name}, removed_ref_deps);
        loading_dependencies.addDependencies(StorageID{new_name, table_name}, removed_loading_deps);
    }
}

DatabasePtr DatabaseCatalog::getDatabase(const String & database_name) const
{
    std::lock_guard lock{databases_mutex};
    assertDatabaseExistsUnlocked(database_name);
    return databases.find(database_name)->second;
}

DatabasePtr DatabaseCatalog::tryGetDatabase(const String & database_name) const
{
    assert(!database_name.empty());
    std::lock_guard lock{databases_mutex};
    auto it = databases.find(database_name);
    if (it == databases.end())
        return {};
    return it->second;
}

DatabasePtr DatabaseCatalog::getDatabase(const UUID & uuid) const
{
    auto db_and_table = tryGetByUUID(uuid);
    if (!db_and_table.first || db_and_table.second)
        throw Exception(ErrorCodes::UNKNOWN_DATABASE, "Database UUID {} does not exist", uuid);
    return db_and_table.first;
}

DatabasePtr DatabaseCatalog::tryGetDatabase(const UUID & uuid) const
{
    assert(uuid != UUIDHelpers::Nil);
    auto db_and_table = tryGetByUUID(uuid);
    if (!db_and_table.first || db_and_table.second)
        return {};
    return db_and_table.first;
}

bool DatabaseCatalog::isDatabaseExist(const String & database_name) const
{
    assert(!database_name.empty());
    std::lock_guard lock{databases_mutex};
    return databases.end() != databases.find(database_name);
}

Databases DatabaseCatalog::getDatabases() const
{
    std::lock_guard lock{databases_mutex};
    return databases;
}

bool DatabaseCatalog::isTableExist(const DB::StorageID & table_id, ContextPtr context_) const
{
    if (table_id.hasUUID())
        return tryGetByUUID(table_id.uuid).second != nullptr;

    DatabasePtr db;
    {
        std::lock_guard lock{databases_mutex};
        auto iter = databases.find(table_id.database_name);
        if (iter != databases.end())
            db = iter->second;
    }
    return db && db->isTableExist(table_id.table_name, context_);
}

void DatabaseCatalog::assertTableDoesntExist(const StorageID & table_id, ContextPtr context_) const
{
    if (isTableExist(table_id, context_))
        throw Exception(ErrorCodes::TABLE_ALREADY_EXISTS, "Table {} already exists.", table_id);
}

DatabasePtr DatabaseCatalog::getDatabaseForTemporaryTables() const
{
    return getDatabase(TEMPORARY_DATABASE);
}

DatabasePtr DatabaseCatalog::getSystemDatabase() const
{
    return getDatabase(SYSTEM_DATABASE);
}

void DatabaseCatalog::addUUIDMapping(const UUID & uuid)
{
    addUUIDMapping(uuid, nullptr, nullptr);
}

void DatabaseCatalog::addUUIDMapping(const UUID & uuid, const DatabasePtr & database, const StoragePtr & table)
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    assert(database || !table);
    UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    auto [it, inserted] = map_part.map.try_emplace(uuid, database, table);
    if (inserted)
    {
        /// Mapping must be locked before actually inserting something
        chassert((!database && !table));
        return;
    }

    auto & prev_database = it->second.first;
    auto & prev_table = it->second.second;
    assert(prev_database || !prev_table);

    if (!prev_database && database)
    {
        /// It's empty mapping, it was created to "lock" UUID and prevent collision. Just update it.
        prev_database = database;
        prev_table = table;
        return;
    }

    /// We are trying to replace existing mapping (prev_database != nullptr), it's logical error
    if (database || table)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Mapping for table with UUID={} already exists", uuid);
    /// Normally this should never happen, but it's possible when the same UUIDs are explicitly specified in different CREATE queries,
    /// so it's not LOGICAL_ERROR
    throw Exception(ErrorCodes::TABLE_ALREADY_EXISTS, "Mapping for table with UUID={} already exists. It happened due to UUID collision, "
                    "most likely because some not random UUIDs were manually specified in CREATE queries.", uuid);
}

void DatabaseCatalog::removeUUIDMapping(const UUID & uuid)
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    auto it = map_part.map.find(uuid);
    if (it == map_part.map.end())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Mapping for table with UUID={} doesn't exist", uuid);
    it->second = {};
}

void DatabaseCatalog::removeUUIDMappingFinally(const UUID & uuid)
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    if (!map_part.map.erase(uuid))
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Mapping for table with UUID={} doesn't exist", uuid);
}

void DatabaseCatalog::updateUUIDMapping(const UUID & uuid, DatabasePtr database, StoragePtr table)
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    assert(database && table);
    UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    auto it = map_part.map.find(uuid);
    if (it == map_part.map.end())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Mapping for table with UUID={} doesn't exist", uuid);
    auto & prev_database = it->second.first;
    auto & prev_table = it->second.second;
    assert(prev_database && prev_table);
    prev_database = std::move(database);
    prev_table = std::move(table);
}

bool DatabaseCatalog::hasUUIDMapping(const UUID & uuid)
{
    assert(uuid != UUIDHelpers::Nil && getFirstLevelIdx(uuid) < uuid_map.size());
    UUIDToStorageMapPart & map_part = uuid_map[getFirstLevelIdx(uuid)];
    std::lock_guard lock{map_part.mutex};
    return map_part.map.contains(uuid);
}

std::unique_ptr<DatabaseCatalog> DatabaseCatalog::database_catalog;

DatabaseCatalog::DatabaseCatalog(ContextMutablePtr global_context_)
    : WithMutableContext(global_context_)
    , referential_dependencies{"ReferentialDeps"}
    , loading_dependencies{"LoadingDeps"}
    , view_dependencies{"ViewDeps"}
    , log(&Poco::Logger::get("DatabaseCatalog"))
{
}

DatabaseCatalog & DatabaseCatalog::init(ContextMutablePtr global_context_)
{
    if (database_catalog)
    {
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Database catalog is initialized twice. This is a bug.");
    }

    database_catalog.reset(new DatabaseCatalog(global_context_));

    return *database_catalog;
}

DatabaseCatalog & DatabaseCatalog::instance()
{
    if (!database_catalog)
    {
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Database catalog is not initialized. This is a bug.");
    }

    return *database_catalog;
}

void DatabaseCatalog::shutdown()
{
    // The catalog might not be initialized yet by init(global_context). It can
    // happen if some exception was thrown on first steps of startup.
    if (database_catalog)
    {
        database_catalog->shutdownImpl();
    }
}

DatabasePtr DatabaseCatalog::getDatabase(const String & database_name, ContextPtr local_context) const
{
    String resolved_database = local_context->resolveDatabase(database_name);
    return getDatabase(resolved_database);
}

void DatabaseCatalog::addViewDependency(const StorageID & source_table_id, const StorageID & view_id)
{
    std::lock_guard lock{databases_mutex};
    view_dependencies.addDependency(source_table_id, view_id);

}

void DatabaseCatalog::removeViewDependency(const StorageID & source_table_id, const StorageID & view_id)
{
    std::lock_guard lock{databases_mutex};
    view_dependencies.removeDependency(source_table_id, view_id, /* remove_isolated_tables= */ true);
}

std::vector<StorageID> DatabaseCatalog::getDependentViews(const StorageID & source_table_id) const
{
    std::lock_guard lock{databases_mutex};
    return view_dependencies.getDependencies(source_table_id);
}

void DatabaseCatalog::updateViewDependency(const StorageID & old_source_table_id, const StorageID & old_view_id,
                                           const StorageID & new_source_table_id, const StorageID & new_view_id)
{
    std::lock_guard lock{databases_mutex};
    if (!old_source_table_id.empty())
        view_dependencies.removeDependency(old_source_table_id, old_view_id, /* remove_isolated_tables= */ true);
    if (!new_source_table_id.empty())
        view_dependencies.addDependency(new_source_table_id, new_view_id);
}

DDLGuardPtr DatabaseCatalog::getDDLGuard(const String & database, const String & table)
{
    std::unique_lock lock(ddl_guards_mutex);
    /// TSA does not support unique_lock
    auto db_guard_iter = TSA_SUPPRESS_WARNING_FOR_WRITE(ddl_guards).try_emplace(database).first;
    DatabaseGuard & db_guard = db_guard_iter->second;
    return std::make_unique<DDLGuard>(db_guard.first, db_guard.second, std::move(lock), table, database);
}

std::unique_lock<SharedMutex> DatabaseCatalog::getExclusiveDDLGuardForDatabase(const String & database)
{
    DDLGuards::iterator db_guard_iter;
    {
        std::lock_guard lock(ddl_guards_mutex);
        db_guard_iter = ddl_guards.try_emplace(database).first;
        assert(db_guard_iter->second.first.contains(""));
    }
    DatabaseGuard & db_guard = db_guard_iter->second;
    return std::unique_lock{db_guard.second};
}

bool DatabaseCatalog::isDictionaryExist(const StorageID & table_id) const
{
    auto storage = tryGetTable(table_id, getContext());
    bool storage_is_dictionary = storage && storage->isDictionary();

    return storage_is_dictionary;
}

StoragePtr DatabaseCatalog::getTable(const StorageID & table_id, ContextPtr local_context) const
{
    std::optional<Exception> exc;
    auto res = getTableImpl(table_id, local_context, &exc);
    if (!res.second)
        throw Exception(*exc);
    return res.second;
}

StoragePtr DatabaseCatalog::tryGetTable(const StorageID & table_id, ContextPtr local_context) const
{
    return getTableImpl(table_id, local_context, nullptr).second;
}

DatabaseAndTable DatabaseCatalog::getDatabaseAndTable(const StorageID & table_id, ContextPtr local_context) const
{
    std::optional<Exception> exc;
    auto res = getTableImpl(table_id, local_context, &exc);
    if (!res.second)
        throw Exception(*exc);
    return res;
}

DatabaseAndTable DatabaseCatalog::tryGetDatabaseAndTable(const StorageID & table_id, ContextPtr local_context) const
{
    return getTableImpl(table_id, local_context, nullptr);
}

void DatabaseCatalog::loadMarkedAsDroppedTables()
{
    assert(!cleanup_task);

    /// /clickhouse_root/metadata_dropped/ contains files with metadata of tables,
    /// which where marked as dropped by Atomic databases.
    /// Data directories of such tables still exists in store/
    /// and metadata still exists in ZooKeeper for ReplicatedMergeTree tables.
    /// If server restarts before such tables was completely dropped,
    /// we should load them and enqueue cleanup to remove data from store/ and metadata from ZooKeeper

    std::map<String, StorageID> dropped_metadata;
    String path = getContext()->getPath() + "metadata_dropped/";

    if (!std::filesystem::exists(path))
    {
        return;
    }

    Poco::DirectoryIterator dir_end;
    for (Poco::DirectoryIterator it(path); it != dir_end; ++it)
    {
        /// File name has the following format:
        /// database_name.table_name.uuid.sql

        /// Ignore unexpected files
        if (!it.name().ends_with(".sql"))
            continue;

        /// Process .sql files with metadata of tables which were marked as dropped
        StorageID dropped_id = StorageID::createEmpty();
        size_t dot_pos = it.name().find('.');
        if (dot_pos == std::string::npos)
            continue;
        dropped_id.database_name = unescapeForFileName(it.name().substr(0, dot_pos));

        size_t prev_dot_pos = dot_pos;
        dot_pos = it.name().find('.', prev_dot_pos + 1);
        if (dot_pos == std::string::npos)
            continue;
        dropped_id.table_name = unescapeForFileName(it.name().substr(prev_dot_pos + 1, dot_pos - prev_dot_pos - 1));

        prev_dot_pos = dot_pos;
        dot_pos = it.name().find('.', prev_dot_pos + 1);
        if (dot_pos == std::string::npos)
            continue;
        dropped_id.uuid = parse<UUID>(it.name().substr(prev_dot_pos + 1, dot_pos - prev_dot_pos - 1));

        String full_path = path + it.name();
        dropped_metadata.emplace(std::move(full_path), std::move(dropped_id));
    }

    LOG_INFO(log, "Found {} partially dropped tables. Will load them and retry removal.", dropped_metadata.size());

    ThreadPool pool(CurrentMetrics::DatabaseCatalogThreads, CurrentMetrics::DatabaseCatalogThreadsActive);
    for (const auto & elem : dropped_metadata)
    {
        pool.scheduleOrThrowOnError([&]()
        {
            this->enqueueDroppedTableCleanup(elem.second, nullptr, elem.first);
        });
    }
    pool.wait();
}

String DatabaseCatalog::getPathForDroppedMetadata(const StorageID & table_id) const
{
    return getContext()->getPath() + "metadata_dropped/" +
           escapeForFileName(table_id.getDatabaseName()) + "." +
           escapeForFileName(table_id.getTableName()) + "." +
           toString(table_id.uuid) + ".sql";
}

String DatabaseCatalog::getPathForMetadata(const StorageID & table_id) const
{
    return getContext()->getPath() + "metadata/" +
           escapeForFileName(table_id.getDatabaseName()) + "/" +
           escapeForFileName(table_id.getTableName()) + ".sql";
}

void DatabaseCatalog::enqueueDroppedTableCleanup(StorageID table_id, StoragePtr table, String dropped_metadata_path, bool ignore_delay)
{
    assert(table_id.hasUUID());
    assert(!table || table->getStorageID().uuid == table_id.uuid);
    assert(dropped_metadata_path == getPathForDroppedMetadata(table_id));

    /// Table was removed from database. Enqueue removal of its data from disk.
    time_t drop_time;
    if (table)
    {
        chassert(hasUUIDMapping(table_id.uuid));
        drop_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        /// Do not postpone removal of in-memory tables
        ignore_delay = ignore_delay || !table->storesDataOnDisk();
        table->is_dropped = true;
    }
    else
    {
        /// Try load table from metadata to drop it correctly (e.g. remove metadata from zk or remove data from all volumes)
        LOG_INFO(log, "Trying load partially dropped table {} from {}", table_id.getNameForLogs(), dropped_metadata_path);
        ASTPtr ast = DatabaseOnDisk::parseQueryFromMetadata(
            log, getContext(), dropped_metadata_path, /*throw_on_error*/ false, /*remove_empty*/ false);
        auto * create = typeid_cast<ASTCreateQuery *>(ast.get());
        assert(!create || create->uuid == table_id.uuid);

        if (create)
        {
            String data_path = "store/" + getPathForUUID(table_id.uuid);
            create->setDatabase(table_id.database_name);
            create->setTable(table_id.table_name);
            try
            {
                table = createTableFromAST(*create, table_id.getDatabaseName(), data_path, getContext(), /* force_restore */ true).second;
                table->is_dropped = true;
            }
            catch (...)
            {
                tryLogCurrentException(log, "Cannot load partially dropped table " + table_id.getNameForLogs() +
                                            " from: " + dropped_metadata_path +
                                            ". Parsed query: " + serializeAST(*create) +
                                            ". Will remove metadata and " + data_path +
                                            ". Garbage may be left in ZooKeeper.");
            }
        }
        else
        {
            LOG_WARNING(log, "Cannot parse metadata of partially dropped table {} from {}. Will remove metadata file and data directory. Garbage may be left in /store directory and ZooKeeper.", table_id.getNameForLogs(), dropped_metadata_path);
        }

        addUUIDMapping(table_id.uuid);
        drop_time = FS::getModificationTime(dropped_metadata_path);
    }

    std::lock_guard lock(tables_marked_dropped_mutex);
    if (ignore_delay)
        tables_marked_dropped.push_front({table_id, table, dropped_metadata_path, drop_time});
    else
        tables_marked_dropped.push_back({table_id, table, dropped_metadata_path, drop_time + drop_delay_sec});
    tables_marked_dropped_ids.insert(table_id.uuid);
    CurrentMetrics::add(CurrentMetrics::TablesToDropQueueSize, 1);

    /// If list of dropped tables was empty, start a drop task.
    /// If ignore_delay is set, schedule drop task as soon as possible.
    if (drop_task && (tables_marked_dropped.size() == 1 || ignore_delay))
        (*drop_task)->schedule();
}

void DatabaseCatalog::dequeueDroppedTableCleanup(StorageID table_id)
{
    String latest_metadata_dropped_path;
    TableMarkedAsDropped dropped_table;
    {
        std::lock_guard lock(tables_marked_dropped_mutex);
        time_t latest_drop_time = std::numeric_limits<time_t>::min();
        auto it_dropped_table = tables_marked_dropped.end();
        for (auto it = tables_marked_dropped.begin(); it != tables_marked_dropped.end(); ++it)
        {
            auto storage_ptr = it->table;
            if (it->table_id.uuid == table_id.uuid)
            {
                it_dropped_table = it;
                dropped_table = *it;
                break;
            }
            /// If table uuid exists, only find tables with equal uuid.
            if (table_id.uuid != UUIDHelpers::Nil)
                continue;
            if (it->table_id.database_name == table_id.database_name &&
                it->table_id.table_name == table_id.table_name &&
                it->drop_time >= latest_drop_time)
            {
                latest_drop_time = it->drop_time;
                it_dropped_table = it;
                dropped_table = *it;
            }
        }
        if (it_dropped_table == tables_marked_dropped.end())
            throw Exception(ErrorCodes::UNKNOWN_TABLE,
                "The drop task of table {} is in progress, has been dropped or the database engine doesn't support it",
                table_id.getNameForLogs());
        latest_metadata_dropped_path = it_dropped_table->metadata_path;
        String table_metadata_path = getPathForMetadata(it_dropped_table->table_id);

        /// a table is successfully marked undropped,
        /// if and only if its metadata file was moved to a database.
        /// This maybe throw exception.
        renameNoReplace(latest_metadata_dropped_path, table_metadata_path);

        tables_marked_dropped.erase(it_dropped_table);
        [[maybe_unused]] auto removed = tables_marked_dropped_ids.erase(dropped_table.table_id.uuid);
        assert(removed);
        CurrentMetrics::sub(CurrentMetrics::TablesToDropQueueSize, 1);
    }

    LOG_INFO(log, "Attaching undropped table {} (metadata moved from {})",
             dropped_table.table_id.getNameForLogs(), latest_metadata_dropped_path);

    /// It's unsafe to create another instance while the old one exists
    /// We cannot wait on shared_ptr's refcount, so it's busy wait
    while (!dropped_table.table.unique())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    dropped_table.table.reset();

    auto ast_attach = std::make_shared<ASTCreateQuery>();
    ast_attach->attach = true;
    ast_attach->setDatabase(dropped_table.table_id.database_name);
    ast_attach->setTable(dropped_table.table_id.table_name);

    auto query_context = Context::createCopy(getContext());
    /// Attach table needs to acquire ddl guard, that has already been acquired in undrop table,
    /// and cannot be acquired in the attach table again.
    InterpreterCreateQuery interpreter(ast_attach, query_context);
    interpreter.setForceAttach(true);
    interpreter.setForceRestoreData(true);
    interpreter.setDontNeedDDLGuard();  /// It's already locked by caller
    interpreter.execute();

    LOG_INFO(log, "Table {} was successfully undropped.", dropped_table.table_id.getNameForLogs());
}

void DatabaseCatalog::dropTableDataTask()
{
    /// Background task that removes data of tables which were marked as dropped by Atomic databases.
    /// Table can be removed when it's not used by queries and drop_delay_sec elapsed since it was marked as dropped.

    bool need_reschedule = true;
    /// Default reschedule time for the case when we are waiting for reference count to become 1.
    size_t schedule_after_ms = reschedule_time_ms;
    TableMarkedAsDropped table;
    try
    {
        std::lock_guard lock(tables_marked_dropped_mutex);
        if (tables_marked_dropped.empty())
            return;
        time_t current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        time_t min_drop_time = std::numeric_limits<time_t>::max();
        size_t tables_in_use_count = 0;
        auto it = std::find_if(tables_marked_dropped.begin(), tables_marked_dropped.end(), [&](const auto & elem)
        {
            bool not_in_use = !elem.table || elem.table.unique();
            bool old_enough = elem.drop_time <= current_time;
            min_drop_time = std::min(min_drop_time, elem.drop_time);
            tables_in_use_count += !not_in_use;
            return not_in_use && old_enough;
        });
        if (it != tables_marked_dropped.end())
        {
            table = std::move(*it);
            LOG_INFO(log, "Have {} tables in drop queue ({} of them are in use), will try drop {}",
                     tables_marked_dropped.size(), tables_in_use_count, table.table_id.getNameForLogs());
            tables_marked_dropped.erase(it);
            /// Schedule the task as soon as possible, while there are suitable tables to drop.
            schedule_after_ms = 0;
        }
        else if (current_time < min_drop_time)
        {
            /// We are waiting for drop_delay_sec to exceed, no sense to wakeup until min_drop_time.
            /// If new table is added to the queue with ignore_delay flag, schedule() is called to wakeup the task earlier.
            schedule_after_ms = (min_drop_time - current_time) * 1000;
            LOG_TRACE(log, "Not found any suitable tables to drop, still have {} tables in drop queue ({} of them are in use). "
                           "Will check again after {} seconds", tables_marked_dropped.size(), tables_in_use_count, min_drop_time - current_time);
        }
        need_reschedule = !tables_marked_dropped.empty();
    }
    catch (...)
    {
        tryLogCurrentException(log, __PRETTY_FUNCTION__);
    }

    if (table.table_id)
    {
        try
        {
            dropTableFinally(table);
            std::lock_guard lock(tables_marked_dropped_mutex);
            [[maybe_unused]] auto removed = tables_marked_dropped_ids.erase(table.table_id.uuid);
            assert(removed);
        }
        catch (...)
        {
            tryLogCurrentException(log, "Cannot drop table " + table.table_id.getNameForLogs() +
                                        ". Will retry later.");
            {
                table.drop_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + drop_error_cooldown_sec;
                std::lock_guard lock(tables_marked_dropped_mutex);
                tables_marked_dropped.emplace_back(std::move(table));
                /// If list of dropped tables was empty, schedule a task to retry deletion.
                if (tables_marked_dropped.size() == 1)
                {
                    need_reschedule = true;
                    schedule_after_ms = drop_error_cooldown_sec * 1000;
                }
            }
        }

        wait_table_finally_dropped.notify_all();
    }

    /// Do not schedule a task if there is no tables to drop
    if (need_reschedule)
        (*drop_task)->scheduleAfter(schedule_after_ms);
}

void DatabaseCatalog::dropTableFinally(const TableMarkedAsDropped & table)
{
    if (table.table)
    {
        table.table->drop();
    }

    /// Even if table is not loaded, try remove its data from disks.
    for (const auto & [disk_name, disk] : getContext()->getDisksMap())
    {
        String data_path = "store/" + getPathForUUID(table.table_id.uuid);
        if (disk->isReadOnly() || !disk->exists(data_path))
            continue;

        LOG_INFO(log, "Removing data directory {} of dropped table {} from disk {}", data_path, table.table_id.getNameForLogs(), disk_name);
        disk->removeRecursive(data_path);
    }

    LOG_INFO(log, "Removing metadata {} of dropped table {}", table.metadata_path, table.table_id.getNameForLogs());
    fs::remove(fs::path(table.metadata_path));

    removeUUIDMappingFinally(table.table_id.uuid);
    CurrentMetrics::sub(CurrentMetrics::TablesToDropQueueSize, 1);
}

String DatabaseCatalog::getPathForUUID(const UUID & uuid)
{
    const size_t uuid_prefix_len = 3;
    return toString(uuid).substr(0, uuid_prefix_len) + '/' + toString(uuid) + '/';
}

void DatabaseCatalog::waitTableFinallyDropped(const UUID & uuid)
{
    if (uuid == UUIDHelpers::Nil)
        return;

    LOG_DEBUG(log, "Waiting for table {} to be finally dropped", toString(uuid));
    std::unique_lock lock{tables_marked_dropped_mutex};
    wait_table_finally_dropped.wait(lock, [&]() TSA_REQUIRES(tables_marked_dropped_mutex) -> bool
    {
        return !tables_marked_dropped_ids.contains(uuid) || is_shutting_down;
    });

    /// TSA doesn't support unique_lock
    if (TSA_SUPPRESS_WARNING_FOR_READ(tables_marked_dropped_ids).contains(uuid))
        throw Exception(ErrorCodes::UNFINISHED, "Did not finish dropping the table with UUID {} because the server is shutting down, "
                                                "will finish after restart", uuid);
}

void DatabaseCatalog::addDependencies(
    const StorageID & table_id,
    const std::vector<StorageID> & new_referential_dependencies,
    const std::vector<StorageID> & new_loading_dependencies)
{
    if (new_referential_dependencies.empty() && new_loading_dependencies.empty())
        return;
    std::lock_guard lock{databases_mutex};
    if (!new_referential_dependencies.empty())
        referential_dependencies.addDependencies(table_id, new_referential_dependencies);
    if (!new_loading_dependencies.empty())
        loading_dependencies.addDependencies(table_id, new_loading_dependencies);
}

void DatabaseCatalog::addDependencies(
    const QualifiedTableName & table_name,
    const TableNamesSet & new_referential_dependencies,
    const TableNamesSet & new_loading_dependencies)
{
    if (new_referential_dependencies.empty() && new_loading_dependencies.empty())
        return;
    std::lock_guard lock{databases_mutex};
    if (!new_referential_dependencies.empty())
        referential_dependencies.addDependencies(table_name, new_referential_dependencies);
    if (!new_loading_dependencies.empty())
        loading_dependencies.addDependencies(table_name, new_loading_dependencies);
}

void DatabaseCatalog::addDependencies(
    const TablesDependencyGraph & new_referential_dependencies, const TablesDependencyGraph & new_loading_dependencies)
{
    std::lock_guard lock{databases_mutex};
    referential_dependencies.mergeWith(new_referential_dependencies);
    loading_dependencies.mergeWith(new_loading_dependencies);
}

std::vector<StorageID> DatabaseCatalog::getReferentialDependencies(const StorageID & table_id) const
{
    std::lock_guard lock{databases_mutex};
    return referential_dependencies.getDependencies(table_id);
}

std::vector<StorageID> DatabaseCatalog::getReferentialDependents(const StorageID & table_id) const
{
    std::lock_guard lock{databases_mutex};
    return referential_dependencies.getDependents(table_id);
}

std::vector<StorageID> DatabaseCatalog::getLoadingDependencies(const StorageID & table_id) const
{
    std::lock_guard lock{databases_mutex};
    return loading_dependencies.getDependencies(table_id);
}

std::vector<StorageID> DatabaseCatalog::getLoadingDependents(const StorageID & table_id) const
{
    std::lock_guard lock{databases_mutex};
    return loading_dependencies.getDependents(table_id);
}

std::pair<std::vector<StorageID>, std::vector<StorageID>> DatabaseCatalog::removeDependencies(
    const StorageID & table_id, bool check_referential_dependencies, bool check_loading_dependencies, bool is_drop_database)
{
    std::lock_guard lock{databases_mutex};
    checkTableCanBeRemovedOrRenamedUnlocked(table_id, check_referential_dependencies, check_loading_dependencies, is_drop_database);
    return {referential_dependencies.removeDependencies(table_id, /* remove_isolated_tables= */ true),
            loading_dependencies.removeDependencies(table_id, /* remove_isolated_tables= */ true)};
}

void DatabaseCatalog::updateDependencies(
    const StorageID & table_id, const TableNamesSet & new_referential_dependencies, const TableNamesSet & new_loading_dependencies)
{
    std::lock_guard lock{databases_mutex};
    referential_dependencies.removeDependencies(table_id, /* remove_isolated_tables= */ true);
    loading_dependencies.removeDependencies(table_id, /* remove_isolated_tables= */ true);
    if (!new_referential_dependencies.empty())
        referential_dependencies.addDependencies(table_id, new_referential_dependencies);
    if (!new_loading_dependencies.empty())
        loading_dependencies.addDependencies(table_id, new_loading_dependencies);
}

void DatabaseCatalog::checkTableCanBeRemovedOrRenamed(
    const StorageID & table_id, bool check_referential_dependencies, bool check_loading_dependencies, bool is_drop_database) const
{
    if (!check_referential_dependencies && !check_loading_dependencies)
        return;
    std::lock_guard lock{databases_mutex};
    return checkTableCanBeRemovedOrRenamedUnlocked(table_id, check_referential_dependencies, check_loading_dependencies, is_drop_database);
}

void DatabaseCatalog::checkTableCanBeRemovedOrRenamedUnlocked(
    const StorageID & removing_table, bool check_referential_dependencies, bool check_loading_dependencies, bool is_drop_database) const
{
    chassert(!check_referential_dependencies || !check_loading_dependencies); /// These flags must not be both set.
    std::vector<StorageID> dependents;
    if (check_referential_dependencies)
        dependents = referential_dependencies.getDependents(removing_table);
    else if (check_loading_dependencies)
        dependents = loading_dependencies.getDependents(removing_table);
    else
        return;

    if (!is_drop_database)
    {
        if (!dependents.empty())
            throw Exception(ErrorCodes::HAVE_DEPENDENT_OBJECTS, "Cannot drop or rename {}, because some tables depend on it: {}",
                            removing_table, fmt::join(dependents, ", "));
        return;
    }

    /// For DROP DATABASE we should ignore dependent tables from the same database.
    /// TODO unload tables in reverse topological order and remove this code
    std::vector<StorageID> from_other_databases;
    for (const auto & dependent : dependents)
        if (dependent.database_name != removing_table.database_name)
            from_other_databases.push_back(dependent);

    if (!from_other_databases.empty())
        throw Exception(ErrorCodes::HAVE_DEPENDENT_OBJECTS, "Cannot drop or rename {}, because some tables depend on it: {}",
                        removing_table, fmt::join(from_other_databases, ", "));
}

void DatabaseCatalog::cleanupStoreDirectoryTask()
{
    for (const auto & [disk_name, disk] : getContext()->getDisksMap())
    {
        if (!disk->supportsStat() || !disk->supportsChmod())
            continue;

        size_t affected_dirs = 0;
        size_t checked_dirs = 0;
        for (auto it = disk->iterateDirectory("store"); it->isValid(); it->next())
        {
            String prefix = it->name();
            bool expected_prefix_dir = disk->isDirectory(it->path()) && prefix.size() == 3 && isHexDigit(prefix[0]) && isHexDigit(prefix[1])
                && isHexDigit(prefix[2]);

            if (!expected_prefix_dir)
            {
                LOG_WARNING(log, "Found invalid directory {} on disk {}, will try to remove it", it->path(), disk_name);
                checked_dirs += 1;
                affected_dirs += maybeRemoveDirectory(disk_name, disk, it->path());
                continue;
            }

            for (auto jt = disk->iterateDirectory(it->path()); jt->isValid(); jt->next())
            {
                String uuid_str = jt->name();
                UUID uuid;
                bool parsed = tryParse(uuid, uuid_str);

                bool expected_dir = disk->isDirectory(jt->path()) && parsed && uuid != UUIDHelpers::Nil && uuid_str.starts_with(prefix);

                if (!expected_dir)
                {
                    LOG_WARNING(log, "Found invalid directory {} on disk {}, will try to remove it", jt->path(), disk_name);
                    checked_dirs += 1;
                    affected_dirs += maybeRemoveDirectory(disk_name, disk, jt->path());
                    continue;
                }

                /// Order is important
                if (!hasUUIDMapping(uuid))
                {
                    /// We load uuids even for detached and permanently detached tables,
                    /// so it looks safe enough to remove directory if we don't have uuid mapping for it.
                    /// No table or database using this directory should concurrently appear,
                    /// because creation of new table would fail with "directory already exists".
                    checked_dirs += 1;
                    affected_dirs += maybeRemoveDirectory(disk_name, disk, jt->path());
                }
            }
        }

        if (affected_dirs)
            LOG_INFO(log, "Cleaned up {} directories from store/ on disk {}", affected_dirs, disk_name);
        if (checked_dirs == 0)
            LOG_TEST(log, "Nothing to clean up from store/ on disk {}", disk_name);
    }

    (*cleanup_task)->scheduleAfter(unused_dir_cleanup_period_sec * 1000);
}

bool DatabaseCatalog::maybeRemoveDirectory(const String & disk_name, const DiskPtr & disk, const String & unused_dir)
{
    /// "Safe" automatic removal of some directory.
    /// At first we do not remove anything and only revoke all access right.
    /// And remove only if nobody noticed it after, for example, one month.

    try
    {
        struct stat st = disk->stat(unused_dir);

        if (st.st_uid != geteuid())
        {
            /// Directory is not owned by clickhouse, it's weird, let's ignore it (chmod will likely fail anyway).
            LOG_WARNING(log, "Found directory {} with unexpected owner (uid={}) on disk {}", unused_dir, st.st_uid, disk_name);
            return false;
        }

        time_t max_modification_time = std::max(st.st_atime, std::max(st.st_mtime, st.st_ctime));
        time_t current_time = time(nullptr);
        if (st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO))
        {
            if (current_time <= max_modification_time + unused_dir_hide_timeout_sec)
                return false;

            LOG_INFO(log, "Removing access rights for unused directory {} from disk {} (will remove it when timeout exceed)", unused_dir, disk_name);

            /// Explicitly update modification time just in case

            disk->setLastModified(unused_dir, Poco::Timestamp::fromEpochTime(current_time));

            /// Remove all access right
            disk->chmod(unused_dir, 0);

            return true;
        }
        else
        {
            if (!unused_dir_rm_timeout_sec)
                return false;

            if (current_time <= max_modification_time + unused_dir_rm_timeout_sec)
                return false;

            LOG_INFO(log, "Removing unused directory {} from disk {}", unused_dir, disk_name);

            /// We have to set these access rights to make recursive removal work
            disk->chmod(unused_dir, S_IRWXU);

            disk->removeRecursive(unused_dir);

            return true;
        }
    }
    catch (...)
    {
        tryLogCurrentException(log, fmt::format("Failed to remove unused directory {} from disk {} ({})",
                                                unused_dir, disk->getName(), disk->getPath()));
        return false;
    }
}

static void maybeUnlockUUID(UUID uuid)
{
    if (uuid == UUIDHelpers::Nil)
        return;

    chassert(DatabaseCatalog::instance().hasUUIDMapping(uuid));
    auto db_and_table = DatabaseCatalog::instance().tryGetByUUID(uuid);
    if (!db_and_table.first && !db_and_table.second)
    {
        DatabaseCatalog::instance().removeUUIDMappingFinally(uuid);
        return;
    }
    chassert(db_and_table.first || !db_and_table.second);
}

TemporaryLockForUUIDDirectory::TemporaryLockForUUIDDirectory(UUID uuid_)
    : uuid(uuid_)
{
    if (uuid != UUIDHelpers::Nil)
        DatabaseCatalog::instance().addUUIDMapping(uuid);
}

TemporaryLockForUUIDDirectory::~TemporaryLockForUUIDDirectory()
{
    maybeUnlockUUID(uuid);
}

TemporaryLockForUUIDDirectory::TemporaryLockForUUIDDirectory(TemporaryLockForUUIDDirectory && rhs) noexcept
    : uuid(rhs.uuid)
{
    rhs.uuid = UUIDHelpers::Nil;
}

TemporaryLockForUUIDDirectory & TemporaryLockForUUIDDirectory::operator = (TemporaryLockForUUIDDirectory && rhs) noexcept
{
    maybeUnlockUUID(uuid);
    uuid = rhs.uuid;
    rhs.uuid = UUIDHelpers::Nil;
    return *this;
}


DDLGuard::DDLGuard(Map & map_, SharedMutex & db_mutex_, std::unique_lock<std::mutex> guards_lock_, const String & elem, const String & database_name)
        : map(map_), db_mutex(db_mutex_), guards_lock(std::move(guards_lock_))
{
    it = map.emplace(elem, Entry{std::make_unique<std::mutex>(), 0}).first;
    ++it->second.counter;
    guards_lock.unlock();
    table_lock = std::unique_lock(*it->second.mutex);
    is_database_guard = elem.empty();
    if (!is_database_guard)
    {

        bool locked_database_for_read = db_mutex.try_lock_shared();
        if (!locked_database_for_read)
        {
            releaseTableLock();
            throw Exception(ErrorCodes::UNKNOWN_DATABASE, "Database {} is currently dropped or renamed", database_name);
        }
    }
}

void DDLGuard::releaseTableLock() noexcept
{
    if (table_lock_removed)
        return;

    table_lock_removed = true;
    guards_lock.lock();
    UInt32 counter = --it->second.counter;
    table_lock.unlock();
    if (counter == 0)
        map.erase(it);
    guards_lock.unlock();
}

DDLGuard::~DDLGuard()
{
    if (!is_database_guard)
        db_mutex.unlock_shared();
    releaseTableLock();
}

}
