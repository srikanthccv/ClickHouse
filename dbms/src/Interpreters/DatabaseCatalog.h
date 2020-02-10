#pragma once
#include <Storages/IStorage_fwd.h>
#include <Core/UUID.h>
#include <boost/noncopyable.hpp>
#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <array>


namespace DB
{

class Context;
class IDatabase;
struct StorageID;
class Exception;
using DatabasePtr = std::shared_ptr<IDatabase>;
using DatabaseAndTable = std::pair<DatabasePtr, StoragePtr>;
using Databases = std::map<String, std::shared_ptr<IDatabase>>;


/// Allows executing DDL query only in one thread.
/// Puts an element into the map, locks tables's mutex, counts how much threads run parallel query on the table,
/// when counter is 0 erases element in the destructor.
/// If the element already exists in the map, waits, when ddl query will be finished in other thread.
class DDLGuard
{
public:
    struct Entry
    {
        std::unique_ptr<std::mutex> mutex;
        UInt32 counter;
    };

    /// Element name -> (mutex, counter).
    /// NOTE: using std::map here (and not std::unordered_map) to avoid iterator invalidation on insertion.
    using Map = std::map<String, Entry>;

    DDLGuard(Map & map_, std::unique_lock<std::mutex> guards_lock_, const String & elem);
    ~DDLGuard();

private:
    Map & map;
    Map::iterator it;
    std::unique_lock<std::mutex> guards_lock;
    std::unique_lock<std::mutex> table_lock;
};


class DatabaseCatalog : boost::noncopyable
{
public:
    static constexpr const char * TEMPORARY_DATABASE = "_temporary_and_external_tables";
    static constexpr const char * SYSTEM_DATABASE = "system";

    static DatabaseCatalog & instance();

    //DatabaseCatalog(/*Context & global_context_, String default_database_*/) {}
    //: global_context(global_context_)/*, default_database(std::move(default_database_))*/ {}

    void loadDatabases();
    void shutdown();

    /// Get an object that protects the table from concurrently executing multiple DDL operations.
    std::unique_ptr<DDLGuard> getDDLGuard(const String & database, const String & table);

    //static String resolveDatabase(const String & database_name, const String & current_database);
    void assertDatabaseExists(const String & database_name) const;
    void assertDatabaseDoesntExist(const String & database_name) const;

    DatabasePtr getDatabaseForTemporaryTables() const;
    DatabasePtr getSystemDatabase() const;

    void attachDatabase(const String & database_name, const DatabasePtr & database);     // ca, a
    DatabasePtr detachDatabase(const String & database_name);                            // (sr), ca, a

    DatabasePtr getDatabase(const String & database_name, const Context & local_context) const;
    DatabasePtr getDatabase(const String & database_name) const;                         // sr, ca, a
    DatabasePtr tryGetDatabase(const String & database_name) const;                      // sr
    bool isDatabaseExist(const String & database_name) const;                                                           // sr, ca
    Databases getDatabases() const;

    DatabaseAndTable tryGetByUUID(const UUID & uuid) const;

    void assertTableDoesntExist(const StorageID & table_id, const Context & context) const;                             // sr, ca
    bool isTableExist(const StorageID & table_id, const Context & context) const;                                       // sr, ca

    void addUUIDMapping(const UUID & uuid, DatabasePtr database, StoragePtr table);
    void removeUUIDMapping(const UUID & uuid);

    StoragePtr getTable(const StorageID & table_id, const Context & local_context, std::optional<Exception> * exception) const;

private:
    DatabaseCatalog() = default;
    void assertDatabaseExistsUnlocked(const String & database_name) const;
    void assertDatabaseDoesntExistUnlocked(const String & database_name) const;

    struct UUIDToStorageMapPart
    {
        std::unordered_map<UUID, DatabaseAndTable> map;
        mutable std::mutex mutex;
    };

    static constexpr UInt64 bits_for_first_level = 8;
    using UUIDToStorageMap = std::array<UUIDToStorageMapPart, 1ull << bits_for_first_level>;

    inline size_t getFirstLevelIdx(const UUID & uuid) const
    {
        return uuid.toUnderType().low >> (64 - bits_for_first_level);
    }

private:
    //[[maybe_unused]] Context & global_context;
    mutable std::mutex databases_mutex;
    //const String default_database;
    Databases databases;
    UUIDToStorageMap uuid_map;


    /// Do not allow simultaneous execution of DDL requests on the same table.
    /// database -> object -> (mutex, counter), counter: how many threads are running a query on the table at the same time
    /// For the duration of the operation, an element is placed here, and an object is returned,
    /// which deletes the element in the destructor when counter becomes zero.
    /// In case the element already exists, waits, when query will be executed in other thread. See class DDLGuard below.
    using DDLGuards = std::unordered_map<String, DDLGuard::Map>;
    DDLGuards ddl_guards;
    /// If you capture mutex and ddl_guards_mutex, then you need to grab them strictly in this order.
    mutable std::mutex ddl_guards_mutex;
};

}
