#pragma once

#include <mutex>
#include <unordered_set>
#include <Databases/DatabasesCommon.h>
#include <Databases/IDatabase.h>
#include <Storages/IStorage_fwd.h>


namespace Poco
{
    class Logger;
}


namespace DB
{

/* Database to store StorageDictionary tables
 * automatically creates tables for all dictionaries
 */
class DatabaseDictionary final : public IDatabase
{
public:
    DatabaseDictionary(const String & name_, const Context & global_context);

    String getEngineName() const override
    {
        return "Dictionary";
    }

    bool isTableExist(
        const Context & context,
        const String & table_name) const override;

    StoragePtr tryGetTable(
        const Context & context,
        const String & table_name) const override;

    DatabaseTablesIteratorPtr getTablesIterator(const FilterByNameFunction & filter_by_table_name) override;

    bool empty(const Context & context) const override;

    ASTPtr getCreateDatabaseQuery(const Context & context) const override;

    bool shouldBeEmptyOnDetach() const override { return false; }

    void shutdown() override;

protected:
    ASTPtr getCreateTableQueryImpl(const Context & context, const String & table_name, bool throw_on_error) const override;

private:
    mutable std::mutex mutex;

    Poco::Logger * log;
    const Context & global_context;

    Tables listTables(const FilterByNameFunction & filter_by_name);
};

}
