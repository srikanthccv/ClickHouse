#include <iomanip>

#include <Core/Settings.h>
#include <Databases/DatabaseMemory.h>
#include <Databases/DatabaseLazy.h>
#include <Databases/DatabasesCommon.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteBufferFromFile.h>
#include <IO/WriteHelpers.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterCreateQuery.h>
#include <Parsers/ASTCreateQuery.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/parseQuery.h>
#include <Storages/IStorage.h>

#include <Poco/DirectoryIterator.h>
#include <Poco/Event.h>
#include <Common/Stopwatch.h>
#include <Common/StringUtils/StringUtils.h>
#include <Common/ThreadPool.h>
#include <Common/escapeForFileName.h>
#include <Common/typeid_cast.h>
#include <common/logger_useful.h>
#include <ext/scope_guard.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int TABLE_ALREADY_EXISTS;
    extern const int UNKNOWN_TABLE;
    extern const int UNSUPPORTED_METHOD;
    extern const int CANNOT_CREATE_TABLE_FROM_METADATA;
    extern const int INCORRECT_FILE_NAME;
    extern const int FILE_DOESNT_EXIST;
    extern const int LOGICAL_ERROR;
    extern const int CANNOT_GET_CREATE_TABLE_QUERY;
    extern const int SYNTAX_ERROR;
}


static constexpr size_t PRINT_MESSAGE_EACH_N_TABLES = 256;
static constexpr size_t PRINT_MESSAGE_EACH_N_SECONDS = 5;
static constexpr size_t METADATA_FILE_BUFFER_SIZE = 32768;

namespace detail
{
    String getTableMetadataPath(const String & base_path, const String & table_name)
    {
        return base_path + (endsWith(base_path, "/") ? "" : "/") + escapeForFileName(table_name) + ".sql";
    }

    String getDatabaseMetadataPath(const String & base_path)
    {
        return (endsWith(base_path, "/") ? base_path.substr(0, base_path.size() - 1) : base_path) + ".sql";
    }

}

static void loadTable(
    Context & context,
    const String & database_metadata_path,
    DatabaseLazy & database,
    const String & database_name,
    const String & database_data_path,
    const String & file_name,
    bool has_force_restore_data_flag)
{
    Logger * log = &Logger::get("loadTable");

    const String table_metadata_path = database_metadata_path + "/" + file_name;

    String s;
    {
        char in_buf[METADATA_FILE_BUFFER_SIZE];
        ReadBufferFromFile in(table_metadata_path, METADATA_FILE_BUFFER_SIZE, -1, in_buf);
        readStringUntilEOF(s, in);
    }

    /** Empty files with metadata are generated after a rough restart of the server.
      * Remove these files to slightly reduce the work of the admins on startup.
      */
    if (s.empty())
    {
        LOG_ERROR(log, "File " << table_metadata_path << " is empty. Removing.");
        Poco::File(table_metadata_path).remove();
        return;
    }

    try
    {
        String table_name;
        StoragePtr table;
        std::tie(table_name, table) = createTableFromDefinition(
            s, database_name, database_data_path, context, has_force_restore_data_flag, "in file " + table_metadata_path);
        database.attachTable(table_name, table);
    }
    catch (const Exception & e)
    {
        throw Exception("Cannot create table from metadata file " + table_metadata_path + ", error: " + e.displayText() +
            ", stack trace:\n" + e.getStackTrace().toString(),
            ErrorCodes::CANNOT_CREATE_TABLE_FROM_METADATA);
    }
}


DatabaseLazy::DatabaseLazy(String name_, const String & metadata_path_, const Context & context)
    : DatabaseWithOwnTablesBase(std::move(name_))
    , metadata_path(metadata_path_)
    , data_path(context.getPath() + "data/" + escapeForFileName(name) + "/")
    , log(&Logger::get("DatabaseLazy (" + name + ")"))
{
    Poco::File(data_path).createDirectories();
}


void DatabaseLazy::loadTables(
    Context & /* context */,
    bool /* has_force_restore_data_flag */)
{
}


void DatabaseLazy::createTable(
    const Context & context,
    const String & table_name,
    const StoragePtr & table,
    const ASTPtr & query)
{
    const auto & settings = context.getSettingsRef();

    /// Create a file with metadata if necessary - if the query is not ATTACH.
    /// Write the query of `ATTACH table` to it.

    /** The code is based on the assumption that all threads share the same order of operations
      * - creating the .sql.tmp file;
      * - adding a table to `tables`;
      * - rename .sql.tmp to .sql.
      */

    /// A race condition would be possible if a table with the same name is simultaneously created using CREATE and using ATTACH.
    /// But there is protection from it - see using DDLGuard in InterpreterCreateQuery.

    if (isTableExist(context, table_name))
        throw Exception("Table " + name + "." + table_name + " already exists.", ErrorCodes::TABLE_ALREADY_EXISTS);

    String table_metadata_path = getTableMetadataPath(table_name);
    String table_metadata_tmp_path = table_metadata_path + ".tmp";
    String statement;

    {
        statement = getTableDefinitionFromCreateQuery(query);

        /// Exclusive flags guarantees, that table is not created right now in another thread. Otherwise, exception will be thrown.
        WriteBufferFromFile out(table_metadata_tmp_path, statement.size(), O_WRONLY | O_CREAT | O_EXCL);
        writeString(statement, out);
        out.next();
        if (settings.fsync_metadata)
            out.sync();
        out.close();
    }

    try
    {
        /// Add a table to the map of known tables.
        attachTable(table_name, table);

        /// If it was ATTACH query and file with table metadata already exist
        /// (so, ATTACH is done after DETACH), then rename atomically replaces old file with new one.
        Poco::File(table_metadata_tmp_path).renameTo(table_metadata_path);
    }
    catch (...)
    {
        Poco::File(table_metadata_tmp_path).remove();
        throw;
    }
}


void DatabaseLazy::removeTable(
    const Context & /*context*/,
    const String & table_name)
{
    StoragePtr res = detachTable(table_name);

    String table_metadata_path = getTableMetadataPath(table_name);

    try
    {
        Poco::File(table_metadata_path).remove();
    }
    catch (...)
    {
        try
        {
            Poco::File(table_metadata_path + ".tmp_drop").remove();
            return;
        }
        catch (...)
        {
            LOG_WARNING(log, getCurrentExceptionMessage(__PRETTY_FUNCTION__));
        }
        attachTable(table_name, res);
        throw;
    }
}

static ASTPtr getQueryFromMetadata(const String & metadata_path, bool throw_on_error = true)
{
    String query;

    try
    {
        ReadBufferFromFile in(metadata_path, 4096);
        readStringUntilEOF(query, in);
    }
    catch (const Exception & e)
    {
        if (!throw_on_error && e.code() == ErrorCodes::FILE_DOESNT_EXIST)
            return nullptr;
        else
            throw;
    }

    ParserCreateQuery parser;
    const char * pos = query.data();
    std::string error_message;
    auto ast = tryParseQuery(parser, pos, pos + query.size(), error_message, /* hilite = */ false,
                             "in file " + metadata_path, /* allow_multi_statements = */ false, 0);

    if (!ast && throw_on_error)
        throw Exception(error_message, ErrorCodes::SYNTAX_ERROR);

    return ast;
}

static ASTPtr getCreateQueryFromMetadata(const String & metadata_path, const String & database, bool throw_on_error)
{
    ASTPtr ast = getQueryFromMetadata(metadata_path, throw_on_error);

    if (ast)
    {
        auto & ast_create_query = ast->as<ASTCreateQuery &>();
        ast_create_query.attach = false;
        ast_create_query.database = database;
    }

    return ast;
}


void DatabaseLazy::renameTable(
    const Context & context,
    const String & table_name,
    IDatabase & to_database,
    const String & to_table_name,
    TableStructureWriteLockHolder & lock)
{
    DatabaseLazy * to_database_concrete = typeid_cast<DatabaseLazy *>(&to_database);

    if (!to_database_concrete)
        throw Exception("Moving tables between databases of different engines is not supported", ErrorCodes::NOT_IMPLEMENTED);

    StoragePtr table = tryGetTable(context, table_name);

    if (!table)
        throw Exception("Table " + name + "." + table_name + " doesn't exist.", ErrorCodes::UNKNOWN_TABLE);

    /// Notify the table that it is renamed. If the table does not support renaming, exception is thrown.
    try
    {
        table->rename(context.getPath() + "/data/" + escapeForFileName(to_database_concrete->name) + "/",
            to_database_concrete->name,
            to_table_name, lock);
    }
    catch (const Exception &)
    {
        throw;
    }
    catch (const Poco::Exception & e)
    {
        /// Better diagnostics.
        throw Exception{Exception::CreateFromPoco, e};
    }

    ASTPtr ast = getQueryFromMetadata(detail::getTableMetadataPath(metadata_path, table_name));
    if (!ast)
        throw Exception("There is no metadata file for table " + table_name, ErrorCodes::FILE_DOESNT_EXIST);
    ast->as<ASTCreateQuery &>().table = to_table_name;

    /// NOTE Non-atomic.
    to_database_concrete->createTable(context, to_table_name, table, ast);
    removeTable(context, table_name);
}


time_t DatabaseLazy::getTableMetadataModificationTime(
    const Context & /*context*/,
    const String & table_name)
{
    String table_metadata_path = getTableMetadataPath(table_name);
    Poco::File meta_file(table_metadata_path);

    if (meta_file.exists())
    {
        return meta_file.getLastModified().epochTime();
    }
    else
    {
        return static_cast<time_t>(0);
    }
}

ASTPtr DatabaseLazy::getCreateTableQueryImpl(const Context & context,
                                                 const String & table_name, bool throw_on_error) const
{
    ASTPtr ast;

    auto table_metadata_path = detail::getTableMetadataPath(metadata_path, table_name);
    ast = getCreateQueryFromMetadata(table_metadata_path, name, throw_on_error);
    if (!ast && throw_on_error)
    {
        /// Handle system.* tables for which there are no table.sql files.
        bool has_table = tryGetTable(context, table_name) != nullptr;

        auto msg = has_table
                   ? "There is no CREATE TABLE query for table "
                   : "There is no metadata file for table ";

        throw Exception(msg + table_name, ErrorCodes::CANNOT_GET_CREATE_TABLE_QUERY);
    }

    return ast;
}

ASTPtr DatabaseLazy::getCreateTableQuery(const Context & context, const String & table_name) const
{
    return getCreateTableQueryImpl(context, table_name, true);
}

ASTPtr DatabaseLazy::tryGetCreateTableQuery(const Context & context, const String & table_name) const
{
    return getCreateTableQueryImpl(context, table_name, false);
}

ASTPtr DatabaseLazy::getCreateDatabaseQuery(const Context & /*context*/) const
{
    ASTPtr ast;

    auto database_metadata_path = detail::getDatabaseMetadataPath(metadata_path);
    ast = getCreateQueryFromMetadata(database_metadata_path, name, true);
    if (!ast)
    {
        /// Handle databases (such as default) for which there are no database.sql files.
        String query = "CREATE DATABASE " + backQuoteIfNeed(name) + " ENGINE = Lazy";
        ParserCreateQuery parser;
        ast = parseQuery(parser, query.data(), query.data() + query.size(), "", 0);
    }

    return ast;
}

void DatabaseLazy::alterTable(
    const Context & /* context */,
    const String & /* table_name */,
    const ColumnsDescription & /* columns */,
    const IndicesDescription & /* indices */,
    const ConstraintsDescription & /* constraints */,
    const ASTModifier & /* storage_modifier */)
{
    throw Exception("ALTER query is not supported for Lazy database.", ErrorCodes::UNSUPPORTED_METHOD);
}


void DatabaseLazy::drop()
{
    Poco::File(data_path).remove(false);
    Poco::File(metadata_path).remove(false);
}

bool DatabaseLazy::isTableExist(
    const Context & context,
    const String & table_name) const
{
    return DatabaseWithOwnTablesBase::isTableExist(context, table_name) || Poco::File(getTableMetadataPath(table_name)).exists();
}

StoragePtr DatabaseLazy::tryGetTable(
    const Context & context,
    const String & table_name) const
{
    auto storage_ptr = DatabaseWithOwnTablesBase::tryGetTable(context, table_name);
    if (storage_ptr)
        return storage_ptr;
    
    if (!Poco::File(getTableMetadataPath(table_name)).exists())
        return nullptr;
    loadTable(context, metadata_path, *this, name, data_path, table_name + ".sql", has_force_restore_data_flag);

    return tryGetTable(context, table_name);
}

DatabaseIteratorPtr DatabaseLazy::getIterator(const Context & /*context*/, const FilterByNameFunction & filter_by_table_name)
{
    std::lock_guard lock(mutex);
    return DatabaseLazyIterator(*this, filter_by_table_name);
}

bool DatabaseLazy::empty(const Context & context) const
{
    if (!DatabaseWithOwnTablesBase::empty(context))
        return true;
    return getIterator(context)->isValid();
}

String DatabaseLazy::getDataPath() const
{
    return data_path;
}

String DatabaseLazy::getMetadataPath() const
{
    return metadata_path;
}

String DatabaseLazy::getDatabaseName() const
{
    return name;
}

String DatabaseLazy::getTableMetadataPath(const String & table_name) const
{
    return detail::getTableMetadataPath(metadata_path, table_name);
}

DatabaseLazyIterator::DatabaseLazyIterator(DatabaseLazy & database_, bool has_force_restore_data_flag_)
    : database(database_)
    , has_force_restore_data_flag(has_force_restore_data_flag_)
    , dir_it(database.metadata_path)
{
    moveToTable();
}

void DatabaseLazyIterator::moveToTable()
{
    Poco::DirectoryIterator dir_end;
    for (; dir_it != dir_end; ++dir_it)
    {
        /// For '.svn', '.gitignore' directory and similar.
        if (dir_it.name().at(0) == '.')
            continue;

        /// There are .sql.bak files - skip them.
        if (endsWith(dir_it.name(), ".sql.bak"))
            continue;

        // There are files that we tried to delete previously
        static const char * tmp_drop_ext = ".sql.tmp_drop";
        if (endsWith(dir_it.name(), tmp_drop_ext))
        {
            const std::string table_name = dir_it.name().substr(0, dir_it.name().size() - strlen(tmp_drop_ext));
            if (Poco::File(data_path + '/' + table_name).exists())
            {
                Poco::File(dir_it->path()).renameTo(table_name + ".sql");
                LOG_WARNING(log, "Table " << backQuote(table_name) << " was not dropped previously");
            }
            else
            {
                LOG_INFO(log, "Removing file " << dir_it->path());
                Poco::File(dir_it->path()).remove();
            }
            continue;
        }

        /// There are files .sql.tmp - delete
        if (endsWith(dir_it.name(), ".sql.tmp"))
        {
            LOG_INFO(log, "Removing file " << dir_it->path());
            Poco::File(dir_it->path()).remove();
            continue;
        }

        /// The required files have names like `table_name.sql`
        if (endsWith(dir_it.name(), ".sql"))
            return;
        else
            throw Exception("Incorrect file extension: " + dir_it.name() + " in metadata directory " + metadata_path,
                ErrorCodes::INCORRECT_FILE_NAME);
    }
}

void DatabaseLazyIterator::next()
{
    ++dir_it;
    moveToTable();
}

bool DatabaseLazyIterator::isValid() const
{
    return dir_it != Poco::DirectoryIterator{};
}

const String & DatabaseLazyIterator::name() const
{
    /// cut .sql
    const std::string & table_file = dir_it.name();
    return table_file.substr(0, table_file.size() - 4);
}

StoragePtr & table() const
{
    loadTable(context, database.metadata_path, database, database.name, database.data_path, table, has_force_restore_data_flag);
    return database.tryGetTable(name());
}

}
