#include <Storages/IStorage.h>

#include <Storages/AlterCommands.h>

#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>


namespace DB
{

namespace ErrorCodes
{
    extern const int COLUMN_QUERIED_MORE_THAN_ONCE;
    extern const int DUPLICATE_COLUMN;
    extern const int EMPTY_LIST_OF_COLUMNS_PASSED;
    extern const int EMPTY_LIST_OF_COLUMNS_QUERIED;
    extern const int NO_SUCH_COLUMN_IN_TABLE;
    extern const int NOT_FOUND_COLUMN_IN_BLOCK;
    extern const int TYPE_MISMATCH;
}

IStorage::IStorage(ColumnsDescription columns_)
{
    setColumns(std::move(columns_));
}

const ColumnsDescription & IStorage::getColumns() const
{
    return columns;
}

void IStorage::setColumns(ColumnsDescription columns_)
{
    if (columns_.getOrdinary().empty())
        throw Exception("Empty list of columns passed", ErrorCodes::EMPTY_LIST_OF_COLUMNS_PASSED);
    columns = std::move(columns_);
}

const IndicesDescription & IStorage::getIndices() const
{
    return indices;
}

void IStorage::setIndices(IndicesDescription indices_)
{
    indices = std::move(indices_);
}

NameAndTypePair IStorage::getColumn(const String & column_name) const
{
    /// By default, we assume that there are no virtual columns in the storage.
    return getColumns().getPhysical(column_name);
}

bool IStorage::hasColumn(const String & column_name) const
{
    /// By default, we assume that there are no virtual columns in the storage.
    return getColumns().hasPhysical(column_name);
}

Block IStorage::getSampleBlock() const
{
    Block res;

    for (const auto & column : getColumns().getAllPhysical())
        res.insert({column.type->createColumn(), column.type, column.name});

    return res;
}

Block IStorage::getSampleBlockNonMaterialized() const
{
    Block res;

    for (const auto & column : getColumns().getOrdinary())
        res.insert({column.type->createColumn(), column.type, column.name});

    return res;
}

Block IStorage::getSampleBlockForColumns(const Names & column_names) const
{
    Block res;

    NamesAndTypesList all_columns = getColumns().getAll();
    std::unordered_map<String, DataTypePtr> columns_map;
    for (const auto & elem : all_columns)
        columns_map.emplace(elem.name, elem.type);

    for (const auto & name : column_names)
    {
        auto it = columns_map.find(name);
        if (it != columns_map.end())
        {
            res.insert({it->second->createColumn(), it->second, it->first});
        }
        else
        {
            /// Virtual columns.
            NameAndTypePair elem = getColumn(name);
            res.insert({elem.type->createColumn(), elem.type, elem.name});
        }
    }

    return res;
}

namespace
{
    using NamesAndTypesMap = GOOGLE_NAMESPACE::dense_hash_map<StringRef, const IDataType *, StringRefHash>;
    using UniqueStrings = GOOGLE_NAMESPACE::dense_hash_set<StringRef, StringRefHash>;

    String listOfColumns(const NamesAndTypesList & available_columns)
    {
        std::stringstream ss;
        for (auto it = available_columns.begin(); it != available_columns.end(); ++it)
        {
            if (it != available_columns.begin())
                ss << ", ";
            ss << it->name;
        }
        return ss.str();
    }

    NamesAndTypesMap getColumnsMap(const NamesAndTypesList & columns)
    {
        NamesAndTypesMap res;
        res.set_empty_key(StringRef());

        for (const auto & column : columns)
            res.insert({column.name, column.type.get()});

        return res;
    }

    UniqueStrings initUniqueStrings()
    {
        UniqueStrings strings;
        strings.set_empty_key(StringRef());
        return strings;
    }
}

void IStorage::check(const Names & column_names) const
{
    const NamesAndTypesList & available_columns = getColumns().getAllPhysical();
    const String list_of_columns = listOfColumns(available_columns);

    if (column_names.empty())
        throw Exception("Empty list of columns queried. There are columns: " + list_of_columns, ErrorCodes::EMPTY_LIST_OF_COLUMNS_QUERIED);

    const auto columns_map = getColumnsMap(available_columns);

    auto unique_names = initUniqueStrings();
    for (const auto & name : column_names)
    {
        if (columns_map.end() == columns_map.find(name))
            throw Exception(
                "There is no column with name " + name + " in table. There are columns: " + list_of_columns,
                ErrorCodes::NO_SUCH_COLUMN_IN_TABLE);

        if (unique_names.end() != unique_names.find(name))
            throw Exception("Column " + name + " queried more than once", ErrorCodes::COLUMN_QUERIED_MORE_THAN_ONCE);
        unique_names.insert(name);
    }
}

void IStorage::check(const NamesAndTypesList & provided_columns) const
{
    const NamesAndTypesList & available_columns = getColumns().getAllPhysical();
    const auto columns_map = getColumnsMap(available_columns);

    auto unique_names = initUniqueStrings();
    for (const NameAndTypePair & column : provided_columns)
    {
        auto it = columns_map.find(column.name);
        if (columns_map.end() == it)
            throw Exception(
                "There is no column with name " + column.name + ". There are columns: " + listOfColumns(available_columns),
                ErrorCodes::NO_SUCH_COLUMN_IN_TABLE);

        if (!column.type->equals(*it->second))
            throw Exception(
                "Type mismatch for column " + column.name + ". Column has type " + it->second->getName() + ", got type "
                    + column.type->getName(),
                ErrorCodes::TYPE_MISMATCH);

        if (unique_names.end() != unique_names.find(column.name))
            throw Exception("Column " + column.name + " queried more than once", ErrorCodes::COLUMN_QUERIED_MORE_THAN_ONCE);
        unique_names.insert(column.name);
    }
}

void IStorage::check(const NamesAndTypesList & provided_columns, const Names & column_names) const
{
    const NamesAndTypesList & available_columns = getColumns().getAllPhysical();
    const auto available_columns_map = getColumnsMap(available_columns);
    const auto & provided_columns_map = getColumnsMap(provided_columns);

    if (column_names.empty())
        throw Exception(
            "Empty list of columns queried. There are columns: " + listOfColumns(available_columns),
            ErrorCodes::EMPTY_LIST_OF_COLUMNS_QUERIED);

    auto unique_names = initUniqueStrings();
    for (const String & name : column_names)
    {
        auto it = provided_columns_map.find(name);
        if (provided_columns_map.end() == it)
            continue;

        auto jt = available_columns_map.find(name);
        if (available_columns_map.end() == jt)
            throw Exception(
                "There is no column with name " + name + ". There are columns: " + listOfColumns(available_columns),
                ErrorCodes::NO_SUCH_COLUMN_IN_TABLE);

        if (!it->second->equals(*jt->second))
            throw Exception(
                "Type mismatch for column " + name + ". Column has type " + jt->second->getName() + ", got type " + it->second->getName(),
                ErrorCodes::TYPE_MISMATCH);

        if (unique_names.end() != unique_names.find(name))
            throw Exception("Column " + name + " queried more than once", ErrorCodes::COLUMN_QUERIED_MORE_THAN_ONCE);
        unique_names.insert(name);
    }
}

void IStorage::check(const Block & block, bool need_all) const
{
    const NamesAndTypesList & available_columns = getColumns().getAllPhysical();
    const auto columns_map = getColumnsMap(available_columns);

    NameSet names_in_block;

    block.checkNumberOfRows();

    for (const auto & column : block)
    {
        if (names_in_block.count(column.name))
            throw Exception("Duplicate column " + column.name + " in block", ErrorCodes::DUPLICATE_COLUMN);

        names_in_block.insert(column.name);

        auto it = columns_map.find(column.name);
        if (columns_map.end() == it)
            throw Exception(
                "There is no column with name " + column.name + ". There are columns: " + listOfColumns(available_columns),
                ErrorCodes::NO_SUCH_COLUMN_IN_TABLE);

        if (!column.type->equals(*it->second))
            throw Exception(
                "Type mismatch for column " + column.name + ". Column has type " + it->second->getName() + ", got type "
                    + column.type->getName(),
                ErrorCodes::TYPE_MISMATCH);
    }

    if (need_all && names_in_block.size() < columns_map.size())
    {
        for (auto it = available_columns.begin(); it != available_columns.end(); ++it)
        {
            if (!names_in_block.count(it->name))
                throw Exception("Expected column " + it->name, ErrorCodes::NOT_FOUND_COLUMN_IN_BLOCK);
        }
    }
}

void IStorage::alter(
    const AlterCommands & params,
    const String & database_name,
    const String & table_name,
    const Context & context,
    TableStructureWriteLockHolder & table_lock_holder)
{
    for (const auto & param : params)
    {
        if (param.isMutable())
            throw Exception("Method alter supports only change comment of column for storage " + getName(), ErrorCodes::NOT_IMPLEMENTED);
    }

    lockStructureExclusively(table_lock_holder, context.getCurrentQueryId());
    auto new_columns = getColumns();
    auto new_indices = getIndices();
    params.apply(new_columns);
    context.getDatabase(database_name)->alterTable(context, table_name, new_columns, new_indices, {});
    setColumns(std::move(new_columns));
}
}
