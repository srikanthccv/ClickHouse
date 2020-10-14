#include <Storages/StorageURL.h>
#include <Storages/ColumnsDescription.h>
#include <Access/AccessFlags.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/TableFunctionURL.h>
#include <Poco/URI.h>
#include "registerTableFunctions.h"


namespace DB
{
StoragePtr TableFunctionURL::getStorage(
    const String & source, const String & format_, const ColumnsDescription & columns, Context & global_context,
    const std::string & table_name, const String & compression_method_) const
{
    Poco::URI uri(source);
    return StorageURL::create(uri, StorageID(getDatabaseName(), table_name), format_, columns, ConstraintsDescription{},
            global_context, compression_method_);
}

void registerTableFunctionURL(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionURL>();
}
}
