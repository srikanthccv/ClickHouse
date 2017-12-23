#pragma once

#include <Core/NamesAndTypes.h>


namespace DB
{

namespace DataTypeNested
{
    std::string concatenateNestedName(const std::string & nested_table_name, const std::string & nested_field_name);
    /// Returns the prefix of the name to the first '.'. Or the name is unchanged if there is no dot.
    std::string extractNestedTableName(const std::string & nested_name);
    /// Returns the name suffix after the first dot on the right '.'. Or the name is unchanged if there is no dot.
    std::string extractNestedColumnName(const std::string & nested_name);

    /// Creates a new list in which Nested-type columns are replaced by several columns form of `column_name.cell_name`
    NamesAndTypesListPtr expandNestedColumns(const NamesAndTypesList & names_and_types);
};

}
