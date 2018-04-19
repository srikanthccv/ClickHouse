#include <Columns/ColumnString.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataStreams/OneBlockInputStream.h>
#include <Interpreters/Settings.h>
#include <Storages/System/StorageSystemBuildOptions.h>
#include <Common/config_build.h>

namespace DB
{


StorageSystemBuildOptions::StorageSystemBuildOptions(const std::string & name_)
    : name(name_)
{
    setColumns(ColumnsDescription({
        { "name", std::make_shared<DataTypeString>() },
        { "value", std::make_shared<DataTypeString>() },
    }));
}


BlockInputStreams StorageSystemBuildOptions::read(
    const Names & column_names,
    const SelectQueryInfo &,
    const Context & context,
    QueryProcessingStage::Enum processed_stage,
    const size_t /*max_block_size*/,
    const unsigned /*num_streams*/)
{
    check(column_names);
    checkQueryProcessingStage(processed_stage, context);

    MutableColumns res_columns = getSampleBlock().cloneEmptyColumns();

    for (auto it = auto_config_build; *it; it += 2)
    {
        res_columns[0]->insert(String(it[0]));
        res_columns[1]->insert(String(it[1]));
    }

    return BlockInputStreams(1, std::make_shared<OneBlockInputStream>(getSampleBlock().cloneWithColumns(std::move(res_columns))));
}


}
