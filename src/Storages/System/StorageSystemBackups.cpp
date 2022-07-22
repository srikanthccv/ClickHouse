#include <Storages/System/StorageSystemBackups.h>
#include <Backups/BackupsWorker.h>
#include <DataTypes/DataTypeEnum.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeUUID.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Interpreters/Context.h>


namespace DB
{

NamesAndTypesList StorageSystemBackups::getNamesAndTypes()
{
    NamesAndTypesList names_and_types{
        {"id", std::make_shared<DataTypeString>()},
        {"name", std::make_shared<DataTypeString>()},
        {"status", std::make_shared<DataTypeEnum8>(getBackupStatusEnumValues())},
        {"status_changed_time", std::make_shared<DataTypeDateTime>()},
        {"error", std::make_shared<DataTypeString>()},
    };
    return names_and_types;
}


void StorageSystemBackups::fillData(MutableColumns & res_columns, ContextPtr context, const SelectQueryInfo &) const
{
    size_t column_index = 0;
    auto & column_id = assert_cast<ColumnString &>(*res_columns[column_index++]);
    auto & column_name = assert_cast<ColumnString &>(*res_columns[column_index++]);
    auto & column_status = assert_cast<ColumnInt8 &>(*res_columns[column_index++]);
    auto & column_status_changed_time = assert_cast<ColumnUInt32 &>(*res_columns[column_index++]);
    auto & column_error = assert_cast<ColumnString &>(*res_columns[column_index++]);

    auto add_row = [&](const BackupsWorker::Info & info)
    {
        column_id.insertData(info.id.data(), info.id.size());
        column_name.insertData(info.name.data(), info.name.size());
        column_status.insertValue(static_cast<Int8>(info.status));
        column_status_changed_time.insertValue(info.status_changed_time);
        column_error.insertData(info.error_message.data(), info.error_message.size());
    };

    for (const auto & entry : context->getBackupsWorker().getAllInfos())
        add_row(entry);
}

}
