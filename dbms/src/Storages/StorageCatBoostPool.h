#pragma once

#include <Storages/IStorage.h>
#include <Core/Defines.h>
#include <common/MultiVersion.h>
#include <ext/shared_ptr_helper.h>

namespace DB
{

class StorageCatBoostPool : private ext::shared_ptr_helper<StorageCatBoostPool>, public IStorage
{
    friend class ext::shared_ptr_helper<StorageCatBoostPool>;

public:
    static StoragePtr create(const String & column_description_file_name, const String & data_description_file_name);

    std::string getName() const override { return "CatBoostPool"; }

    std::string getTableName() const override { return table_name; }

    const NamesAndTypesList & getColumnsListImpl() const override { return columns; }

    BlockInputStreams read(const Names & column_names,
                           const SelectQueryInfo & query_info,
                           const Context & context,
                           QueryProcessingStage::Enum & processed_stage,
                           size_t max_block_size,
                           unsigned threads) override;

private:
    String table_name;
    NamesAndTypesList columns;
    String column_description_file_name;
    String data_description_file_name;
    Block sample_block;

    enum class DatasetColumnType
    {
        Target,
        Num,
        Categ,
        Auxiliary,
        DocId,
        Weight,
        Baseline
    };

    using ColumnTypesMap = std::map<std::string, DatasetColumnType>;

    ColumnTypesMap getColumnTypesMap() const
    {
        return {
                {"Target", DatasetColumnType::Target},
                {"Num", DatasetColumnType::Num},
                {"Categ", DatasetColumnType::Categ},
                {"Auxiliary", DatasetColumnType::Auxiliary},
                {"DocId", DatasetColumnType::DocId},
                {"Weight", DatasetColumnType::Weight},
                {"Baseline", DatasetColumnType::Baseline},
        };
    };

    std::string getColumnTypesString(const ColumnTypesMap & columnTypesMap);

    struct ColumnDescription
    {
        std::string column_name;
        DatasetColumnType column_type;

        ColumnDescription() : column_type(DatasetColumnType::Num) {}
        ColumnDescription(const std::string & column_name, DatasetColumnType column_type)
                : column_name(column_name), column_type(column_type) {}
    };

    std::vector<ColumnDescription> columns_description;

    StorageCatBoostPool(const String & column_description_file_name, const String & data_description_file_name);

    void checkDatasetDescription();
    void parseColumnDescription();
    void createSampleBlockAndColumns();
};

}
