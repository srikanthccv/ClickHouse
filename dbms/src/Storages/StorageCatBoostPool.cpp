#include <Storages/StorageCatBoostPool.h>
#include <DataStreams/IProfilingBlockInputStream.h>
#include <DataStreams/FormatFactory.h>
#include <IO/ReadBufferFromFile.h>
#include <fstream>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataStreams/FilterColumnsBlockInputStream.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_OPEN_FILE;
    extern const int CANNOT_PARSE_TEXT;
}

namespace
{
class CatBoostDatasetBlockInputStream : public IProfilingBlockInputStream
{
public:

    CatBoostDatasetBlockInputStream(const std::string & file_name, const std::string & format_name,
                                    const Block & sample_block, const Context & context, size_t max_block_size)
            : file_name(file_name), format_name(format_name)
    {
        read_buf = std::make_unique<ReadBufferFromFile>(file_name);
        reader = FormatFactory().getInput(format_name, *read_buf, sample_block, context, max_block_size);
    }

    String getName() const override
    {
        return "CatBoostDatasetBlockInputStream";
    }

    String getID() const override
    {
        return "CatBoostDataset(" + format_name + ", " + file_name + ")";
    }

    Block readImpl() override
    {
        return reader->read();
    }

    void readPrefixImpl() override
    {
        reader->readPrefix();
    }

    void readSuffixImpl() override
    {
        reader->readSuffix();
    }

private:
    Block sample_block;
    std::unique_ptr<ReadBufferFromFileDescriptor> read_buf;
    BlockInputStreamPtr reader;
    std::string file_name;
    std::string format_name;
};

}

StoragePtr StorageCatBoostPool::create(const String & column_description_file_name,
                                       const String & data_description_file_name)
{
    return make_shared(column_description_file_name, data_description_file_name);
}

StorageCatBoostPool::StorageCatBoostPool(const String & column_description_file_name,
                                         const String & data_description_file_name)
        : column_description_file_name(column_description_file_name),
          data_description_file_name(data_description_file_name)
{
    parseColumnDescription();
    createSampleBlockAndColumns();
}

std::string StorageCatBoostPool::getColumnTypesString(const ColumnTypesMap & columnTypesMap)
{
    std::string types_string;
    bool first = true;
    for (const auto & value : columnTypesMap)
    {
        if (!first)
            types_string.append(", ");

        first = false;
        types_string += value.first;
    }

    return types_string;
}

void StorageCatBoostPool::checkDatasetDescription()
{
    std::ifstream in(data_description_file_name);
    if (!in.good())
        throw Exception("Cannot open file: " + data_description_file_name, ErrorCodes::CANNOT_OPEN_FILE);

    std::string line;
    if (!std::getline(in, line))
        throw Exception("File is empty: " + data_description_file_name, ErrorCodes::CANNOT_PARSE_TEXT);

    size_t columns_count = 1;
    for (char sym : line)
        if (sym == '\t')
            ++columns_count;

    columns_description.resize(columns_count);
}

void StorageCatBoostPool::parseColumnDescription()
{
    /// NOTE: simple parsing
    /// TODO: use ReadBufferFromFile

    checkDatasetDescription();

    std::ifstream in(column_description_file_name);
    if (!in.good())
        throw Exception("Cannot open file: " + column_description_file_name, ErrorCodes::CANNOT_OPEN_FILE);

    std::string line;
    size_t line_num = 0;
    auto column_types_map = getColumnTypesMap();
    auto column_types_string = getColumnTypesString(column_types_map);

    while (std::getline(in, line))
    {
        ++line_num;
        std::string str_line_num = std::to_string(line_num);

        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(iss, token, '\t'))
            tokens.push_back(token);

        if (tokens.size() != 2 && tokens.size() != 3)
            throw Exception("Cannot parse column description at line " + str_line_num + " '" + line + "' "
                            + ": expected 2 or 3 columns, got " + std::to_string(tokens.size()),
                            ErrorCodes::CANNOT_PARSE_TEXT);

        std::string str_id = tokens[0];
        std::string col_type = tokens[1];
        std::string col_name = tokens.size() > 2 ? tokens[2] : str_id;

        size_t num_id;
        try
        {
            num_id = std::stoull(str_id);
        }
        catch (std::exception & e)
        {
            throw Exception("Cannot parse column index at row " + str_line_num + ": " + e.what(),
                            ErrorCodes::CANNOT_PARSE_TEXT);
        }

        if (num_id >= columns_description.size())
            throw Exception("Invalid index at row  " + str_line_num + ": " + str_id
                            + ", expected in range [0, " + std::to_string(columns_description.size()) + ")",
                            ErrorCodes::CANNOT_PARSE_TEXT);

        if (column_types_map.count(col_type) == 0)
            throw Exception("Invalid column type: " + col_type + ", expected: " + column_types_string,
                            ErrorCodes::CANNOT_PARSE_TEXT);

        auto type = column_types_map[col_type];
        if (type != DatasetColumnType::Num && type != DatasetColumnType::Categ)
            col_name = col_type;
        columns_description[num_id] = ColumnDescription(col_name, type);
    }
}

void StorageCatBoostPool::createSampleBlockAndColumns()
{
    columns.clear();
    NamesAndTypesList cat_columns;
    NamesAndTypesList num_columns;
    sample_block.clear();
    for (auto & desc : columns_description)
    {
        DataTypePtr type;
        if (desc.column_type == DatasetColumnType::Categ
            || desc.column_type == DatasetColumnType::Auxiliary
            || desc.column_type == DatasetColumnType::DocId)
            type = std::make_shared<DataTypeString>();
        else
            type = std::make_shared<DataTypeFloat64>();

        if (desc.column_type == DatasetColumnType::Categ)
            cat_columns.emplace_back(desc.column_name, type);
        if (desc.column_type == DatasetColumnType::Num)
            num_columns.emplace_back(desc.column_name, type);
        sample_block.insert(ColumnWithTypeAndName(type->createColumn(), type, desc.column_name));
    }
    columns.insert(columns.end(), num_columns.begin(), num_columns.end());
    columns.insert(columns.end(), cat_columns.begin(), cat_columns.end());
}

BlockInputStreams StorageCatBoostPool::read(const Names & column_names,
                       const SelectQueryInfo & query_info,
                       const Context & context,
                       QueryProcessingStage::Enum & processed_stage,
                       size_t max_block_size,
                       unsigned threads)
{
    auto stream = std::make_shared<CatBoostDatasetBlockInputStream>(
            data_description_file_name, "TSV", sample_block, context, max_block_size);

    auto filter_stream = std::make_shared<FilterColumnsBlockInputStream>(stream, column_names, false);
    return { filter_stream };
}

}
