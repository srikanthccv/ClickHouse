#include <DataTypes/NestedUtils.h>
#include <DataTypes/DataTypeArray.h>
#include <Common/escapeForFileName.h>
#include <Compression/CachedCompressedReadBuffer.h>
#include <Columns/ColumnArray.h>
#include <Interpreters/inplaceBlockConversions.h>
#include <Interpreters/Context.h>
#include <Storages/MergeTree/IMergeTreeReader.h>
#include <Common/typeid_cast.h>

namespace DB
{

namespace
{
    using OffsetColumns = std::map<std::string, ColumnPtr>;
}
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}


IMergeTreeReader::IMergeTreeReader(
    MergeTreeDataPartInfoForReaderPtr data_part_info_for_read_,
    const NamesAndTypesList & columns_,
    const StorageMetadataPtr & metadata_snapshot_,
    UncompressedCache * uncompressed_cache_,
    MarkCache * mark_cache_,
    const MarkRanges & all_mark_ranges_,
    const MergeTreeReaderSettings & settings_,
    const ValueSizeMap & avg_value_size_hints_)
    : data_part_info_for_read(data_part_info_for_read_)
    , avg_value_size_hints(avg_value_size_hints_)
    , uncompressed_cache(uncompressed_cache_)
    , mark_cache(mark_cache_)
    , settings(settings_)
    , metadata_snapshot(metadata_snapshot_)
    , all_mark_ranges(all_mark_ranges_)
    , alter_conversions(data_part_info_for_read->getAlterConversions())
    /// For wide parts convert plain arrays of Nested to subcolumns
    /// to allow to use shared offset column from cache.
    , requested_columns(data_part_info_for_read->isWidePart() ? Nested::convertToSubcolumns(columns_) : columns_)
    , part_columns(data_part_info_for_read->isWidePart() ? Nested::collect(data_part_info_for_read->getColumns()) : data_part_info_for_read->getColumns())
    , last_read_end_offset(0)
    , part_offset_column_index(-1)
{
    columns_to_read.reserve(requested_columns.size());
    serializations.reserve(requested_columns.size());

    ssize_t i = 0;
    for (const auto & column : requested_columns)
    {
        if (column.name == "_part_offset")
        {
            assert(part_offset_column_index == -1);
            part_offset_column_index = i;
        }
        else
        {
            columns_to_read.emplace_back(getColumnInPart(column));
            serializations.emplace_back(getSerializationInPart(column));
        }
        ++i;
    }
}


size_t IMergeTreeReader::readRows(size_t from_mark, size_t current_task_last_mark,
    bool continue_reading, size_t max_rows_to_read, Columns & res_columns)
{
//    Columns * physical_columns_to_fill = &res_columns;
//    Columns tmp_columns;
//    if (part_offset_column_index != -1)
//    {
//        tmp_columns.reserve(res_columns.size()-1);
//        tmp_columns.assign(res_columns.begin(), res_columns.begin() + part_offset_column_index);
//        tmp_columns.insert(tmp_columns.end(), res_columns.begin() + part_offset_column_index + 1, res_columns.end());
//        physical_columns_to_fill = &tmp_columns;
//    }

    ColumnPtr part_offset_column_before;
    if (part_offset_column_index != -1)
    {
        part_offset_column_before = res_columns[part_offset_column_index];
        res_columns.erase(res_columns.begin() + part_offset_column_index);
    }

    const size_t rows_read = readPhysicalRows(from_mark, current_task_last_mark, continue_reading, max_rows_to_read, res_columns/* *physical_columns_to_fill*/);
    const size_t start_row = continue_reading ? last_read_end_offset : data_part_info_for_read->getIndexGranularity().getMarkStartingRow(from_mark);
    const size_t end_row = start_row + rows_read;
    last_read_end_offset = end_row;

//    std::cerr << "READ from " << start_row << " to " << end_row << "\n\n\n";


    if (part_offset_column_index != -1)
    {
        MutableColumnPtr part_offset_column = part_offset_column_before ?
            part_offset_column_before->assumeMutable()
            : ColumnUInt64::create()->getPtr();
        part_offset_column->reserve(part_offset_column->size() + rows_read);

        // TODO: fill the column with int values efficiently
        for (size_t row_offset = start_row; row_offset < end_row; ++ row_offset)
            part_offset_column->insert(row_offset);

        res_columns.insert(res_columns.begin() + part_offset_column_index, part_offset_column->getPtr());
    }

//    if (part_offset_column_index != -1)
//    {
//        MutableColumnPtr part_offset_column = res_columns[part_offset_column_index] ?
//            res_columns[part_offset_column_index]->assumeMutable()
//            : MutableColumnPtr(ColumnUInt64::create());
//        part_offset_column->reserve(part_offset_column->size() + rows_read);
//
//        // TODO: fill the column with int values efficiently
//        for (size_t row_offset = start_row; row_offset < end_row; ++ row_offset)
//            part_offset_column->insert(row_offset);
//
//        res_columns.clear();
//        res_columns.reserve(tmp_columns.size() + 1);
//        res_columns.assign(tmp_columns.begin(), tmp_columns.begin() + part_offset_column_index);
//        res_columns.push_back(part_offset_column->getPtr());
//        res_columns.insert(res_columns.end(), tmp_columns.begin() + part_offset_column_index, tmp_columns.end());
//    }

    return rows_read;
}

const IMergeTreeReader::ValueSizeMap & IMergeTreeReader::getAvgValueSizeHints() const
{
    return avg_value_size_hints;
}

void IMergeTreeReader::fillMissingColumns(Columns & res_columns, bool & should_evaluate_missing_defaults, size_t num_rows) const
{
    try
    {
        NamesAndTypesList available_columns(columns_to_read.begin(), columns_to_read.end());
        DB::fillMissingColumns(
            res_columns, num_rows,
            Nested::convertToSubcolumns(requested_columns),
            Nested::convertToSubcolumns(available_columns),
            partially_read_columns, metadata_snapshot);

        should_evaluate_missing_defaults = std::any_of(
            res_columns.begin(), res_columns.end(), [](const auto & column) { return column == nullptr; });
    }
    catch (Exception & e)
    {
        /// Better diagnostics.
        e.addMessage("(while reading from part " + data_part_info_for_read->getDataPartStorage()->getFullPath() + ")");
        throw;
    }
}

void IMergeTreeReader::evaluateMissingDefaults(Block additional_columns, Columns & res_columns) const
{
    try
    {
        size_t num_columns = requested_columns.size();

        if (res_columns.size() != num_columns)
            throw Exception("invalid number of columns passed to MergeTreeReader::fillMissingColumns. "
                            "Expected " + toString(num_columns) + ", "
                            "got " + toString(res_columns.size()), ErrorCodes::LOGICAL_ERROR);

        /// Convert columns list to block.
        /// TODO: rewrite with columns interface. It will be possible after changes in ExpressionActions.
        auto name_and_type = requested_columns.begin();
        for (size_t pos = 0; pos < num_columns; ++pos, ++name_and_type)
        {
            if (res_columns[pos] == nullptr)
                continue;

            additional_columns.insert({res_columns[pos], name_and_type->type, name_and_type->name});
        }

        auto dag = DB::evaluateMissingDefaults(
                additional_columns, requested_columns, metadata_snapshot->getColumns(), data_part_info_for_read->getContext());
        if (dag)
        {
            dag->addMaterializingOutputActions();
            auto actions = std::make_shared<
                ExpressionActions>(std::move(dag),
                ExpressionActionsSettings::fromSettings(data_part_info_for_read->getContext()->getSettingsRef()));
            actions->execute(additional_columns);
        }

        /// Move columns from block.
        name_and_type = requested_columns.begin();
        for (size_t pos = 0; pos < num_columns; ++pos, ++name_and_type)
            res_columns[pos] = std::move(additional_columns.getByName(name_and_type->name).column);
    }
    catch (Exception & e)
    {
        /// Better diagnostics.
        e.addMessage("(while reading from part " + data_part_info_for_read->getDataPartStorage()->getFullPath() + ")");
        throw;
    }
}

String IMergeTreeReader::getColumnNameInPart(const NameAndTypePair & required_column) const
{
    auto name_in_storage = required_column.getNameInStorage();
    if (alter_conversions.isColumnRenamed(name_in_storage))
    {
        name_in_storage = alter_conversions.getColumnOldName(name_in_storage);
        return Nested::concatenateName(name_in_storage, required_column.getSubcolumnName());
    }

    return required_column.name;
}

NameAndTypePair IMergeTreeReader::getColumnInPart(const NameAndTypePair & required_column) const
{
    auto name_in_part = getColumnNameInPart(required_column);
    auto column_in_part = part_columns.tryGetColumnOrSubcolumn(GetColumnsOptions::AllPhysical, name_in_part);
    if (column_in_part)
        return *column_in_part;

    return required_column;
}

SerializationPtr IMergeTreeReader::getSerializationInPart(const NameAndTypePair & required_column) const
{
    auto name_in_part = getColumnNameInPart(required_column);
    auto column_in_part = part_columns.tryGetColumnOrSubcolumn(GetColumnsOptions::AllPhysical, name_in_part);
    if (!column_in_part)
        return IDataType::getSerialization(required_column);

    const auto & infos = data_part_info_for_read->getSerializationInfos();
    if (auto it = infos.find(column_in_part->getNameInStorage()); it != infos.end())
        return IDataType::getSerialization(*column_in_part, *it->second);

    return IDataType::getSerialization(*column_in_part);
}

void IMergeTreeReader::performRequiredConversions(Columns & res_columns) const
{
    try
    {
        size_t num_columns = requested_columns.size();

        if (res_columns.size() != num_columns)
        {
            throw Exception(
                "Invalid number of columns passed to MergeTreeReader::performRequiredConversions. "
                "Expected "
                    + toString(num_columns)
                    + ", "
                      "got "
                    + toString(res_columns.size()),
                ErrorCodes::LOGICAL_ERROR);
        }

        Block copy_block;
        auto name_and_type = requested_columns.begin();

        for (size_t pos = 0; pos < num_columns; ++pos, ++name_and_type)
        {
            if (res_columns[pos] == nullptr)
                continue;

            copy_block.insert({res_columns[pos], getColumnInPart(*name_and_type).type, name_and_type->name});
        }

        DB::performRequiredConversions(copy_block, requested_columns, data_part_info_for_read->getContext());

        /// Move columns from block.
        name_and_type = requested_columns.begin();
        for (size_t pos = 0; pos < num_columns; ++pos, ++name_and_type)
            res_columns[pos] = std::move(copy_block.getByName(name_and_type->name).column);
    }
    catch (Exception & e)
    {
        /// Better diagnostics.
        e.addMessage("(while reading from part " + data_part_info_for_read->getDataPartStorage()->getFullPath() + ")");
        throw;
    }
}

IMergeTreeReader::ColumnPosition IMergeTreeReader::findColumnForOffsets(const NameAndTypePair & required_column) const
{
    auto get_offsets_streams = [](const auto & serialization, const auto & name_in_storage)
    {
        Names offsets_streams;
        serialization->enumerateStreams([&](const auto & subpath)
        {
            if (subpath.empty() || subpath.back().type != ISerialization::Substream::ArraySizes)
                return;

            auto subname = ISerialization::getSubcolumnNameForStream(subpath);
            auto full_name = Nested::concatenateName(name_in_storage, subname);
            offsets_streams.push_back(full_name);
        });

        return offsets_streams;
    };

    auto required_name_in_storage = Nested::extractTableName(required_column.getNameInStorage());
    auto required_offsets_streams = get_offsets_streams(getSerializationInPart(required_column), required_name_in_storage);

    size_t max_matched_streams = 0;
    ColumnPosition position;

    /// Find column that has maximal number of matching
    /// offsets columns with required_column.
    for (const auto & part_column : data_part_info_for_read->getColumns())
    {
        auto name_in_storage = Nested::extractTableName(part_column.name);
        if (name_in_storage != required_name_in_storage)
            continue;

        auto offsets_streams = get_offsets_streams(data_part_info_for_read->getSerialization(part_column), name_in_storage);
        NameSet offsets_streams_set(offsets_streams.begin(), offsets_streams.end());

        size_t i = 0;
        for (; i < required_offsets_streams.size(); ++i)
        {
            if (!offsets_streams_set.contains(required_offsets_streams[i]))
                break;
        }

        if (i && (!position || i > max_matched_streams))
        {
            max_matched_streams = i;
            position = data_part_info_for_read->getColumnPosition(part_column.name);
        }
    }

    return position;
}

void IMergeTreeReader::checkNumberOfColumns(size_t num_columns_to_read) const
{
    if (num_columns_to_read != columns_to_read.size())//requested_columns.size())
        throw Exception("invalid number of columns passed to MergeTreeReader::readRows. "
                        "Expected " + toString(requested_columns.size()) + ", "
                        "got " + toString(num_columns_to_read), ErrorCodes::LOGICAL_ERROR);
}

}
