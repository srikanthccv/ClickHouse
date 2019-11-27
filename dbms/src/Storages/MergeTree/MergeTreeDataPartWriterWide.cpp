#include <Storages/MergeTree/MergeTreeDataPartWriterWide.h>

namespace DB
{

namespace
{
    constexpr auto DATA_FILE_EXTENSION = ".bin";
}

MergeTreeDataPartWriterWide::MergeTreeDataPartWriterWide(
    const String & part_path_,
    const MergeTreeData & storage_,
    const NamesAndTypesList & columns_list_,
    const std::vector<MergeTreeIndexPtr> & indices_to_recalc_, 
    const String & marks_file_extension_,
    const CompressionCodecPtr & default_codec_,
    const WriterSettings & settings_,
    const MergeTreeIndexGranularity & index_granularity_)
    : IMergeTreeDataPartWriter(part_path_,
        storage_, columns_list_, indices_to_recalc_,
        marks_file_extension_, default_codec_, settings_, index_granularity_, false)
    , can_use_adaptive_granularity(storage_.canUseAdaptiveGranularity())
{
    const auto & columns = storage.getColumns();
    for (const auto & it : columns_list)
        addStreams(it.name, *it.type, columns.getCodecOrDefault(it.name, default_codec), settings.estimated_size, false);
}

void MergeTreeDataPartWriterWide::addStreams(
    const String & name,
    const IDataType & type,
    const CompressionCodecPtr & effective_codec,
    size_t estimated_size,
    bool skip_offsets)
{
    IDataType::StreamCallback callback = [&] (const IDataType::SubstreamPath & substream_path)
    {
        if (skip_offsets && !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes)
            return;

        String stream_name = IDataType::getFileNameForStream(name, substream_path);

        /// Shared offsets for Nested type.
        if (column_streams.count(stream_name))
            return;

        column_streams[stream_name] = std::make_unique<ColumnStream>(
            stream_name,
            part_path + stream_name, DATA_FILE_EXTENSION,
            part_path + stream_name, marks_file_extension,
            effective_codec,
            settings.max_compress_block_size,
            estimated_size,
            settings.aio_threshold);
    };

    std::cerr << "(addStreams) name: " << name << "\n";

    IDataType::SubstreamPath stream_path;
    type.enumerateStreams(callback, stream_path);
}


IDataType::OutputStreamGetter MergeTreeDataPartWriterWide::createStreamGetter(
        const String & name, WrittenOffsetColumns & offset_columns, bool skip_offsets)
{
    return [&, skip_offsets] (const IDataType::SubstreamPath & substream_path) -> WriteBuffer *
    {
        bool is_offsets = !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes;
        if (is_offsets && skip_offsets)
            return nullptr;

        String stream_name = IDataType::getFileNameForStream(name, substream_path);

        /// Don't write offsets more than one time for Nested type.
        if (is_offsets && offset_columns.count(stream_name))
            return nullptr;

        return &column_streams[stream_name]->compressed;
    };
}

void MergeTreeDataPartWriterWide::write(const Block & block, 
    const IColumn::Permutation * permutation,
    const Block & primary_key_block, const Block & skip_indexes_block)
{
    // if (serialization_states.empty())
    // {
    //     serialization_states.reserve(columns_list.size());
    //     WrittenOffsetColumns tmp_offset_columns;
    //     IDataType::SerializeBinaryBulkSettings serialize_settings;

    //     for (const auto & col : columns_list)
    //     {
    //         serialize_settings.getter = createStreamGetter(col.name, tmp_offset_columns, false);
    //         serialization_states.emplace_back(nullptr);
    //         col.type->serializeBinaryBulkStatePrefix(serialize_settings, serialization_states.back());
    //     }
    // }

    /// Fill index granularity for this block
    /// if it's unknown (in case of insert data or horizontal merge,
    /// but not in case of vertical merge)
    if (compute_granularity)
        fillIndexGranularity(block);
    
    std::cerr << "(MergeTreeDataPartWriterWide::write) marks_count: " << index_granularity.getMarksCount() << "\n";

    WrittenOffsetColumns offset_columns;
    MarkWithOffset result;

    auto it = columns_list.begin();
    for (size_t i = 0; i < columns_list.size(); ++i, ++it)
    {
        const ColumnWithTypeAndName & column = block.getByName(it->name);

        if (permutation)
        {
            if (primary_key_block.has(it->name))
            {
                const auto & primary_column = *primary_key_block.getByName(it->name).column;
                result = writeColumn(column.name, *column.type, primary_column, offset_columns);
            }
            else if (skip_indexes_block.has(it->name))
            {
                const auto & index_column = *skip_indexes_block.getByName(it->name).column;
                result = writeColumn(column.name, *column.type, index_column, offset_columns);
            }
            else
            {
                /// We rearrange the columns that are not included in the primary key here; Then the result is released - to save RAM.
                ColumnPtr permuted_column = column.column->permute(*permutation, 0);
                result = writeColumn(column.name, *column.type, *permuted_column, offset_columns);
            }
        }
        else
        {
            result = writeColumn(column.name, *column.type, *column.column, offset_columns);
        }
    }
}

void MergeTreeDataPartWriterWide::writeSingleMark(
    const String & name,
    const IDataType & type,
    WrittenOffsetColumns & offset_columns,
    bool skip_offsets,
    size_t number_of_rows,
    DB::IDataType::SubstreamPath & path)
{
     type.enumerateStreams([&] (const IDataType::SubstreamPath & substream_path)
     {
         bool is_offsets = !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes;
         if (is_offsets && skip_offsets)
             return;

         String stream_name = IDataType::getFileNameForStream(name, substream_path);

         /// Don't write offsets more than one time for Nested type.
         if (is_offsets && offset_columns.count(stream_name))
             return;

         ColumnStream & stream = *column_streams[stream_name];

         /// There could already be enough data to compress into the new block.
         if (stream.compressed.offset() >= settings.min_compress_block_size)
             stream.compressed.next();

         writeIntBinary(stream.plain_hashing.count(), stream.marks);
         writeIntBinary(stream.compressed.offset(), stream.marks);
         if (settings.can_use_adaptive_granularity)
             writeIntBinary(number_of_rows, stream.marks);
     }, path);
}

size_t MergeTreeDataPartWriterWide::writeSingleGranule(
    const String & name,
    const IDataType & type,
    const IColumn & column,
    WrittenOffsetColumns & offset_columns,
    bool skip_offsets,
    IDataType::SerializeBinaryBulkStatePtr & serialization_state,
    IDataType::SerializeBinaryBulkSettings & serialize_settings,
    size_t from_row,
    size_t number_of_rows,
    bool write_marks)
{
    if (write_marks)
        writeSingleMark(name, type, offset_columns, skip_offsets, number_of_rows, serialize_settings.path);

    type.serializeBinaryBulkWithMultipleStreams(column, from_row, number_of_rows, serialize_settings, serialization_state);

    /// So that instead of the marks pointing to the end of the compressed block, there were marks pointing to the beginning of the next one.
    type.enumerateStreams([&] (const IDataType::SubstreamPath & substream_path)
    {
        bool is_offsets = !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes;
        if (is_offsets && skip_offsets)
            return;

        String stream_name = IDataType::getFileNameForStream(name, substream_path);

        /// Don't write offsets more than one time for Nested type.
        if (is_offsets && offset_columns.count(stream_name))
            return;

        column_streams[stream_name]->compressed.nextIfAtEnd();
    }, serialize_settings.path);

    return from_row + number_of_rows;
}

/// column must not be empty. (column.size() !== 0)

std::pair<size_t, size_t> MergeTreeDataPartWriterWide::writeColumn(
    const String & name,
    const IDataType & type,
    const IColumn & column,
    WrittenOffsetColumns & offset_columns,
    bool skip_offsets)
{
    auto [it, inserted] = serialization_states.emplace(name, nullptr);
    if (inserted)
    {
        IDataType::SerializeBinaryBulkSettings serialize_settings;
        serialize_settings.getter = createStreamGetter(name, offset_columns, false);
        type.serializeBinaryBulkStatePrefix(serialize_settings, it->second);
    }

    std::cerr << "(writeColumn) table: " << storage.getTableName() << "\n";
    std::cerr << "(writeColumn) column: " << name << "\n";
    std::cerr << "(writeColumn) index_offset: " << index_offset << "\n";

    auto & settings = storage.global_context.getSettingsRef();
    IDataType::SerializeBinaryBulkSettings serialize_settings;
    serialize_settings.getter = createStreamGetter(name, offset_columns, skip_offsets);
    serialize_settings.low_cardinality_max_dictionary_size = settings.low_cardinality_max_dictionary_size;
    serialize_settings.low_cardinality_use_single_dictionary_for_part = settings.low_cardinality_use_single_dictionary_for_part != 0;

    size_t total_rows = column.size();
    size_t current_row = 0;
    size_t current_column_mark = current_mark;
    while (current_row < total_rows)
    {
        size_t rows_to_write;
        bool write_marks = true;

        /// If there is `index_offset`, then the first mark goes not immediately, but after this number of rows.
        if (current_row == 0 && index_offset != 0)
        {
            write_marks = false;
            rows_to_write = index_offset;
        }
        else
        {
            if (index_granularity.getMarksCount() <= current_column_mark)
                throw Exception(
                    "Incorrect size of index granularity expect mark " + toString(current_column_mark) + " totally have marks " + toString(index_granularity.getMarksCount()),
                    ErrorCodes::LOGICAL_ERROR);

            rows_to_write = index_granularity.getMarkRows(current_column_mark);
        }

        if (rows_to_write != 0)
            data_written = true;

        current_row = writeSingleGranule(
            name,
            type,
            column,
            offset_columns,
            skip_offsets,
            it->second,
            serialize_settings,
            current_row,
            rows_to_write,
            write_marks
        );

        if (write_marks)
            current_column_mark++;
    }

    type.enumerateStreams([&] (const IDataType::SubstreamPath & substream_path)
    {
        bool is_offsets = !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes;
        if (is_offsets)
        {
            String stream_name = IDataType::getFileNameForStream(name, substream_path);
            offset_columns.insert(stream_name);
        }
    }, serialize_settings.path);

    next_mark = current_column_mark;
    next_index_offset = current_row - total_rows;

    return std::make_pair(current_column_mark, current_row - total_rows);
}

void MergeTreeDataPartWriterWide::finishDataSerialization(IMergeTreeDataPart::Checksums & checksums, bool sync)
{
    const auto & settings = storage.global_context.getSettingsRef();
    IDataType::SerializeBinaryBulkSettings serialize_settings;
    serialize_settings.low_cardinality_max_dictionary_size = settings.low_cardinality_max_dictionary_size;
    serialize_settings.low_cardinality_use_single_dictionary_for_part = settings.low_cardinality_use_single_dictionary_for_part != 0;
    WrittenOffsetColumns offset_columns;

    bool write_final_mark = (with_final_mark && data_written);

    {
        auto it = columns_list.begin();
        for (size_t i = 0; i < columns_list.size(); ++i, ++it)
        {
            if (!serialization_states.empty())
            {
                serialize_settings.getter = createStreamGetter(it->name, offset_columns, false);
                it->type->serializeBinaryBulkStateSuffix(serialize_settings, serialization_states[it->name]);
            }

            if (write_final_mark)
            {
                writeFinalMark(it->name, it->type, offset_columns, false, serialize_settings.path);
            }
        }
    }

    for (ColumnStreams::iterator it = column_streams.begin(); it != column_streams.end(); ++it)
    {
        it->second->finalize();
        if (sync)
            it->second->sync();
        it->second->addToChecksums(checksums);
    }

    column_streams.clear();
    serialization_states.clear();
}

void MergeTreeDataPartWriterWide::writeFinalMark(
    const std::string & column_name,
    const DataTypePtr column_type,
    WrittenOffsetColumns & offset_columns,
    bool skip_offsets,
    DB::IDataType::SubstreamPath & path)
{
    writeSingleMark(column_name, *column_type, offset_columns, skip_offsets, 0, path);
    /// Memoize information about offsets
    column_type->enumerateStreams([&] (const IDataType::SubstreamPath & substream_path)
    {
        bool is_offsets = !substream_path.empty() && substream_path.back().type == IDataType::Substream::ArraySizes;
        if (is_offsets)
        {
            String stream_name = IDataType::getFileNameForStream(column_name, substream_path);
            offset_columns.insert(stream_name);
        }
    }, path);
}

}
