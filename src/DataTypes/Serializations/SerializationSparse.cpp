#include <DataTypes/Serializations/SerializationSparse.h>
#include <DataTypes/DataTypesNumber.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnSparse.h>
#include <Common/assert_cast.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>

namespace DB
{

namespace
{

static constexpr auto END_OF_GRANULE_FLAG = 1ULL << 63;

struct SerializeStateSparse : public ISerialization::SerializeBinaryBulkState
{
    size_t num_trailing_default_values = 0;
    ISerialization::SerializeBinaryBulkStatePtr nested;
};

struct DeserializeStateSparse : public ISerialization::DeserializeBinaryBulkState
{
    size_t num_trailing_defaults = 0;
    bool has_value_after_defaults = false;
    ISerialization::DeserializeBinaryBulkStatePtr nested;
};

void serializeOffsetsPositionIndependent(const IColumn::Offsets & offsets, WriteBuffer & ostr, size_t start, size_t end)
{
    // std::cerr << "writing start: " << start << ", end: " << end << "\n";
    // std::cerr << "offsets: ";
    // for (const auto & x : offsets)
    //     std::cerr << x << " ";
    // std::cerr << "\n";

    size_t size = offsets.size();
    for (size_t i = 0; i < size; ++i)
    {
        size_t group_size = offsets[i] - start;

        // std::cerr << "writing group_size: " << group_size << "\n";

        writeIntBinary(group_size, ostr);
        start += group_size + 1;
    }

    // std::cerr << "writing start: " << start << ", end: " << end << "\n";
    size_t group_size = start < end ? end - start : 0;
    // std::cerr << "writing end group_size: " << group_size << "\n";
    group_size |= END_OF_GRANULE_FLAG;
    writeIntBinary(group_size, ostr);
}

// struct DeserializedRows
// {
//     size_t total = 0;
//     size_t trailing_defaults = 0;
// };

size_t deserializeOffsetsPositionIndependent(IColumn::Offsets & offsets,
    ReadBuffer & istr, size_t limit, DeserializeStateSparse & state)
{
    // std::cerr << "limit: " << limit << ", num_trailing: " << state.num_trailing_defaults
    //     << ", has_value_after_defaults: " << state.has_value_after_defaults << "\n";

    if (limit && state.num_trailing_defaults >= limit)
    {
        state.num_trailing_defaults -= limit;
        return limit;
    }

    /// TODO:
    offsets.reserve(limit / 10); /// TODO

    size_t total_rows = state.num_trailing_defaults;
    if (state.has_value_after_defaults)
    {
        size_t start_of_group = offsets.empty() ? 0 : offsets.back() + 1;
        offsets.push_back(start_of_group + state.num_trailing_defaults);

        state.has_value_after_defaults = false;
        state.num_trailing_defaults = 0;
        ++total_rows;
    }

    size_t group_size;
    while (!istr.eof())
    {
        readIntBinary(group_size, istr);

        bool end_of_granule = group_size & END_OF_GRANULE_FLAG;
        group_size &= ~END_OF_GRANULE_FLAG;

        // std::cerr << "read group_size: " << group_size << ", end_of_granule: " << end_of_granule << "\n";
        size_t next_total_rows = total_rows + group_size;
        group_size += state.num_trailing_defaults;


        // std::cerr << "group_size: " << group_size << ", end_of_granule: " << end_of_granule << "\n";
        // std::cerr << "next_total_rows: " << next_total_rows << "\n";

        if (limit && next_total_rows >= limit)
        {
            state.num_trailing_defaults = next_total_rows - limit;
            state.has_value_after_defaults = !end_of_granule;
            return limit;
        }

        if (end_of_granule)
        {
            state.has_value_after_defaults = false;
            state.num_trailing_defaults = group_size;
        }
        else
        {
            size_t start_of_group = offsets.empty() ? 0 : offsets.back() + 1;
            offsets.push_back(start_of_group + group_size);

            state.num_trailing_defaults = 0;
            state.has_value_after_defaults = false;
            ++next_total_rows;
        }

        total_rows = next_total_rows;
    }

    return total_rows;
}

}

SerializationSparse::SerializationSparse(const SerializationPtr & nested_serialization_)
    : SerializationWrapper(nested_serialization_)
{
}

void SerializationSparse::enumerateStreams(const StreamCallback & callback, SubstreamPath & path) const
{
    path.push_back(Substream::SparseOffsets);
    callback(path);
    path.back() = Substream::SparseElements;
    nested_serialization->enumerateStreams(callback, path);
    path.pop_back();
}

void SerializationSparse::serializeBinaryBulkStatePrefix(
    SerializeBinaryBulkSettings & settings,
    SerializeBinaryBulkStatePtr & state) const
{
    auto state_sparse = std::make_shared<SerializeStateSparse>();

    settings.path.push_back(Substream::SparseElements);
    nested_serialization->serializeBinaryBulkStatePrefix(settings, state_sparse->nested);
    settings.path.pop_back();

    state = std::move(state_sparse);
}

void SerializationSparse::serializeBinaryBulkWithMultipleStreams(
    const IColumn & column,
    size_t offset,
    size_t limit,
    SerializeBinaryBulkSettings & settings,
    SerializeBinaryBulkStatePtr & state) const
{
    size_t size = column.size();
    auto * state_sparse = checkAndGetSerializeState<SerializeStateSparse>(state, *this);

    // std::cerr << "writing column: " << column.dumpStructure() << "\n";

    auto offsets_column = DataTypeNumber<IColumn::Offset>().createColumn();
    auto & offsets_data = assert_cast<ColumnVector<IColumn::Offset> &>(*offsets_column).getData();
    column.getIndicesOfNonDefaultValues(offsets_data, offset, limit);

    settings.path.push_back(Substream::SparseOffsets);
    if (auto * stream = settings.getter(settings.path))
    {
        size_t end = limit && offset + limit < size ? offset + limit : size;
        serializeOffsetsPositionIndependent(offsets_data, *stream, offset, end);
    }

    if (!offsets_data.empty())
    {
        settings.path.back() = Substream::SparseElements;
        if (const auto * column_sparse = typeid_cast<const ColumnSparse *>(&column))
        {
            const auto & values = column_sparse->getValuesColumn();
            size_t begin = column_sparse->getValueIndex(offsets_data[0]);
            size_t end = column_sparse->getValueIndex(offsets_data.back());
            // std::cerr << "begin: " << begin << ", end: " << end << "\n";
            nested_serialization->serializeBinaryBulkWithMultipleStreams(values, begin, end - begin + 1, settings, state_sparse->nested);
        }
        else
        {
            auto values = column.index(*offsets_column, 0);
            nested_serialization->serializeBinaryBulkWithMultipleStreams(*values, 0, values->size(), settings, state_sparse->nested);
        }
    }

    settings.path.pop_back();
}

void SerializationSparse::serializeBinaryBulkStateSuffix(
    SerializeBinaryBulkSettings & settings,
    SerializeBinaryBulkStatePtr & state) const
{
    auto * state_sparse = checkAndGetSerializeState<SerializeStateSparse>(state, *this);

    settings.path.push_back(Substream::SparseElements);
    nested_serialization->serializeBinaryBulkStateSuffix(settings, state_sparse->nested);
    settings.path.pop_back();
}

void SerializationSparse::deserializeBinaryBulkStatePrefix(
    DeserializeBinaryBulkSettings & settings,
    DeserializeBinaryBulkStatePtr & state) const
{
    auto state_sparse = std::make_shared<DeserializeStateSparse>();

    settings.path.push_back(Substream::SparseElements);
    nested_serialization->deserializeBinaryBulkStatePrefix(settings, state_sparse->nested);
    settings.path.pop_back();

    state = std::move(state_sparse);
}

void SerializationSparse::deserializeBinaryBulkWithMultipleStreams(
    ColumnPtr & column,
    size_t limit,
    DeserializeBinaryBulkSettings & settings,
    DeserializeBinaryBulkStatePtr & state,
    SubstreamsCache * cache) const
{
    settings.path.push_back(Substream::SparseOffsets);
    auto * state_sparse = checkAndGetDeserializeState<DeserializeStateSparse>(state, *this);

    auto mutable_column = column->assumeMutable();
    auto & column_sparse = assert_cast<ColumnSparse &>(*mutable_column);
    auto & offsets_data = column_sparse.getOffsetsData();

    size_t old_size = offsets_data.size();

    size_t read_rows = 0;
    if (auto * stream = settings.getter(settings.path))
        read_rows = deserializeOffsetsPositionIndependent(offsets_data, *stream, limit, *state_sparse);

    auto & values_column = column_sparse.getValuesPtr();
    size_t values_limit = offsets_data.size() - old_size;

    settings.path.back() = Substream::SparseElements;
    nested_serialization->deserializeBinaryBulkWithMultipleStreams(values_column, values_limit, settings, state_sparse->nested, cache);
    settings.path.pop_back();

    if (offsets_data.size() + 1 != values_column->size())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Inconsistent sizes of values and offsets in SerializationSparse."
        " Offsets size: {}, values size: {}", offsets_data.size(), values_column->size());

    column_sparse.insertManyDefaults(read_rows);

    // std::cerr << "column_sparse: " << column_sparse.dumpStructure() << "\n";
    // std::cerr << "offsets: ";
    // for (const auto & x : column_sparse.getOffsetsData())
    //     std::cerr << x << " ";
    // std::cerr << "\n";

    // std::cerr << "values: ";
    // for (size_t i = 0; i < column_sparse.getValuesColumn().size(); ++i)
    //     std::cerr << toString(column_sparse.getValuesColumn()[i]) << " ";
    // std::cerr << "\n";

    column = std::move(mutable_column);
}

// void SerializationSparse::serializeText(const IColumn & column, size_t row_num, WriteBuffer & ostr, const FormatSettings & settings) const
// {
//     const auto & column_sparse = assert_cast<const ColumnSparse &>(column);
//     const auto & values_column = column_sparse.getValuesColumn();
//     nested_serialization->serializeText(values_column, column_sparse.getValueIndex(row_num), ostr, settings);
// }

}
