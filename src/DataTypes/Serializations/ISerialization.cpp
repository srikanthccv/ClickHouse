#include <DataTypes/Serializations/ISerialization.h>
#include <Compression/CompressionFactory.h>
#include <Columns/IColumn.h>
#include <IO/WriteHelpers.h>
#include <IO/Operators.h>
#include <Common/escapeForFileName.h>
#include <DataTypes/NestedUtils.h>
#include <base/EnumReflection.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int MULTIPLE_STREAMS_REQUIRED;
    extern const int LOGICAL_ERROR;
}

ISerialization::Kind ISerialization::getKind(const IColumn & column)
{
    if (column.isSparse())
        return Kind::SPARSE;

    return Kind::DEFAULT;
}

String ISerialization::kindToString(Kind kind)
{
    switch (kind)
    {
        case Kind::DEFAULT:
            return "Default";
        case Kind::SPARSE:
            return "Sparse";
    }

    __builtin_unreachable();
}

ISerialization::Kind ISerialization::stringToKind(const String & str)
{
    if (str == "Default")
        return Kind::DEFAULT;
    else if (str == "Sparse")
        return Kind::SPARSE;
    else
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Unknown serialization kind '{}'", str);
}

String ISerialization::Substream::toString() const
{
    if (type == TupleElement)
        return fmt::format("TupleElement({}, escape_tuple_delimiter = {})",
            tuple_element_name, escape_tuple_delimiter ? "true" : "false");

    return String(magic_enum::enum_name(type));
}

String ISerialization::SubstreamPath::toString() const
{
    WriteBufferFromOwnString wb;
    wb << "{";
    for (size_t i = 0; i < size(); ++i)
    {
        if (i != 0)
            wb << ", ";
        wb << at(i).toString();
    }
    wb << "}";
    return wb.str();
}

void ISerialization::enumerateStreams(const StreamCallback & callback, SubstreamPath & path) const
{
    callback(path);
}

void ISerialization::serializeBinaryBulk(const IColumn & column, WriteBuffer &, size_t, size_t) const
{
    throw Exception(ErrorCodes::MULTIPLE_STREAMS_REQUIRED, "Column {} must be serialized with multiple streams", column.getName());
}

void ISerialization::deserializeBinaryBulk(IColumn & column, ReadBuffer &, size_t, double) const
{
    throw Exception(ErrorCodes::MULTIPLE_STREAMS_REQUIRED, "Column {} must be deserialized with multiple streams", column.getName());
}

void ISerialization::serializeBinaryBulkWithMultipleStreams(
    const IColumn & column,
    size_t offset,
    size_t limit,
    SerializeBinaryBulkSettings & settings,
    SerializeBinaryBulkStatePtr & /* state */) const
{
    if (WriteBuffer * stream = settings.getter(settings.path))
        serializeBinaryBulk(column, *stream, offset, limit);
}

void ISerialization::deserializeBinaryBulkWithMultipleStreams(
    ColumnPtr & column,
    size_t limit,
    DeserializeBinaryBulkSettings & settings,
    DeserializeBinaryBulkStatePtr & /* state */,
    SubstreamsCache * cache) const
{
    auto cached_column = getFromSubstreamsCache(cache, settings.path);
    if (cached_column)
    {
        column = cached_column;
    }
    else if (ReadBuffer * stream = settings.getter(settings.path))
    {
        auto mutable_column = column->assumeMutable();
        deserializeBinaryBulk(*mutable_column, *stream, limit, settings.avg_value_size_hint);
        column = std::move(mutable_column);
        addToSubstreamsCache(cache, settings.path, column);
    }
}

static String getNameForSubstreamPath(
    String stream_name,
    const ISerialization::SubstreamPath & path,
    bool escape_tuple_delimiter)
{
    using Substream = ISerialization::Substream;

    size_t array_level = 0;
    for (const auto & elem : path)
    {
        if (elem.type == Substream::NullMap)
            stream_name += ".null";
        else if (elem.type == Substream::ArraySizes)
            stream_name += ".size" + toString(array_level);
        else if (elem.type == Substream::ArrayElements)
            ++array_level;
        else if (elem.type == Substream::DictionaryKeys)
            stream_name += ".dict";
        else if (elem.type == Substream::SparseOffsets)
            stream_name += ".sparse.idx";
        else if (elem.type == Substream::TupleElement)
        {
            /// For compatibility reasons, we use %2E (escaped dot) instead of dot.
            /// Because nested data may be represented not by Array of Tuple,
            ///  but by separate Array columns with names in a form of a.b,
            ///  and name is encoded as a whole.
            stream_name += (escape_tuple_delimiter && elem.escape_tuple_delimiter ?
                escapeForFileName(".") : ".") + escapeForFileName(elem.tuple_element_name);
        }
    }

    return stream_name;
}

String ISerialization::getFileNameForStream(const NameAndTypePair & column, const SubstreamPath & path)
{
    return getFileNameForStream(column.getNameInStorage(), path);
}

static size_t isOffsetsOfNested(const ISerialization::SubstreamPath & path)
{
    if (path.empty())
        return false;

    for (const auto & elem : path)
        if (elem.type == ISerialization::Substream::ArrayElements)
            return false;

    return path.back().type == ISerialization::Substream::ArraySizes;
}

String ISerialization::getFileNameForStream(const String & name_in_storage, const SubstreamPath & path)
{
    String stream_name;
    auto nested_storage_name = Nested::extractTableName(name_in_storage);
    if (name_in_storage != nested_storage_name && isOffsetsOfNested(path))
        stream_name = escapeForFileName(nested_storage_name);
    else
        stream_name = escapeForFileName(name_in_storage);

    return getNameForSubstreamPath(std::move(stream_name), path, true);
}

String ISerialization::getSubcolumnNameForStream(const SubstreamPath & path)
{
    auto subcolumn_name = getNameForSubstreamPath("", path, false);
    if (!subcolumn_name.empty())
        subcolumn_name = subcolumn_name.substr(1); // It starts with a dot.

    return subcolumn_name;
}

void ISerialization::addToSubstreamsCache(SubstreamsCache * cache, const SubstreamPath & path, ColumnPtr column)
{
    if (cache && !path.empty())
        cache->emplace(getSubcolumnNameForStream(path), column);
}

ColumnPtr ISerialization::getFromSubstreamsCache(SubstreamsCache * cache, const SubstreamPath & path)
{
    if (!cache || path.empty())
        return nullptr;

    auto it = cache->find(getSubcolumnNameForStream(path));
    if (it == cache->end())
        return nullptr;

    return it->second;
}

bool ISerialization::isSpecialCompressionAllowed(const SubstreamPath & path)
{
    for (const auto & elem : path)
    {
        if (elem.type == Substream::NullMap
            || elem.type == Substream::ArraySizes
            || elem.type == Substream::DictionaryIndexes
            || elem.type == Substream::SparseOffsets)
            return false;
    }
    return true;
}

}
