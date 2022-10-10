#include <Processors/Formats/Impl/MsgPackRowInputFormat.h>

#if USE_MSGPACK

/// FIXME: there is some issue with clang-15, that incorrectly detect a
/// "Attempt to free released memory" in msgpack::unpack(), because of delete
/// operator for zone (from msgpack/v1/detail/cpp11_zone.hpp), hence NOLINT
///
/// NOTE: that I was not able to suppress it locally, only with
/// NOLINTBEGIN/NOLINTEND
//
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)

#include <cstdlib>
#include <Common/assert_cast.h>
#include <IO/ReadHelpers.h>
#include <IO/ReadBufferFromMemory.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeUUID.h>

#include <Columns/ColumnArray.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnMap.h>
#include <Columns/ColumnLowCardinality.h>

#include <Formats/MsgPackExtensionTypes.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
    extern const int INCORRECT_DATA;
    extern const int BAD_ARGUMENTS;
    extern const int UNEXPECTED_END_OF_FILE;
}

MsgPackRowInputFormat::MsgPackRowInputFormat(const Block & header_, ReadBuffer & in_, Params params_)
    : MsgPackRowInputFormat(header_, std::make_unique<PeekableReadBuffer>(in_), params_) {}

MsgPackRowInputFormat::MsgPackRowInputFormat(const Block & header_, std::unique_ptr<PeekableReadBuffer> buf_, Params params_)
    : IRowInputFormat(header_, *buf_, std::move(params_)), buf(std::move(buf_)), parser(visitor), data_types(header_.getDataTypes())  {}

void MsgPackRowInputFormat::resetParser()
{
    IRowInputFormat::resetParser();
    buf->reset();
    visitor.reset();
}

void MsgPackVisitor::set_info(IColumn & column, DataTypePtr type) // NOLINT
{
    while (!info_stack.empty())
    {
        info_stack.pop();
    }
    info_stack.push(Info{column, type});
}

void MsgPackVisitor::reset()
{
    info_stack = {};
}

template <typename InsertFunc>
static bool checkAndInsertNullable(IColumn & column, DataTypePtr type, InsertFunc insert_func)
{
    if (type->isNullable())
    {
        auto & nullable_column = assert_cast<ColumnNullable &>(column);
        auto & nested_column = nullable_column.getNestedColumn();
        const auto & nested_type = assert_cast<const DataTypeNullable *>(type.get())->getNestedType();
        insert_func(nested_column, nested_type);
        nullable_column.getNullMapColumn().insertValue(0);
        return true;
    }

    return false;
}

template <typename InsertFunc>
static bool checkAndInsertLowCardinality(IColumn & column, DataTypePtr type, InsertFunc insert_func)
{
    if (type->lowCardinality())
    {
        auto & lc_column = assert_cast<ColumnLowCardinality &>(column);
        auto tmp_column = lc_column.getDictionary().getNestedColumn()->cloneEmpty();
        auto dict_type = assert_cast<const DataTypeLowCardinality *>(type.get())->getDictionaryType();
        insert_func(*tmp_column, dict_type);
        lc_column.insertFromFullColumn(*tmp_column, 0);
        return true;
    }
    return false;
}

static void insertInteger(IColumn & column, DataTypePtr type, UInt64 value)
{
    auto insert_func = [&](IColumn & column_, DataTypePtr type_)
    {
        insertInteger(column_, type_, value);
    };

    if (checkAndInsertNullable(column, type, insert_func) || checkAndInsertLowCardinality(column, type, insert_func))
        return;

    switch (type->getTypeId())
    {
        case TypeIndex::UInt8:
        {
            assert_cast<ColumnUInt8 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::Date: [[fallthrough]];
        case TypeIndex::UInt16:
        {
            assert_cast<ColumnUInt16 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::DateTime: [[fallthrough]];
        case TypeIndex::UInt32:
        {
            assert_cast<ColumnUInt32 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::UInt64:
        {
            assert_cast<ColumnUInt64 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::Int8:
        {
            assert_cast<ColumnInt8 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::Int16:
        {
            assert_cast<ColumnInt16 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::Int32:
        {
            assert_cast<ColumnInt32 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::Int64:
        {
            assert_cast<ColumnInt64 &>(column).insertValue(value);
            break;
        }
        case TypeIndex::DateTime64:
        {
            assert_cast<DataTypeDateTime64::ColumnType &>(column).insertValue(value);
            break;
        }
        default:
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack integer into column with type {}.", type->getName());
    }
}

static void insertString(IColumn & column, DataTypePtr type, const char * value, size_t size, bool bin)
{
    auto insert_func = [&](IColumn & column_, DataTypePtr type_)
    {
        insertString(column_, type_, value, size, bin);
    };

    if (checkAndInsertNullable(column, type, insert_func) || checkAndInsertLowCardinality(column, type, insert_func))
        return;

    if (isUUID(type))
    {
        ReadBufferFromMemory buf(value, size);
        UUID uuid;
        if (bin)
            readBinary(uuid, buf);
        else
            readUUIDText(uuid, buf);

        assert_cast<ColumnUUID &>(column).insertValue(uuid);
        return;
    }

    if (!isStringOrFixedString(type))
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack string into column with type {}.", type->getName());

    column.insertData(value, size);
}

static void insertFloat32(IColumn & column, DataTypePtr type, Float32 value) // NOLINT
{
    auto insert_func = [&](IColumn & column_, DataTypePtr type_)
    {
        insertFloat32(column_, type_, value);
    };

    if (checkAndInsertNullable(column, type, insert_func) || checkAndInsertLowCardinality(column, type, insert_func))
        return;

    if (!WhichDataType(type).isFloat32())
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack float32 into column with type {}.", type->getName());

    assert_cast<ColumnFloat32 &>(column).insertValue(value);
}

static void insertFloat64(IColumn & column, DataTypePtr type, Float64 value) // NOLINT
{
    auto insert_func = [&](IColumn & column_, DataTypePtr type_)
    {
        insertFloat64(column_, type_, value);
    };

    if (checkAndInsertNullable(column, type, insert_func) || checkAndInsertLowCardinality(column, type, insert_func))
        return;

    if (!WhichDataType(type).isFloat64())
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack float64 into column with type {}.", type->getName());

    assert_cast<ColumnFloat64 &>(column).insertValue(value);
}

static void insertNull(IColumn & column, DataTypePtr type)
{
    auto insert_func = [&](IColumn & column_, DataTypePtr type_)
    {
        insertNull(column_, type_);
    };

    /// LowCardinality(Nullable(...))
    if (checkAndInsertLowCardinality(column, type, insert_func))
        return;

    if (!type->isNullable())
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack null into non-nullable column with type {}.", type->getName());

    assert_cast<ColumnNullable &>(column).insertDefault();
}

static void insertUUID(IColumn & column, DataTypePtr type, const char * value, size_t size)
{
    if (!isUUID(type))
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack UUID into column with type {}.", type->getName());
    ReadBufferFromMemory buf(value, size);
    UUID uuid;
    readBinaryBigEndian(uuid.toUnderType().items[0], buf);
    readBinaryBigEndian(uuid.toUnderType().items[1], buf);
    assert_cast<ColumnUUID &>(column).insertValue(uuid);
}

bool MsgPackVisitor::visit_positive_integer(UInt64 value) // NOLINT
{
    insertInteger(info_stack.top().column, info_stack.top().type, value);
    return true;
}

bool MsgPackVisitor::visit_negative_integer(Int64 value) // NOLINT
{
    insertInteger(info_stack.top().column, info_stack.top().type, value);
    return true;
}

bool MsgPackVisitor::visit_str(const char * value, size_t size) // NOLINT
{
    insertString(info_stack.top().column, info_stack.top().type, value, size, false);
    return true;
}

bool MsgPackVisitor::visit_bin(const char * value, size_t size) // NOLINT
{
    insertString(info_stack.top().column, info_stack.top().type, value, size, true);
    return true;
}

bool MsgPackVisitor::visit_float32(Float32 value) // NOLINT
{
    insertFloat32(info_stack.top().column, info_stack.top().type, value);
    return true;
}

bool MsgPackVisitor::visit_float64(Float64 value) // NOLINT
{
    insertFloat64(info_stack.top().column, info_stack.top().type, value);
    return true;
}

bool MsgPackVisitor::visit_boolean(bool value)
{
    insertInteger(info_stack.top().column, info_stack.top().type, UInt64(value));
    return true;
}

bool MsgPackVisitor::start_array(size_t size) // NOLINT
{
    if (!isArray(info_stack.top().type))
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack array into column with type {}.", info_stack.top().type->getName());

    auto nested_type = assert_cast<const DataTypeArray &>(*info_stack.top().type).getNestedType();
    ColumnArray & column_array = assert_cast<ColumnArray &>(info_stack.top().column);
    ColumnArray::Offsets & offsets = column_array.getOffsets();
    IColumn & nested_column = column_array.getData();
    offsets.push_back(offsets.back() + size);
    info_stack.push(Info{nested_column, nested_type});
    return true;
}

bool MsgPackVisitor::end_array() // NOLINT
{
    info_stack.pop();
    return true;
}

bool MsgPackVisitor::start_map(uint32_t size) // NOLINT
{
    if (!isMap(info_stack.top().type))
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert MessagePack map into column with type {}.", info_stack.top().type->getName());
    ColumnArray & column_array = assert_cast<ColumnMap &>(info_stack.top().column).getNestedColumn();
    ColumnArray::Offsets & offsets = column_array.getOffsets();
    offsets.push_back(offsets.back() + size);
    return true;
}

bool MsgPackVisitor::start_map_key() // NOLINT
{
    auto key_column = assert_cast<ColumnMap &>(info_stack.top().column).getNestedData().getColumns()[0];
    auto key_type = assert_cast<const DataTypeMap &>(*info_stack.top().type).getKeyType();
    info_stack.push(Info{*key_column, key_type});
    return true;
}

bool MsgPackVisitor::end_map_key() // NOLINT
{
    info_stack.pop();
    return true;
}

bool MsgPackVisitor::start_map_value() // NOLINT
{
    auto value_column = assert_cast<ColumnMap &>(info_stack.top().column).getNestedData().getColumns()[1];
    auto value_type = assert_cast<const DataTypeMap &>(*info_stack.top().type).getValueType();
    info_stack.push(Info{*value_column, value_type});
    return true;
}

bool MsgPackVisitor::end_map_value() // NOLINT
{
    info_stack.pop();
    return true;
}

bool MsgPackVisitor::visit_nil()
{
    insertNull(info_stack.top().column, info_stack.top().type);
    return true;
}

bool MsgPackVisitor::visit_ext(const char * value, uint32_t size)
{
    int8_t type = *value;
    if (*value == int8_t(MsgPackExtensionTypes::UUIDType))
    {
        insertUUID(info_stack.top().column, info_stack.top().type, value + 1, size - 1);
        return true;
    }

    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unsupported MsgPack extension type: {:x}", type);
}

void MsgPackVisitor::parse_error(size_t, size_t) // NOLINT
{
    throw Exception("Error occurred while parsing msgpack data.", ErrorCodes::INCORRECT_DATA);
}

bool MsgPackRowInputFormat::readObject()
{
    if (buf->eof())
        return false;

    PeekableReadBufferCheckpoint checkpoint{*buf};
    size_t offset = 0;
    while (!parser.execute(buf->position(), buf->available(), offset))
    {
        buf->position() = buf->buffer().end();
        if (buf->eof())
            throw Exception("Unexpected end of file while parsing msgpack object.", ErrorCodes::INCORRECT_DATA);
        buf->position() = buf->buffer().end();
        buf->makeContinuousMemoryFromCheckpointToPos();
        buf->rollbackToCheckpoint();
    }
    buf->position() += offset;
    return true;
}

bool MsgPackRowInputFormat::readRow(MutableColumns & columns, RowReadExtension &)
{
    size_t column_index = 0;
    bool has_more_data = true;
    for (; column_index != columns.size(); ++column_index)
    {
        visitor.set_info(*columns[column_index], data_types[column_index]);
        has_more_data = readObject();
        if (!has_more_data)
            break;
    }
    if (!has_more_data)
    {
        if (column_index != 0)
            throw Exception("Not enough values to complete the row.", ErrorCodes::INCORRECT_DATA);
        return false;
    }
    return true;
}

void MsgPackRowInputFormat::setReadBuffer(ReadBuffer & in_)
{
    buf = std::make_unique<PeekableReadBuffer>(in_);
    IInputFormat::setReadBuffer(in_);
}

MsgPackSchemaReader::MsgPackSchemaReader(ReadBuffer & in_, const FormatSettings & format_settings_)
    : IRowSchemaReader(buf, format_settings_), buf(in_), number_of_columns(format_settings_.msgpack.number_of_columns)
{
    if (!number_of_columns)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "You must specify setting input_format_msgpack_number_of_columns to extract table schema from MsgPack data");
}


msgpack::object_handle MsgPackSchemaReader::readObject()
{
    if (buf.eof())
        throw Exception(ErrorCodes::UNEXPECTED_END_OF_FILE, "Unexpected eof while parsing msgpack object");

    PeekableReadBufferCheckpoint checkpoint{buf};
    size_t offset = 0;
    bool need_more_data = true;
    msgpack::object_handle object_handle;
    while (need_more_data)
    {
        offset = 0;
        try
        {
            object_handle = msgpack::unpack(buf.position(), buf.buffer().end() - buf.position(), offset);
            need_more_data = false;
        }
        catch (msgpack::insufficient_bytes &)
        {
            buf.position() = buf.buffer().end();
            if (buf.eof())
                throw Exception("Unexpected end of file while parsing msgpack object", ErrorCodes::UNEXPECTED_END_OF_FILE);
            buf.position() = buf.buffer().end();
            buf.makeContinuousMemoryFromCheckpointToPos();
            buf.rollbackToCheckpoint();
        }
    }
    buf.position() += offset;
    return object_handle;
}

DataTypePtr MsgPackSchemaReader::getDataType(const msgpack::object & object)
{
    switch (object.type)
    {
        case msgpack::type::object_type::POSITIVE_INTEGER: [[fallthrough]];
        case msgpack::type::object_type::NEGATIVE_INTEGER:
            return makeNullable(std::make_shared<DataTypeInt64>());
        case msgpack::type::object_type::FLOAT32:
            return makeNullable(std::make_shared<DataTypeFloat32>());
        case msgpack::type::object_type::FLOAT64:
            return makeNullable(std::make_shared<DataTypeFloat64>());
        case msgpack::type::object_type::BOOLEAN:
            return makeNullable(std::make_shared<DataTypeUInt8>());
        case msgpack::type::object_type::BIN: [[fallthrough]];
        case msgpack::type::object_type::STR:
            return makeNullable(std::make_shared<DataTypeString>());
        case msgpack::type::object_type::ARRAY:
        {
            msgpack::object_array object_array = object.via.array;
            if (object_array.size)
            {
                auto nested_type = getDataType(object_array.ptr[0]);
                if (nested_type)
                    return std::make_shared<DataTypeArray>(getDataType(object_array.ptr[0]));
            }
            return nullptr;
        }
        case msgpack::type::object_type::MAP:
        {
            msgpack::object_map object_map = object.via.map;
            if (object_map.size)
            {
                auto key_type = removeNullable(getDataType(object_map.ptr[0].key));
                auto value_type = getDataType(object_map.ptr[0].val);
                if (key_type && value_type)
                    return std::make_shared<DataTypeMap>(key_type, value_type);
            }
            return nullptr;
        }
        case msgpack::type::object_type::NIL:
            return nullptr;
        case msgpack::type::object_type::EXT:
        {
            msgpack::object_ext object_ext = object.via.ext;
            if (object_ext.type() == int8_t(MsgPackExtensionTypes::UUIDType))
                return std::make_shared<DataTypeUUID>();
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Msgpack extension type {:x} is not supported", object_ext.type());
        }
    }
    UNREACHABLE();
}

DataTypes MsgPackSchemaReader::readRowAndGetDataTypes()
{
    if (buf.eof())
        return {};

    DataTypes data_types;
    data_types.reserve(number_of_columns);
    for (size_t i = 0; i != number_of_columns; ++i)
    {
        auto object_handle = readObject();
        data_types.push_back(getDataType(object_handle.get()));
    }

    return data_types;
}

void registerInputFormatMsgPack(FormatFactory & factory)
{
    factory.registerInputFormat("MsgPack", [](
            ReadBuffer & buf,
            const Block & sample,
            const RowInputFormatParams & params,
            const FormatSettings &)
    {
        return std::make_shared<MsgPackRowInputFormat>(sample, buf, params);
    });
    factory.registerFileExtension("messagepack", "MsgPack");
}

void registerMsgPackSchemaReader(FormatFactory & factory)
{
    factory.registerSchemaReader("MsgPack", [](ReadBuffer & buf, const FormatSettings & settings)
    {
        return std::make_shared<MsgPackSchemaReader>(buf, settings);
    });
    factory.registerAdditionalInfoForSchemaCacheGetter("MsgPack", [](const FormatSettings & settings)
    {
            return fmt::format(
                "number_of_columns={}, schema_inference_hints={}, max_rows_to_read_for_schema_inference={}",
                settings.msgpack.number_of_columns,
                settings.schema_inference_hints,
                settings.max_rows_to_read_for_schema_inference);
        });
}

}

// NOLINTEND(clang-analyzer-cplusplus.NewDelete)

#else

namespace DB
{
class FormatFactory;
void registerInputFormatMsgPack(FormatFactory &)
{
}

void registerMsgPackSchemaReader(FormatFactory &)
{
}
}

#endif
