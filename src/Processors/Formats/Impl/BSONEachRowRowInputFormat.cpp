#include <IO/ReadBufferFromString.h>

#include <Formats/FormatFactory.h>
#include <Formats/FormatSettings.h>
#include <Formats/BSONTypes.h>
#include <Formats/EscapingRuleUtils.h>
#include <Processors/Formats/Impl/BSONEachRowRowInputFormat.h>
#include <IO/ReadHelpers.h>

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnMap.h>

#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeUUID.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/getLeastSupertype.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int INCORRECT_DATA;
    extern const int ILLEGAL_COLUMN;
    extern const int TOO_LARGE_STRING_SIZE;
    extern const int UNKNOWN_TYPE;
}

namespace
{
    enum
    {
        UNKNOWN_FIELD = size_t(-1),
    };
}

BSONEachRowRowInputFormat::BSONEachRowRowInputFormat(
    ReadBuffer & in_, const Block & header_, Params params_, const FormatSettings & format_settings_)
    : IRowInputFormat(header_, in_, std::move(params_))
    , format_settings(format_settings_)
    , name_map(header_.getNamesToIndexesMap())
    , prev_positions(header_.columns())
    , types(header_.getDataTypes())
{
}

inline size_t BSONEachRowRowInputFormat::columnIndex(const StringRef & name, size_t key_index)
{
    /// Optimization by caching the order of fields (which is almost always the same)
    /// and a quick check to match the next expected field, instead of searching the hash table.

    if (prev_positions.size() > key_index && prev_positions[key_index] && name == prev_positions[key_index]->getKey())
    {
        return prev_positions[key_index]->getMapped();
    }
    else
    {
        auto * it = name_map.find(name);

        if (it)
        {
            if (key_index < prev_positions.size())
                prev_positions[key_index] = it;

            return it->getMapped();
        }
        else
            return UNKNOWN_FIELD;
    }
}

/// Read the field name. Resulting StringRef is valid only before next read from buf.
static StringRef readBSONKeyName(ReadBuffer & in, String & key_holder)
{
    // This is just an optimization: try to avoid copying the name into key_holder

    if (!in.eof())
    {
        char * next_pos = find_first_symbols<0>(in.position(), in.buffer().end());

        if (next_pos != in.buffer().end() )
        {
            StringRef res(in.position(), next_pos - in.position());
            in.position() = next_pos + 1;
            return res;
        }
    }

    key_holder.clear();
    readNullTerminated(key_holder, in);
    return key_holder;
}

static UInt8 readBSONType(ReadBuffer & in)
{
    UInt8 type;
    readBinary(type, in);
    return type;
}

static size_t readBSONSize(ReadBuffer & in)
{
    BSON_SIZE_TYPE size;
    readBinary(size, in);
    return size;
}

template <typename T>
static void readAndInsertInteger(ReadBuffer & in, IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    /// We allow to read any integer into any integer column.
    /// For example we can read BSON Int32 into ClickHouse UInt8.

    if (bson_type == BSONType::INT32)
    {
        Int32 value;
        readBinary(value, in);
        assert_cast<ColumnVector<T> &>(column).insertValue(static_cast<T>(value));
    }
    else if (bson_type == BSONType::INT64 || bson_type == BSONType::UINT64)
    {
        UInt64 value;
        readBinary(value, in);
        assert_cast<ColumnVector<T> &>(column).insertValue(static_cast<T>(value));
    }
    else if (bson_type == BSONType::BOOL)
    {
        UInt8 value;
        readBinary(value, in);
        assert_cast<ColumnVector<T> &>(column).insertValue(static_cast<T>(value));
    }
    else
    {
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into column with type {}", getBSONTypeName(bson_type), data_type->getName());
    }
}

template <typename T>
static void readAndInsertDouble(ReadBuffer & in, IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    if (bson_type != BSONType::DOUBLE)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into column with type {}", getBSONTypeName(bson_type), data_type->getName());

    Float64 value;
    readBinary(value, in);
    assert_cast<ColumnVector<T> &>(column).insertValue(static_cast<T>(value));
}

template <typename DecimalType, BSONType expected_bson_type>
static void readAndInsertSmallDecimal(ReadBuffer & in, IColumn & column, const DataTypePtr & data_type,  BSONType bson_type)
{
    if (bson_type != expected_bson_type)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into column with type {}", getBSONTypeName(bson_type), data_type->getName());

    DecimalType value;
    readBinary(value, in);
    assert_cast<ColumnDecimal<DecimalType> &>(column).insertValue(value);
}

static void readAndInsertDateTime64(ReadBuffer & in, IColumn & column, BSONType bson_type)
{
    if (bson_type != BSONType::INT64 && bson_type != BSONType::DATETIME)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into DateTime64 column", getBSONTypeName(bson_type));

    DateTime64 value;
    readBinary(value, in);
    assert_cast<DataTypeDateTime64::ColumnType &>(column).insertValue(value);
}

template <typename ColumnType>
static void readAndInsertBigInteger(ReadBuffer & in, IColumn & column, const DataTypePtr & data_type,  BSONType bson_type)
{
    if (bson_type != BSONType::BINARY)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into column with type {}", getBSONTypeName(bson_type), data_type->getName());

    auto size = readBSONSize(in);
    auto subtype = getBSONBinarySubtype(readBSONType(in));
    if (subtype != BSONBinarySubtype::BINARY)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON Binary subtype {} into column with type {}", getBSONBinarySubtypeName(subtype), data_type->getName());

    using ValueType = typename ColumnType::ValueType;

    if (size != sizeof(ValueType))
        throw Exception(
            ErrorCodes::INCORRECT_DATA,
            "Cannot parse value of type {}, size of binary data is not equal to the binary size of expected value: {} != {}",
            data_type->getName(),
            size,
            sizeof(ValueType));

    ValueType value;
    readBinary(value, in);
    assert_cast<ColumnType &>(column).insertValue(value);
}

template <bool is_fixed_string>
static void readAndInsertStringImpl(ReadBuffer & in, IColumn & column, size_t size)
{
    if constexpr (is_fixed_string)
    {
        auto & fixed_string_column = assert_cast<ColumnFixedString &>(column);
        size_t n = fixed_string_column.getN();
        if (size > n)
            throw Exception("Too large string for FixedString column", ErrorCodes::TOO_LARGE_STRING_SIZE);

        auto & data = fixed_string_column.getChars();

        size_t old_size = data.size();
        data.resize_fill(old_size + n);

        try
        {
            in.readStrict(reinterpret_cast<char *>(data.data() + old_size), size);
        }
        catch (...)
        {
            /// Restore column state in case of any exception.
            data.resize_assume_reserved(old_size);
            throw;
        }
    }
    else
    {
        auto & column_string = assert_cast<ColumnString &>(column);
        auto & data = column_string.getChars();
        auto & offsets = column_string.getOffsets();

        size_t old_chars_size = data.size();
        size_t offset = old_chars_size + size + 1;
        offsets.push_back(offset);

        try
        {
            data.resize(offset);
            in.readStrict(reinterpret_cast<char *>(&data[offset - size - 1]), size);
            data.back() = 0;
        }
        catch (...)
        {
            /// Restore column state in case of any exception.
            offsets.pop_back();
            data.resize_assume_reserved(old_chars_size);
            throw;
        }
    }
}

template <bool is_fixed_string>
static void readAndInsertString(ReadBuffer & in, IColumn & column, BSONType bson_type)
{
    if (bson_type == BSONType::STRING || bson_type == BSONType::SYMBOL)
    {
        auto size = readBSONSize(in);
        readAndInsertStringImpl<is_fixed_string>(in, column, size - 1);
        assertChar(0, in);
    }
    else if (bson_type == BSONType::BINARY)
    {
        auto size = readBSONSize(in);
        auto subtype = getBSONBinarySubtype(readBSONType(in));
        if (subtype == BSONBinarySubtype::BINARY || subtype == BSONBinarySubtype::BINARY_OLD)
            readAndInsertStringImpl<is_fixed_string>(in, column, size);
        else
            throw Exception(
                ErrorCodes::ILLEGAL_COLUMN,
                "Cannot insert BSON Binary subtype {} into String column",
                getBSONBinarySubtypeName(subtype));
    }
    else
    {
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into String column", getBSONTypeName(bson_type));
    }
}

static void readAndInsertUUID(ReadBuffer & in, IColumn & column, BSONType bson_type)
{
    if (bson_type == BSONType::BINARY)
    {
        auto size = readBSONSize(in);
        auto subtype = getBSONBinarySubtype(readBSONType(in));
        if (subtype == BSONBinarySubtype::UUID || subtype == BSONBinarySubtype::UUID_OLD)
        {
            if (size != sizeof(UUID))
                throw Exception(
                    ErrorCodes::INCORRECT_DATA,
                    "Cannot parse value of type UUID, size of binary data is not equal to the binary size of UUID value: {} != {}",
                    size,
                    sizeof(UUID));

            UUID value;
            readBinary(value, in);
            assert_cast<ColumnUUID &>(column).insertValue(value);
        }
        else
        {
            throw Exception(
                ErrorCodes::ILLEGAL_COLUMN,
                "Cannot insert BSON Binary subtype {} into UUID column",
                getBSONBinarySubtypeName(subtype));
        }
    }
    else
    {
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into UUID column", getBSONTypeName(bson_type));
    }
}

void BSONEachRowRowInputFormat::readArray(IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    if (bson_type != BSONType::ARRAY)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into Array column", getBSONTypeName(bson_type));

    const auto * data_type_array = assert_cast<const DataTypeArray *>(data_type.get());
    const auto & nested_type = data_type_array->getNestedType();
    auto & array_column = assert_cast<ColumnArray &>(column);
    auto & nested_column = array_column.getData();

    size_t document_start = in->count();
    BSON_SIZE_TYPE document_size;
    readBinary(document_size, *in);
    while (in->count() - document_start + sizeof(BSON_DOCUMENT_END) != document_size)
    {
        auto nested_bson_type = getBSONType(readBSONType(*in));
        readBSONKeyName(*in, current_key_name);
        readField(nested_column, nested_type, nested_bson_type);
    }

    assertChar(BSON_DOCUMENT_END, *in);
    array_column.getOffsets().push_back(array_column.getData().size());
}

void BSONEachRowRowInputFormat::readTuple(IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    if (bson_type != BSONType::ARRAY && bson_type != BSONType::DOCUMENT)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into Tuple column", getBSONTypeName(bson_type));

    /// When BSON type is ARRAY, names in nested document are not useful
    /// (most likely they are just sequential numbers).
    bool use_key_names = bson_type == BSONType::DOCUMENT;

    const auto * data_type_tuple = assert_cast<const DataTypeTuple *>(data_type.get());
    auto & tuple_column = assert_cast<ColumnTuple &>(column);
    size_t read_nested_columns = 0;

    size_t document_start = in->count();
    BSON_SIZE_TYPE document_size;
    readBinary(document_size, *in);
    while (in->count() - document_start + sizeof(BSON_DOCUMENT_END) != document_size)
    {
        auto nested_bson_type = getBSONType(readBSONType(*in));
        auto name = readBSONKeyName(*in, current_key_name);

        size_t index = read_nested_columns;
        if (use_key_names)
        {
            auto try_get_index = data_type_tuple->tryGetPositionByName(name.toString());
            if (!try_get_index)
                throw Exception(
                    ErrorCodes::INCORRECT_DATA,
                    "Cannot parse tuple column with type {} from BSON array/embedded document field: tuple doesn't have element with name \"{}\"",
                    data_type->getName(),
                    name);
            index = *try_get_index;
        }

        if (index >= data_type_tuple->getElements().size())
            throw Exception(
                ErrorCodes::INCORRECT_DATA,
                "Cannot parse tuple column with type {} from BSON array/embedded document field: the number of fields BSON document exceeds the number of fields in tuple",
                data_type->getName());

        readField(tuple_column.getColumn(index), data_type_tuple->getElement(index), nested_bson_type);
        ++read_nested_columns;
    }

    assertChar(BSON_DOCUMENT_END, *in);

    if (read_nested_columns != data_type_tuple->getElements().size())
        throw Exception(
            ErrorCodes::INCORRECT_DATA,
            "Cannot parse tuple column with type {} from BSON array/embedded document field, the number of fields in tuple and BSON document doesn't match: {} != {}",
            data_type->getName(),
            data_type_tuple->getElements().size(),
            read_nested_columns);
}

void BSONEachRowRowInputFormat::readMap(IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    if (bson_type != BSONType::DOCUMENT)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON {} into Map column", getBSONTypeName(bson_type));

    const auto * data_type_map = assert_cast<const DataTypeMap *>(data_type.get());
    const auto & key_data_type = data_type_map->getKeyType();
    if (!isStringOrFixedString(key_data_type))
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Only maps with String key type are supported in BSON, got key type: {}", key_data_type->getName());

    const auto & value_data_type = data_type_map->getValueType();
    auto & column_map = assert_cast<ColumnMap &>(column);
    auto & key_column = column_map.getNestedData().getColumn(0);
    auto & value_column = column_map.getNestedData().getColumn(1);
    auto & offsets = column_map.getNestedColumn().getOffsets();

    size_t document_start = in->count();
    BSON_SIZE_TYPE document_size;
    readBinary(document_size, *in);
    while (in->count() - document_start + sizeof(BSON_DOCUMENT_END) != document_size)
    {
        auto nested_bson_type = getBSONType(readBSONType(*in));
        auto name = readBSONKeyName(*in, current_key_name);
        key_column.insertData(name.data, name.size);
        readField(value_column, value_data_type, nested_bson_type);
    }

    assertChar(BSON_DOCUMENT_END, *in);
    offsets.push_back(key_column.size());
}


bool BSONEachRowRowInputFormat::readField(IColumn & column, const DataTypePtr & data_type, BSONType bson_type)
{
    if (bson_type == BSONType::NULL_VALUE)
    {
        if (data_type->isNullable())
        {
            column.insertDefault();
            return true;
        }

        if (!format_settings.null_as_default)
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cannot insert BSON Null value into non-nullable column with type {}", getBSONTypeName(bson_type), data_type->getName());

        column.insertDefault();
        return false;
    }

    switch (data_type->getTypeId())
    {
        case TypeIndex::Nullable:
        {
            auto & nullable_column = assert_cast<ColumnNullable &>(column);
            auto & nested_column = nullable_column.getNestedColumn();
            const auto & nested_type = assert_cast<const DataTypeNullable *>(data_type.get())->getNestedType();
            nullable_column.getNullMapColumn().insertValue(0);
            return readField(nested_column, nested_type, bson_type);
        }
        case TypeIndex::LowCardinality:
        {
            auto & lc_column = assert_cast<ColumnLowCardinality &>(column);
            auto tmp_column = lc_column.getDictionary().getNestedColumn()->cloneEmpty();
            const auto & dict_type = assert_cast<const DataTypeLowCardinality *>(data_type.get())->getDictionaryType();
            auto res = readField(*tmp_column, dict_type, bson_type);
            lc_column.insertFromFullColumn(*tmp_column, 0);
            return res;
        }
        case TypeIndex::Int8:
        {
            readAndInsertInteger<Int8>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::UInt8:
        {
            readAndInsertInteger<UInt8>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Int16:
        {
            readAndInsertInteger<Int16>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Date: [[fallthrough]];
        case TypeIndex::UInt16:
        {
            readAndInsertInteger<UInt16>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Date32: [[fallthrough]];
        case TypeIndex::Int32:
        {
            readAndInsertInteger<Int32>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::DateTime: [[fallthrough]];
        case TypeIndex::UInt32:
        {
            readAndInsertInteger<UInt32>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Int64:
        {
            readAndInsertInteger<Int64>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::UInt64:
        {
            readAndInsertInteger<UInt64>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Int128:
        {
            readAndInsertBigInteger<ColumnInt128>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::UInt128:
        {
            readAndInsertBigInteger<ColumnUInt128>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Int256:
        {
            readAndInsertBigInteger<ColumnInt256>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::UInt256:
        {
            readAndInsertBigInteger<ColumnUInt256>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Float32:
        {
            readAndInsertDouble<Float32>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Float64:
        {
            readAndInsertDouble<Float64>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Decimal32:
        {
            readAndInsertSmallDecimal<Decimal32, BSONType::INT32>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Decimal64:
        {
            readAndInsertSmallDecimal<Decimal64, BSONType::INT64>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Decimal128:
        {
            readAndInsertBigInteger<ColumnDecimal<Decimal128>>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Decimal256:
        {
            readAndInsertBigInteger<ColumnDecimal<Decimal256>>(*in, column, data_type, bson_type);
            return true;
        }
        case TypeIndex::DateTime64:
        {
            readAndInsertDateTime64(*in, column, bson_type);
            return true;
        }
        case TypeIndex::FixedString:
        {
            readAndInsertString<true>(*in, column, bson_type);
            return true;
        }
        case TypeIndex::String:
        {
            readAndInsertString<false>(*in, column, bson_type);
            return true;
        }
        case TypeIndex::UUID:
        {
            readAndInsertUUID(*in, column, bson_type);
            return true;
        }
        case TypeIndex::Array:
        {
            readArray(column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Tuple:
        {
            readTuple(column, data_type, bson_type);
            return true;
        }
        case TypeIndex::Map:
        {
            readMap(column, data_type, bson_type);
            return true;
        }
        default:
        {
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Type {} is not supported for output in BSON format", data_type->getName());
        }
    }
}

static void skipBSONField(ReadBuffer & in, BSONType type)
{
    switch (type)
    {
        case BSONType::DOUBLE:
        {
            in.ignore(sizeof(Float64));
            break;
        }
        case BSONType::BOOL:
        {
            in.ignore(sizeof(UInt8));
            break;
        }
        case BSONType::INT64: [[fallthrough]];
        case BSONType::DATETIME: [[fallthrough]];
        case BSONType::UINT64:
        {
            in.ignore(sizeof(UInt64));
            break;
        }
        case BSONType::INT32:
        {
            in.ignore(sizeof(Int32));
            break;
        }
        case BSONType::JAVA_SCRIPT_CODE: [[fallthrough]];
        case BSONType::SYMBOL: [[fallthrough]];
        case BSONType::STRING:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size);
            break;
        }
        case BSONType::DOCUMENT: [[fallthrough]];
        case BSONType::ARRAY:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size - sizeof(size));
            break;
        }
        case BSONType::BINARY:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size + 1);
            break;
        }
        case BSONType::MIN_KEY: [[fallthrough]];
        case BSONType::MAX_KEY: [[fallthrough]];
        case BSONType::UNDEFINED: [[fallthrough]];
        case BSONType::NULL_VALUE:
        {
            break;
        }
        case BSONType::OBJECT_ID:
        {
            in.ignore(12);
            break;
        }
        case BSONType::REGEXP:
        {
            skipNullTerminated(in);
            skipNullTerminated(in);
            break;
        }
        case BSONType::DB_POINTER:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size + 12);
            break;
        }
        case BSONType::JAVA_SCRIPT_CODE_W_SCOPE:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size - sizeof(size));
            break;
        }
        case BSONType::DECIMAL128:
        {
            in.ignore(16);
            break;
        }
    }
}

void BSONEachRowRowInputFormat::skipUnknownField(BSONType type, const String & key_name)
{
    if (!format_settings.skip_unknown_fields)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Unknown field found while parsing BSONEachRow format: {}", key_name);

    skipBSONField(*in, type);
}

void BSONEachRowRowInputFormat::syncAfterError()
{
    /// Skip all remaining bytes in current document
    size_t already_read_bytes = in->count() - current_document_start;
    in->ignore(current_document_size - already_read_bytes);
}

bool BSONEachRowRowInputFormat::readRow(MutableColumns & columns, RowReadExtension & ext)
{
    size_t num_columns = columns.size();

    read_columns.assign(num_columns, false);
    seen_columns.assign(num_columns, false);

    if (in->eof())
        return false;

    size_t key_index = 0;

    current_document_start = in->count();
    readBinary(current_document_size, *in);
    while (in->count() - current_document_start + sizeof(BSON_DOCUMENT_END) != current_document_size)
    {
        auto type = getBSONType(readBSONType(*in));
        auto name = readBSONKeyName(*in, current_key_name);
        auto index = columnIndex(name, key_index);

        if (index == UNKNOWN_FIELD)
        {
            current_key_name.assign(name.data, name.size);
            skipUnknownField(BSONType(type), current_key_name);
        }
        else
        {
            seen_columns[index] = true;
            read_columns[index] = readField(*columns[index], types[index], BSONType(type));
        }

        ++key_index;
    }

    assertChar(BSON_DOCUMENT_END, *in);

    const auto & header = getPort().getHeader();
    /// Fill non-visited columns with the default values.
    for (size_t i = 0; i < num_columns; ++i)
        if (!seen_columns[i])
            header.getByPosition(i).type->insertDefaultInto(*columns[i]);

    if (format_settings.defaults_for_omitted_fields)
        ext.read_columns = read_columns;
    else
        ext.read_columns.assign(read_columns.size(), true);

    return true;
}

BSONEachRowSchemaReader::BSONEachRowSchemaReader(ReadBuffer & in_, const FormatSettings & settings_)
    : IRowWithNamesSchemaReader(in_, settings_)
{
}

DataTypePtr BSONEachRowSchemaReader::getDataTypeFromBSONField(BSONType type, bool allow_to_skip_unsupported_types, bool & skip)
{
    switch (type)
    {
        case BSONType::DOUBLE:
        {
            in.ignore(sizeof(Float64));
            return makeNullable(std::make_shared<DataTypeFloat64>());
        }
        case BSONType::BOOL:
        {
            in.ignore(sizeof(UInt8));
            return makeNullable(DataTypeFactory::instance().get("Bool"));
        }
        case BSONType::INT64:
        {
            in.ignore(sizeof(Int64));
            return makeNullable(std::make_shared<DataTypeInt64>());
        }
        case BSONType::DATETIME:
        {
            in.ignore(sizeof(Int64));
            return makeNullable(std::make_shared<DataTypeDateTime64>(6, "UTC"));
        }
        case BSONType::UINT64:
        {
            in.ignore(sizeof(UInt64));
            return makeNullable(std::make_shared<DataTypeUInt64>());
        }
        case BSONType::INT32:
        {
            in.ignore(sizeof(Int32));
            return makeNullable(std::make_shared<DataTypeInt32>());
        }
        case BSONType::SYMBOL: [[fallthrough]];
        case BSONType::STRING:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            in.ignore(size);
            return makeNullable(std::make_shared<DataTypeString>());
        }
        case BSONType::DOCUMENT:
        {
            auto nested_names_and_types = getDataTypesFromBSONDocument(false);
            auto nested_types = nested_names_and_types.getTypes();
            bool types_are_equal = true;
            if (nested_types.empty() || !nested_types[0])
                return nullptr;

            for (size_t i = 1; i != nested_types.size(); ++i)
            {
                if (!nested_types[i])
                    return nullptr;

                types_are_equal &= nested_types[i]->equals(*nested_types[0]);
            }

            if (types_are_equal)
                return std::make_shared<DataTypeMap>(std::make_shared<DataTypeString>(), nested_types[0]);

            return std::make_shared<DataTypeTuple>(std::move(nested_types), nested_names_and_types.getNames());

        }
        case BSONType::ARRAY:
        {
            auto nested_types = getDataTypesFromBSONDocument(false).getTypes();
            bool types_are_equal = true;
            if (nested_types.empty() || !nested_types[0])
                return nullptr;

            for (size_t i = 1; i != nested_types.size(); ++i)
            {
                if (!nested_types[i])
                    return nullptr;

                types_are_equal &= nested_types[i]->equals(*nested_types[0]);
            }

            if (types_are_equal)
                return std::make_shared<DataTypeArray>(nested_types[0]);

            return std::make_shared<DataTypeTuple>(std::move(nested_types));
        }
        case BSONType::BINARY:
        {
            BSON_SIZE_TYPE size;
            readBinary(size, in);
            auto subtype = getBSONBinarySubtype(readBSONType(in));
            in.ignore(size);
            switch (subtype)
            {
                case BSONBinarySubtype::BINARY_OLD: [[fallthrough]];
                case BSONBinarySubtype::BINARY:
                    return makeNullable(std::make_shared<DataTypeString>());
                case BSONBinarySubtype::UUID_OLD: [[fallthrough]];
                case BSONBinarySubtype::UUID:
                    return makeNullable(std::make_shared<DataTypeUUID>());
                default:
                    throw Exception(ErrorCodes::UNKNOWN_TYPE, "BSON binary subtype {} is not supported", getBSONBinarySubtypeName(subtype));
            }
        }
        case BSONType::NULL_VALUE:
        {
            return nullptr;
        }
        default:
        {
            if (!allow_to_skip_unsupported_types)
                throw Exception(ErrorCodes::UNKNOWN_TYPE, "BSON type {} is not supported", getBSONTypeName(type));

            skip = true;
            skipBSONField(in, type);
            return nullptr;
        }
    }
}

NamesAndTypesList BSONEachRowSchemaReader::getDataTypesFromBSONDocument(bool allow_to_skip_unsupported_types)
{
    size_t document_start = in.count();
    BSON_SIZE_TYPE document_size;
    readBinary(document_size, in);
    NamesAndTypesList names_and_types;
    while (in.count() - document_start + sizeof(BSON_DOCUMENT_END) != document_size)
    {
        auto bson_type = getBSONType(readBSONType(in));
        String name;
        readNullTerminated(name, in);
        bool skip = false;
        auto type = getDataTypeFromBSONField(bson_type, allow_to_skip_unsupported_types, skip);
        if (!skip)
            names_and_types.emplace_back(name, type);
    }

    assertChar(BSON_DOCUMENT_END, in);

    return names_and_types;
}

NamesAndTypesList BSONEachRowSchemaReader::readRowAndGetNamesAndDataTypes(bool & eof)
{
    if (in.eof())
    {
        eof = true;
        return {};
    }

    return getDataTypesFromBSONDocument(format_settings.bson.skip_fields_with_unsupported_types_in_schema_inference);
}

void BSONEachRowSchemaReader::transformTypesIfNeeded(DataTypePtr & type, DataTypePtr & new_type)
{
    DataTypes types = {type, new_type};
    /// For example for integer conversion Int32,
    auto least_supertype = tryGetLeastSupertype(types);
    if (least_supertype)
        type = new_type = least_supertype;
}

static std::pair<bool, size_t>
fileSegmentationEngineBSONEachRow(ReadBuffer & in, DB::Memory<> & memory, size_t min_bytes, size_t max_rows)
{
    size_t number_of_rows = 0;

    while (!in.eof() && memory.size() < min_bytes && number_of_rows < max_rows)
    {
        BSON_SIZE_TYPE document_size;
        readBinary(document_size, in);
        size_t old_size = memory.size();
        memory.resize(old_size + document_size);
        memcpy(memory.data() + old_size, reinterpret_cast<char *>(&document_size), sizeof(document_size));
        in.readStrict(memory.data() + old_size + sizeof(document_size), document_size - sizeof(document_size));
        ++number_of_rows;
    }

    return {!in.eof(), number_of_rows};
}

void registerInputFormatBSONEachRow(FormatFactory & factory)
{
    factory.registerInputFormat(
        "BSONEachRow",
        [](ReadBuffer & buf, const Block & sample, IRowInputFormat::Params params, const FormatSettings & settings)
        { return std::make_shared<BSONEachRowRowInputFormat>(buf, sample, std::move(params), settings); });
}

void registerFileSegmentationEngineBSONEachRow(FormatFactory & factory)
{
    factory.registerFileSegmentationEngine("BSONEachRow", &fileSegmentationEngineBSONEachRow);
}

void registerBSONEachRowSchemaReader(FormatFactory & factory)
{
    factory.registerSchemaReader("BSONEachRow", [](ReadBuffer & buf, const FormatSettings & settings)
    {
        return std::make_unique<BSONEachRowSchemaReader>(buf, settings);
    });
    factory.registerAdditionalInfoForSchemaCacheGetter("BSONEachRow", [](const FormatSettings & settings)
    {
         String result = getAdditionalFormatInfoForAllRowBasedFormats(settings);
         return result + fmt::format(", skip_fields_with_unsupported_types_in_schema_inference={}",
                                     settings.bson.skip_fields_with_unsupported_types_in_schema_inference);
    });
}

}
