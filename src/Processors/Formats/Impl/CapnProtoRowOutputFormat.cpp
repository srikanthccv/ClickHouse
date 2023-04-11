#include <Processors/Formats/Impl/CapnProtoRowOutputFormat.h>
#if USE_CAPNP

#include <Formats/CapnProtoUtils.h>
#include <Formats/FormatSettings.h>
#include <IO/WriteBuffer.h>
#include <capnp/dynamic.h>
#include <capnp/serialize-packed.h>

#include <Columns/ColumnArray.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnMap.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeEnum.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeMap.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}


CapnProtoOutputStream::CapnProtoOutputStream(WriteBuffer & out_) : out(out_)
{
}

void CapnProtoOutputStream::write(const void * buffer, size_t size)
{
    out.write(reinterpret_cast<const char *>(buffer), size);
}

CapnProtoRowOutputFormat::CapnProtoRowOutputFormat(
    WriteBuffer & out_,
    const Block & header_,
    const FormatSchemaInfo & info,
    const FormatSettings & format_settings_)
    : IRowOutputFormat(header_, out_), column_names(header_.getNames()), column_types(header_.getDataTypes()), output_stream(std::make_unique<CapnProtoOutputStream>(out_)), format_settings(format_settings_)
{
    schema = schema_parser.getMessageSchema(info);
    checkCapnProtoSchemaStructure(schema, getPort(PortKind::Main).getHeader(), format_settings.capn_proto.enum_comparing_mode);
}

template <typename EnumValue>
static capnp::DynamicEnum getDynamicEnum(
    const ColumnPtr & column,
    const DataTypePtr & data_type,
    size_t row_num,
    const capnp::EnumSchema & enum_schema,
    FormatSettings::EnumComparingMode mode)
{
    const auto * enum_data_type = assert_cast<const DataTypeEnum<EnumValue> *>(data_type.get());
    EnumValue enum_value = column->getInt(row_num);
    if (mode == FormatSettings::EnumComparingMode::BY_VALUES)
        return capnp::DynamicEnum(enum_schema, enum_value);

    auto enum_name = enum_data_type->getNameForValue(enum_value);
    for (const auto enumerant : enum_schema.getEnumerants())
    {
        if (compareEnumNames(String(enum_name), enumerant.getProto().getName(), mode))
            return capnp::DynamicEnum(enumerant);
    }

    throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot convert CLickHouse Enum value to CapnProto Enum");
}

static capnp::DynamicValue::Builder initStructFieldBuilder(const ColumnPtr & column, size_t row_num, capnp::DynamicStruct::Builder & struct_builder, capnp::StructSchema::Field field)
{
    if (const auto * array_column = checkAndGetColumn<ColumnArray>(*column))
    {
        size_t size = array_column->getOffsets()[row_num] - array_column->getOffsets()[row_num - 1];
        return struct_builder.init(field, static_cast<unsigned>(size));
    }

    if (field.getType().isStruct())
        return struct_builder.init(field);

    return struct_builder.get(field);
}

static std::optional<capnp::DynamicValue::Reader> convertToDynamicValue(
    const ColumnPtr & column,
    const DataTypePtr & data_type,
    size_t row_num,
    const String & column_name,
    capnp::DynamicValue::Builder builder,
    FormatSettings::EnumComparingMode enum_comparing_mode,
    std::vector<std::unique_ptr<String>> & temporary_text_data_storage)
{
    /// Here we don't do any types validation, because we did it in CapnProtoRowOutputFormat constructor.

    if (data_type->lowCardinality())
    {
        const auto * lc_column = assert_cast<const ColumnLowCardinality *>(column.get());
        const auto & dict_type = assert_cast<const DataTypeLowCardinality *>(data_type.get())->getDictionaryType();
        size_t index = lc_column->getIndexAt(row_num);
        return convertToDynamicValue(lc_column->getDictionary().getNestedColumn(), dict_type, index, column_name, builder, enum_comparing_mode, temporary_text_data_storage);
    }

    switch (builder.getType())
    {
        case capnp::DynamicValue::Type::INT:
            return capnp::DynamicValue::Reader(column->getInt(row_num));
        case capnp::DynamicValue::Type::UINT:
        {
            /// IPv4 column doesn't support getUInt method.
            if (isIPv4(data_type))
                return capnp::DynamicValue::Reader(assert_cast<const ColumnIPv4 *>(column.get())->getElement(row_num));
            return capnp::DynamicValue::Reader(column->getUInt(row_num));
        }
        case capnp::DynamicValue::Type::BOOL:
            return capnp::DynamicValue::Reader(column->getBool(row_num));
        case capnp::DynamicValue::Type::FLOAT:
            return capnp::DynamicValue::Reader(column->getFloat64(row_num));
        case capnp::DynamicValue::Type::ENUM:
        {
            auto enum_schema = builder.as<capnp::DynamicEnum>().getSchema();
            if (data_type->getTypeId() == TypeIndex::Enum8)
                return capnp::DynamicValue::Reader(
                    getDynamicEnum<Int8>(column, data_type, row_num, enum_schema, enum_comparing_mode));
            return capnp::DynamicValue::Reader(
                    getDynamicEnum<Int16>(column, data_type, row_num, enum_schema, enum_comparing_mode));
        }
        case capnp::DynamicValue::Type::DATA:
        {
            auto data = column->getDataAt(row_num);
            return capnp::DynamicValue::Reader(capnp::Data::Reader(reinterpret_cast<const kj::byte *>(data.data), data.size));
        }
        case capnp::DynamicValue::Type::TEXT:
        {
            /// In TEXT type data should be null-terminated, but ClickHouse String data could not be.
            /// To make data null-terminated we should copy it to temporary String object, but
            /// capnp::Text::Reader works only with pointer to the data and it's size, so we should
            /// guarantee that new String object life time is longer than capnp::Text::Reader life time.
            /// To do this we store new String object in a temporary storage, passed in this function
            /// by reference. We use unique_ptr<String> instead of just String to avoid pointers
            /// invalidation on vector reallocation.
            temporary_text_data_storage.push_back(std::make_unique<String>(column->getDataAt(row_num)));
            auto & data = temporary_text_data_storage.back();
            return capnp::DynamicValue::Reader(capnp::Text::Reader(data->data(), data->size()));
        }
        case capnp::DynamicValue::Type::STRUCT:
        {
            auto struct_builder = builder.as<capnp::DynamicStruct>();
            auto nested_struct_schema = struct_builder.getSchema();
            /// Struct can represent Tuple, Nullable (named union with two fields) or single column when it contains one nested column.
            if (data_type->isNullable())
            {
                const auto * nullable_type = assert_cast<const DataTypeNullable *>(data_type.get());
                const auto * nullable_column = assert_cast<const ColumnNullable *>(column.get());
                auto fields = nested_struct_schema.getUnionFields();
                if (nullable_column->isNullAt(row_num))
                {
                    auto null_field = fields[0].getType().isVoid() ? fields[0] : fields[1];
                    struct_builder.set(null_field, capnp::Void());
                }
                else
                {
                    auto value_field = fields[0].getType().isVoid() ? fields[1] : fields[0];
                    struct_builder.clear(value_field);
                    const auto & nested_column = nullable_column->getNestedColumnPtr();
                    auto value_builder = initStructFieldBuilder(nested_column, row_num, struct_builder, value_field);
                    auto value = convertToDynamicValue(nested_column, nullable_type->getNestedType(), row_num, column_name, value_builder, enum_comparing_mode, temporary_text_data_storage);
                    if (value)
                        struct_builder.set(value_field, *value);
                }
            }
            else if (isTuple(data_type))
            {
                const auto * tuple_data_type = assert_cast<const DataTypeTuple *>(data_type.get());
                const auto & nested_types = tuple_data_type->getElements();
                const auto & nested_names = tuple_data_type->getElementNames();
                const auto & nested_columns = assert_cast<const ColumnTuple *>(column.get())->getColumns();
                bool have_explicit_names = tuple_data_type->haveExplicitNames();
                for (uint32_t i = 0; i != nested_names.size(); ++i)
                {
                    capnp::StructSchema::Field nested_field = have_explicit_names ? nested_struct_schema.getFieldByName(nested_names[i]) : nested_struct_schema.getFields()[i];
                    auto field_builder = initStructFieldBuilder(nested_columns[i], row_num, struct_builder, nested_field);
                    auto value = convertToDynamicValue(nested_columns[i], nested_types[i], row_num, nested_names[i], field_builder, enum_comparing_mode, temporary_text_data_storage);
                    if (value)
                        struct_builder.set(nested_field, *value);
                }
            }
            else if (isMap(data_type))
            {
                /// We output Map type as follow CapnProto schema
                ///
                /// struct Map {
                ///     struct Entry {
                ///         key @0: Key;
                ///         value @1: Value;
                ///     }
                ///     entries @0 :List(Entry);
                /// }
                ///
                /// And we don't need to check that struct have this form here because we checked it before.
                const auto & map_type = assert_cast<const DataTypeMap &>(*data_type);
                DataTypes key_value_types = {map_type.getKeyType(), map_type.getValueType()};
                Names key_value_names = {"key", "value"};
                auto entries_type = std::make_shared<DataTypeArray>(std::make_shared<DataTypeTuple>(key_value_types, key_value_names));

                /// Nested column in Map is actually Array(Tuple), so we can output it according to "entries" field schema.
                const auto & entries_column = assert_cast<const ColumnMap *>(column.get())->getNestedColumnPtr();

                auto entries_field = nested_struct_schema.getFields()[0];
                auto field_builder = initStructFieldBuilder(entries_column, row_num, struct_builder, entries_field);
                auto entries_value = convertToDynamicValue(entries_column, entries_type, row_num, column_name, field_builder, enum_comparing_mode, temporary_text_data_storage);
                if (entries_value)
                    struct_builder.set(entries_field, *entries_value);
            }
            else
            {
                /// It can be nested column from Nested type.
                auto [field_name, nested_name] = splitCapnProtoFieldName(column_name);
                auto nested_field = nested_struct_schema.getFieldByName(nested_name);
                auto field_builder = initStructFieldBuilder(column, row_num, struct_builder, nested_field);
                auto value = convertToDynamicValue(column, data_type, row_num, nested_name, field_builder, enum_comparing_mode, temporary_text_data_storage);
                if (value)
                    struct_builder.set(nested_field, *value);
            }
            return std::nullopt;
        }
        case capnp::DynamicValue::Type::LIST:
        {
            auto list_builder = builder.as<capnp::DynamicList>();
            const auto * array_column = assert_cast<const ColumnArray *>(column.get());
            const auto & nested_column = array_column->getDataPtr();
            const auto & nested_type = assert_cast<const DataTypeArray *>(data_type.get())->getNestedType();
            const auto & offsets = array_column->getOffsets();
            auto offset = offsets[row_num - 1];
            size_t size = offsets[row_num] - offset;

            const auto * nested_array_column = checkAndGetColumn<ColumnArray>(*nested_column);
            for (unsigned i = 0; i != static_cast<unsigned>(size); ++i)
            {
                capnp::DynamicValue::Builder value_builder;
                /// For nested arrays we need to initialize nested list builder.
                if (nested_array_column)
                {
                    const auto & nested_offset = nested_array_column->getOffsets();
                    size_t nested_array_size = nested_offset[offset + i] - nested_offset[offset + i - 1];
                    value_builder = list_builder.init(i, static_cast<unsigned>(nested_array_size));
                }
                else
                    value_builder = list_builder[i];

                auto value = convertToDynamicValue(nested_column, nested_type, offset + i, column_name, value_builder, enum_comparing_mode, temporary_text_data_storage);
                if (value)
                    list_builder.set(i, *value);
            }
            return std::nullopt;
        }
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected CapnProto type.");
    }
}

void CapnProtoRowOutputFormat::write(const Columns & columns, size_t row_num)
{
    capnp::MallocMessageBuilder message;
    /// Temporary storage for data that will be outputted in fields with CapnProto type TEXT.
    /// See comment in convertToDynamicValue() for more details.
    std::vector<std::unique_ptr<String>> temporary_text_data_storage;
    capnp::DynamicStruct::Builder root = message.initRoot<capnp::DynamicStruct>(schema);

    /// Some columns can share same field builder. For example when we have
    /// column with Nested type that was flattened into several columns.
    std::unordered_map<size_t, capnp::DynamicValue::Builder> field_builders;
    for (size_t i = 0; i != columns.size(); ++i)
    {
        auto [struct_builder, field] = getStructBuilderAndFieldByColumnName(root, column_names[i]);
        if (!field_builders.contains(field.getIndex()))
        {
            auto field_builder = initStructFieldBuilder(columns[i], row_num, struct_builder, field);
            field_builders[field.getIndex()] = field_builder;
        }
        auto value = convertToDynamicValue(columns[i], column_types[i], row_num, column_names[i], field_builders[field.getIndex()], format_settings.capn_proto.enum_comparing_mode, temporary_text_data_storage);
        if (value)
            struct_builder.set(field, *value);
    }

    capnp::writeMessage(*output_stream, message);
}

void registerOutputFormatCapnProto(FormatFactory & factory)
{
    factory.registerOutputFormat("CapnProto", [](
        WriteBuffer & buf,
        const Block & sample,
        const FormatSettings & format_settings)
    {
        return std::make_shared<CapnProtoRowOutputFormat>(buf, sample, FormatSchemaInfo(format_settings, "CapnProto", true), format_settings);
    });
}

}

#else

namespace DB
{
class FormatFactory;
void registerOutputFormatCapnProto(FormatFactory &) {}
}

#endif // USE_CAPNP
