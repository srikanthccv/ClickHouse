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

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeEnum.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeLowCardinality.h>

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
    const RowOutputFormatParams & params_,
    const FormatSchemaInfo & info,
    const FormatSettings & format_settings_)
    : IRowOutputFormat(header_, out_, params_), column_names(header_.getNames()), column_types(header_.getDataTypes()), output_stream(std::make_unique<CapnProtoOutputStream>(out_)), format_settings(format_settings_)
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
        return struct_builder.init(field, size);
    }

    if (field.getType().isStruct())
        return struct_builder.init(field);

    return struct_builder.get(field);
}

static std::optional<capnp::DynamicValue::Reader> convertToDynamicValue(const ColumnPtr & column, const DataTypePtr & data_type, size_t row_num, capnp::DynamicValue::Builder builder, FormatSettings::EnumComparingMode enum_comparing_mode)
{
    /// Here we don't do any types validation, because we did it in CapnProtoRowOutputFormat constructor.

    if (data_type->lowCardinality())
    {
        const auto * lc_column = assert_cast<const ColumnLowCardinality *>(column.get());
        const auto & dict_type = assert_cast<const DataTypeLowCardinality *>(data_type.get())->getDictionaryType();
        size_t index = lc_column->getIndexAt(row_num);
        return convertToDynamicValue(lc_column->getDictionary().getNestedColumn(), dict_type, index, builder, enum_comparing_mode);
    }

    switch (builder.getType())
    {
        case capnp::DynamicValue::Type::INT:
            /// We allow output DateTime64 as Int64.
            if (WhichDataType(data_type).isDateTime64())
                return capnp::DynamicValue::Reader(assert_cast<const ColumnDecimal<DateTime64> *>(column.get())->getElement(row_num));
            return capnp::DynamicValue::Reader(column->getInt(row_num));
        case capnp::DynamicValue::Type::UINT:
            return capnp::DynamicValue::Reader(column->getUInt(row_num));
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
            auto data = String(column->getDataAt(row_num));
            return capnp::DynamicValue::Reader(capnp::Text::Reader(data.data(), data.size()));
        }
        case capnp::DynamicValue::Type::STRUCT:
        {
            auto struct_builder = builder.as<capnp::DynamicStruct>();
            auto nested_struct_schema = struct_builder.getSchema();
            /// Struct can be represent Tuple or Naullable (named union with two fields)
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
                    auto value = convertToDynamicValue(nested_column, nullable_type->getNestedType(), row_num, value_builder, enum_comparing_mode);
                    if (value)
                        struct_builder.set(value_field, std::move(*value));
                }
            }
            else
            {
                const auto * tuple_data_type = assert_cast<const DataTypeTuple *>(data_type.get());
                auto nested_types = tuple_data_type->getElements();
                const auto & nested_columns = assert_cast<const ColumnTuple *>(column.get())->getColumns();
                for (const auto & name : tuple_data_type->getElementNames())
                {
                    auto pos = tuple_data_type->getPositionByName(name);
                    auto field_builder
                        = initStructFieldBuilder(nested_columns[pos], row_num, struct_builder, nested_struct_schema.getFieldByName(name));
                    auto value = convertToDynamicValue(nested_columns[pos], nested_types[pos], row_num, field_builder, enum_comparing_mode);
                    if (value)
                        struct_builder.set(name, std::move(*value));
                }
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
            for (size_t i = 0; i != size; ++i)
            {
                capnp::DynamicValue::Builder value_builder;
                /// For nested arrays we need to initialize nested list builder.
                if (nested_array_column)
                {
                    const auto & nested_offset = nested_array_column->getOffsets();
                    size_t nested_array_size = nested_offset[offset + i] - nested_offset[offset + i - 1];
                    value_builder = list_builder.init(i, nested_array_size);
                }
                else
                    value_builder = list_builder[i];

                auto value = convertToDynamicValue(nested_column, nested_type, offset + i, value_builder, enum_comparing_mode);
                if (value)
                    list_builder.set(i, std::move(*value));
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
    capnp::DynamicStruct::Builder root = message.initRoot<capnp::DynamicStruct>(schema);
    for (size_t i = 0; i != columns.size(); ++i)
    {
        auto [struct_builder, field] = getStructBuilderAndFieldByColumnName(root, column_names[i]);
        auto field_builder = initStructFieldBuilder(columns[i], row_num, struct_builder, field);
        auto value = convertToDynamicValue(columns[i], column_types[i], row_num, field_builder, format_settings.capn_proto.enum_comparing_mode);
        if (value)
            struct_builder.set(field, std::move(*value));
    }

    capnp::writeMessage(*output_stream, message);
}

void registerOutputFormatProcessorsCapnProto(FormatFactory & factory)
{
    factory.registerOutputFormatProcessor("CapnProto", [](
        WriteBuffer & buf,
        const Block & sample,
        const RowOutputFormatParams & params,
        const FormatSettings & format_settings)
    {
        return std::make_shared<CapnProtoRowOutputFormat>(buf, sample, params, FormatSchemaInfo(format_settings, "CapnProto", true), format_settings);
    });
}

}

#else

namespace DB
{
class FormatFactory;
void registerOutputFormatProcessorsCapnProto(FormatFactory &) {}
}

#endif // USE_CAPNP
