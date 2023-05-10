#include <Processors/Formats/Impl/CapnProtoRowOutputFormat.h>
#if USE_CAPNP

#include <Formats/CapnProtoSchema.h>
#include <Formats/FormatSettings.h>
#include <Formats/CapnProtoSerializer.h>
#include <IO/WriteBuffer.h>
#include <capnp/dynamic.h>
#include <capnp/serialize-packed.h>

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
    const FormatSettings & format_settings)
    : IRowOutputFormat(header_, out_)
    , column_names(header_.getNames())
    , column_types(header_.getDataTypes())
    , output_stream(std::make_unique<CapnProtoOutputStream>(out_))
{
    schema = schema_parser.getMessageSchema(info);
    const auto & header = getPort(PortKind::Main).getHeader();
    serializer = std::make_unique<CapnProtoSerializer>(header.getDataTypes(), header.getNames(), schema, format_settings.capn_proto);
    capnp::MallocMessageBuilder message;
}

void CapnProtoRowOutputFormat::write(const Columns & columns, size_t row_num)
{
    capnp::MallocMessageBuilder message;
    capnp::DynamicStruct::Builder root = message.initRoot<capnp::DynamicStruct>(schema);
    serializer->writeRow(columns, std::move(root), row_num);
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
