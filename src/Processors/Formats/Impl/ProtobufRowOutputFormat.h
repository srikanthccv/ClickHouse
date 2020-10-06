#pragma once

#if !defined(ARCADIA_BUILD)
#    include "config_formats.h"
#endif

#if USE_PROTOBUF
#    include <Core/Block.h>
#    include <Formats/FormatSchemaInfo.h>
#    include <Formats/FormatSettings.h>
#    include <Formats/ProtobufWriter.h>
#    include <Processors/Formats/IRowOutputFormat.h>


namespace google
{
namespace protobuf
{
    class Message;
}
}


namespace DB
{
/** Stream designed to serialize data in the google protobuf format.
  * Each row is written as a separated message.
  *
  * With single_message_mode=1 it can write only single row as plain protobuf message,
  * otherwise Protobuf messages are delimited according to documentation
  * https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/util/delimited_message_util.h
  * Serializing in the protobuf format requires the 'format_schema' setting to be set, e.g.
  * SELECT * from table FORMAT Protobuf SETTINGS format_schema = 'schema:Message'
  * where schema is the name of "schema.proto" file specifying protobuf schema.
  */
class ProtobufRowOutputFormat : public IRowOutputFormat
{
public:
    ProtobufRowOutputFormat(
        WriteBuffer & out_,
        const Block & header,
        const RowOutputFormatParams & params,
        const FormatSchemaInfo & format_schema,
        const bool single_message_mode_);

    String getName() const override { return "ProtobufRowOutputFormat"; }

    void write(const Columns & columns, size_t row_num) override;
    void writeField(const IColumn &, const IDataType &, size_t) override {}
    std::string getContentType() const override { return "application/octet-stream"; }

private:
    DataTypes data_types;
    ProtobufWriter writer;
    std::vector<size_t> value_indices;
};

}
#endif
