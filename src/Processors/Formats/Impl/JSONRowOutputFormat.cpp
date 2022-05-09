#include <IO/WriteHelpers.h>
#include <IO/WriteBufferValidUTF8.h>
#include <Processors/Formats/Impl/JSONRowOutputFormat.h>
#include <Formats/FormatFactory.h>
#include <Formats/JSONUtils.h>


namespace DB
{

JSONRowOutputFormat::JSONRowOutputFormat(
    WriteBuffer & out_,
    const Block & header,
    const RowOutputFormatParams & params_,
    const FormatSettings & settings_,
    bool yield_strings_)
    : IRowOutputFormat(header, out_, params_), settings(settings_), yield_strings(yield_strings_)
{
    bool need_validate_utf8 = false;
    fields = header.getNamesAndTypes();
    makeNamesAndTypesWithValidUTF8(fields, settings, need_validate_utf8);

    if (need_validate_utf8)
    {
        validating_ostr = std::make_unique<WriteBufferValidUTF8>(out);
        ostr = validating_ostr.get();
    }
    else
        ostr = &out;
}


void JSONRowOutputFormat::writePrefix()
{
    writeJSONObjectStart(*ostr);
    writeJSONMetadata(fields, settings, *ostr);
    writeJSONFieldDelimiter(*ostr, 2);
    writeJSONArrayStart(*ostr, 1, "data");
}


void JSONRowOutputFormat::writeField(const IColumn & column, const ISerialization & serialization, size_t row_num)
{
    writeJSONFieldFromColumn(column, serialization, row_num, yield_strings, settings, *ostr, fields[field_number].name, 3);
    ++field_number;
}

void JSONRowOutputFormat::writeFieldDelimiter()
{
    writeJSONFieldDelimiter(*ostr);
}


void JSONRowOutputFormat::writeRowStartDelimiter()
{
    writeJSONObjectStart(*ostr, 2);
}


void JSONRowOutputFormat::writeRowEndDelimiter()
{
    writeJSONObjectEnd(*ostr, 2);
    field_number = 0;
    ++row_count;
}


void JSONRowOutputFormat::writeRowBetweenDelimiter()
{
    writeJSONFieldDelimiter(*ostr);
}


void JSONRowOutputFormat::writeSuffix()
{
    writeJSONArrayEnd(*ostr, 1);
}

void JSONRowOutputFormat::writeBeforeTotals()
{
    writeJSONFieldDelimiter(*ostr, 2);
    writeJSONObjectStart(*ostr, 1, "totals");
}

void JSONRowOutputFormat::writeTotals(const Columns & columns, size_t row_num)
{
    writeJSONColumns(columns, fields, serializations, row_num, yield_strings, settings, *ostr, 2);
}

void JSONRowOutputFormat::writeAfterTotals()
{
    writeJSONObjectEnd(*ostr, 1);
}

void JSONRowOutputFormat::writeBeforeExtremes()
{
    writeJSONFieldDelimiter(*ostr, 2);
    writeJSONObjectStart(*ostr, 1, "extremes");
}

void JSONRowOutputFormat::writeExtremesElement(const char * title, const Columns & columns, size_t row_num)
{
    writeJSONObjectStart(*ostr, 2, title);
    writeJSONColumns(columns, fields, serializations, row_num, yield_strings, settings, *ostr, 3);
    writeJSONObjectEnd(*ostr, 2);
}

void JSONRowOutputFormat::writeMinExtreme(const Columns & columns, size_t row_num)
{
    writeExtremesElement("min", columns, row_num);
}

void JSONRowOutputFormat::writeMaxExtreme(const Columns & columns, size_t row_num)
{
    writeExtremesElement("max", columns, row_num);
}

void JSONRowOutputFormat::writeAfterExtremes()
{
    writeJSONObjectEnd(*ostr, 1);
}

void JSONRowOutputFormat::finalizeImpl()
{
    auto outside_statistics = getOutsideStatistics();
    if (outside_statistics)
        statistics = std::move(*outside_statistics);

    writeJSONAdditionalInfo(
        row_count,
        statistics.rows_before_limit,
        statistics.applied_limit,
        statistics.watch,
        statistics.progress,
        settings.write_statistics,
        *ostr);

    writeJSONObjectEnd(*ostr);
    writeChar('\n', *ostr);
    ostr->next();
}


void JSONRowOutputFormat::onProgress(const Progress & value)
{
    statistics.progress.incrementPiecewiseAtomically(value);
}


void registerOutputFormatJSON(FormatFactory & factory)
{
    factory.registerOutputFormat("JSON", [](
        WriteBuffer & buf,
        const Block & sample,
        const RowOutputFormatParams & params,
        const FormatSettings & format_settings)
    {
        return std::make_shared<JSONRowOutputFormat>(buf, sample, params, format_settings, false);
    });

    factory.markOutputFormatSupportsParallelFormatting("JSON");
    factory.markFormatHasNoAppendSupport("JSON");

    factory.registerOutputFormat("JSONStrings", [](
        WriteBuffer & buf,
        const Block & sample,
        const RowOutputFormatParams & params,
        const FormatSettings & format_settings)
    {
        return std::make_shared<JSONRowOutputFormat>(buf, sample, params, format_settings, true);
    });

    factory.markOutputFormatSupportsParallelFormatting("JSONStrings");
    factory.markFormatHasNoAppendSupport("JSONStrings");
}

}
