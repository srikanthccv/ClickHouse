#include <Formats/FormatFactory.h>

#include <algorithm>
#include <Common/Exception.h>
#include <Interpreters/Context.h>
#include <Core/Settings.h>
#include <DataStreams/MaterializingBlockOutputStream.h>
#include <DataStreams/ParallelParsingBlockInputStream.h>
#include <Formats/FormatSettings.h>
#include <Processors/Formats/IRowInputFormat.h>
#include <Processors/Formats/IRowOutputFormat.h>
#include <Processors/Formats/InputStreamFromInputFormat.h>
#include <Processors/Formats/OutputStreamToOutputFormat.h>
#include <DataStreams/NativeBlockInputStream.h>
#include <Processors/Formats/Impl/ValuesBlockInputFormat.h>
#include <Processors/Formats/Impl/MySQLOutputFormat.h>
#include <Processors/Formats/Impl/PostgreSQLOutputFormat.h>
#include <Poco/URI.h>

#if !defined(ARCADIA_BUILD)
#    include <Common/config.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int UNKNOWN_FORMAT;
    extern const int LOGICAL_ERROR;
    extern const int FORMAT_IS_NOT_SUITABLE_FOR_INPUT;
    extern const int FORMAT_IS_NOT_SUITABLE_FOR_OUTPUT;
}

const FormatFactory::Creators & FormatFactory::getCreators(const String & name) const
{
    auto it = dict.find(name);
    if (dict.end() != it)
        return it->second;
    throw Exception("Unknown format " + name, ErrorCodes::UNKNOWN_FORMAT);
}

FormatSettings getFormatSettings(const Context & context)
{
    const auto & settings = context.getSettingsRef();

    return getFormatSettings(context, settings);
}

template <typename Settings>
FormatSettings getFormatSettings(const Context & context,
    const Settings & settings)
{
    FormatSettings format_settings;

    format_settings.avro.allow_missing_fields = settings.input_format_avro_allow_missing_fields;
    format_settings.avro.output_codec = settings.output_format_avro_codec;
    format_settings.avro.output_sync_interval = settings.output_format_avro_sync_interval;
    format_settings.avro.schema_registry_url = settings.format_avro_schema_registry_url.toString();
    format_settings.csv.allow_double_quotes = settings.format_csv_allow_double_quotes;
    format_settings.csv.allow_single_quotes = settings.format_csv_allow_single_quotes;
    format_settings.csv.crlf_end_of_line = settings.output_format_csv_crlf_end_of_line;
    format_settings.csv.delimiter = settings.format_csv_delimiter;
    format_settings.csv.empty_as_default = settings.input_format_defaults_for_omitted_fields;
    format_settings.csv.input_format_enum_as_number = settings.input_format_csv_enum_as_number;
    format_settings.csv.unquoted_null_literal_as_null = settings.input_format_csv_unquoted_null_literal_as_null;
    format_settings.custom.escaping_rule = settings.format_custom_escaping_rule;
    format_settings.custom.field_delimiter = settings.format_custom_field_delimiter;
    format_settings.custom.result_after_delimiter = settings.format_custom_result_after_delimiter;
    format_settings.custom.result_after_delimiter = settings.format_custom_result_after_delimiter;
    format_settings.custom.result_before_delimiter = settings.format_custom_result_before_delimiter;
    format_settings.custom.row_after_delimiter = settings.format_custom_row_after_delimiter;
    format_settings.custom.row_before_delimiter = settings.format_custom_row_before_delimiter;
    format_settings.custom.row_between_delimiter = settings.format_custom_row_between_delimiter;
    format_settings.date_time_input_format = settings.date_time_input_format;
    format_settings.date_time_output_format = settings.date_time_output_format;
    format_settings.enable_streaming = settings.output_format_enable_streaming;
    format_settings.import_nested_json = settings.input_format_import_nested_json;
    format_settings.input_allow_errors_num = settings.input_format_allow_errors_num;
    format_settings.input_allow_errors_ratio = settings.input_format_allow_errors_ratio;
    format_settings.json.array_of_rows = settings.output_format_json_array_of_rows;
    format_settings.json.escape_forward_slashes = settings.output_format_json_escape_forward_slashes;
    format_settings.json.quote_64bit_integers = settings.output_format_json_quote_64bit_integers;
    format_settings.json.quote_denormals = settings.output_format_json_quote_denormals;
    format_settings.null_as_default = settings.input_format_null_as_default;
    format_settings.parquet.row_group_size = settings.output_format_parquet_row_group_size;
    format_settings.pretty.charset = settings.output_format_pretty_grid_charset.toString() == "ASCII" ? FormatSettings::Pretty::Charset::ASCII : FormatSettings::Pretty::Charset::UTF8;
    format_settings.pretty.color = settings.output_format_pretty_color;
    format_settings.pretty.max_column_pad_width = settings.output_format_pretty_max_column_pad_width;
    format_settings.pretty.max_rows = settings.output_format_pretty_max_rows;
    format_settings.pretty.max_value_width = settings.output_format_pretty_max_value_width;
    format_settings.pretty.output_format_pretty_row_numbers = settings.output_format_pretty_row_numbers;
    format_settings.regexp.escaping_rule = settings.format_regexp_escaping_rule;
    format_settings.regexp.regexp = settings.format_regexp;
    format_settings.regexp.skip_unmatched = settings.format_regexp_skip_unmatched;
    format_settings.schema.format_schema = settings.format_schema;
    format_settings.schema.format_schema_path = context.getFormatSchemaPath();
    format_settings.schema.is_server = context.hasGlobalContext() && (context.getGlobalContext().getApplicationType() == Context::ApplicationType::SERVER);
    format_settings.skip_unknown_fields = settings.input_format_skip_unknown_fields;
    format_settings.template_settings.resultset_format = settings.format_template_resultset;
    format_settings.template_settings.row_between_delimiter = settings.format_template_rows_between_delimiter;
    format_settings.template_settings.row_format = settings.format_template_row;
    format_settings.tsv.crlf_end_of_line = settings.output_format_tsv_crlf_end_of_line;
    format_settings.tsv.empty_as_default = settings.input_format_tsv_empty_as_default;
    format_settings.tsv.input_format_enum_as_number = settings.input_format_tsv_enum_as_number;
    format_settings.tsv.null_representation = settings.output_format_tsv_null_representation;
    format_settings.values.accurate_types_of_literals = settings.input_format_values_accurate_types_of_literals;
    format_settings.values.deduce_templates_of_expressions = settings.input_format_values_deduce_templates_of_expressions;
    format_settings.values.interpret_expressions = settings.input_format_values_interpret_expressions;
    format_settings.with_names_use_header = settings.input_format_with_names_use_header;
    format_settings.write_statistics = settings.output_format_write_statistics;

    /// Validate avro_schema_registry_url with RemoteHostFilter when non-empty and in Server context
    if (format_settings.schema.is_server)
    {
        const Poco::URI & avro_schema_registry_url = settings.format_avro_schema_registry_url;
        if (!avro_schema_registry_url.empty())
            context.getRemoteHostFilter().checkURL(avro_schema_registry_url);
    }

    return format_settings;
}

template
FormatSettings getFormatSettings<FormatFactorySettings>(const Context & context,
    const FormatFactorySettings & settings);

template
FormatSettings getFormatSettings<Settings>(const Context & context,
    const Settings & settings);


BlockInputStreamPtr FormatFactory::getInput(
    const String & name,
    ReadBuffer & buf,
    const Block & sample,
    const Context & context,
    UInt64 max_block_size,
    const std::optional<FormatSettings> & _format_settings) const
{
    if (name == "Native")
        return std::make_shared<NativeBlockInputStream>(buf, sample, 0);

    auto format_settings = _format_settings
        ? *_format_settings : getFormatSettings(context);

    if (!getCreators(name).input_processor_creator)
    {
        const auto & input_getter = getCreators(name).input_creator;
        if (!input_getter)
            throw Exception("Format " + name + " is not suitable for input", ErrorCodes::FORMAT_IS_NOT_SUITABLE_FOR_INPUT);


        return input_getter(buf, sample, max_block_size, {}, format_settings);
    }

    const Settings & settings = context.getSettingsRef();
    const auto & file_segmentation_engine = getCreators(name).file_segmentation_engine;

    // Doesn't make sense to use parallel parsing with less than four threads
    // (segmentator + two parsers + reader).
    bool parallel_parsing = settings.input_format_parallel_parsing && file_segmentation_engine && settings.max_threads >= 4;

    if (settings.min_chunk_bytes_for_parallel_parsing * settings.max_threads * 2 > settings.max_memory_usage)
        parallel_parsing = false;

    if (parallel_parsing && name == "JSONEachRow")
    {
        /// FIXME ParallelParsingBlockInputStream doesn't support formats with non-trivial readPrefix() and readSuffix()

        /// For JSONEachRow we can safely skip whitespace characters
        skipWhitespaceIfAny(buf);
        if (buf.eof() || *buf.position() == '[')
            parallel_parsing = false; /// Disable it for JSONEachRow if data is in square brackets (see JSONEachRowRowInputFormat)
    }

    if (parallel_parsing)
    {
        const auto & input_getter = getCreators(name).input_processor_creator;
        if (!input_getter)
            throw Exception("Format " + name + " is not suitable for input", ErrorCodes::FORMAT_IS_NOT_SUITABLE_FOR_INPUT);

        RowInputFormatParams row_input_format_params;
        row_input_format_params.max_block_size = max_block_size;
        row_input_format_params.allow_errors_num = format_settings.input_allow_errors_num;
        row_input_format_params.allow_errors_ratio = format_settings.input_allow_errors_ratio;
        row_input_format_params.max_execution_time = settings.max_execution_time;
        row_input_format_params.timeout_overflow_mode = settings.timeout_overflow_mode;

        auto input_creator_params =
            ParallelParsingBlockInputStream::InputCreatorParams{sample,
                row_input_format_params, format_settings};
        ParallelParsingBlockInputStream::Params params{buf, input_getter,
            input_creator_params, file_segmentation_engine,
            settings.max_threads,
            settings.min_chunk_bytes_for_parallel_parsing};
        return std::make_shared<ParallelParsingBlockInputStream>(params);
    }

    auto format = getInputFormat(name, buf, sample, context, max_block_size,
        format_settings);
    return std::make_shared<InputStreamFromInputFormat>(std::move(format));
}


BlockOutputStreamPtr FormatFactory::getOutput(const String & name,
    WriteBuffer & buf, const Block & sample, const Context & context,
    WriteCallback callback, const std::optional<FormatSettings> & _format_settings) const
{
    auto format_settings = _format_settings
        ? *_format_settings : getFormatSettings(context);

    if (!getCreators(name).output_processor_creator)
    {
        const auto & output_getter = getCreators(name).output_creator;
        if (!output_getter)
            throw Exception("Format " + name + " is not suitable for output", ErrorCodes::FORMAT_IS_NOT_SUITABLE_FOR_OUTPUT);

        /**  Materialization is needed, because formats can use the functions `IDataType`,
          *  which only work with full columns.
          */
        return std::make_shared<MaterializingBlockOutputStream>(
            output_getter(buf, sample, std::move(callback), format_settings),
            sample);
    }

    auto format = getOutputFormat(name, buf, sample, context, std::move(callback),
        format_settings);
    return std::make_shared<MaterializingBlockOutputStream>(
        std::make_shared<OutputStreamToOutputFormat>(format), sample);
}


InputFormatPtr FormatFactory::getInputFormat(
    const String & name,
    ReadBuffer & buf,
    const Block & sample,
    const Context & context,
    UInt64 max_block_size,
    const std::optional<FormatSettings> & _format_settings) const
{
    const auto & input_getter = getCreators(name).input_processor_creator;
    if (!input_getter)
        throw Exception("Format " + name + " is not suitable for input", ErrorCodes::FORMAT_IS_NOT_SUITABLE_FOR_INPUT);

    const Settings & settings = context.getSettingsRef();

    auto format_settings = _format_settings
        ? *_format_settings : getFormatSettings(context);

    RowInputFormatParams params;
    params.max_block_size = max_block_size;
    params.allow_errors_num = format_settings.input_allow_errors_num;
    params.allow_errors_ratio = format_settings.input_allow_errors_ratio;
    params.max_execution_time = settings.max_execution_time;
    params.timeout_overflow_mode = settings.timeout_overflow_mode;

    auto format = input_getter(buf, sample, params, format_settings);


    /// It's a kludge. Because I cannot remove context from values format.
    if (auto * values = typeid_cast<ValuesBlockInputFormat *>(format.get()))
        values->setContext(context);

    return format;
}


OutputFormatPtr FormatFactory::getOutputFormat(
    const String & name, WriteBuffer & buf, const Block & sample,
    const Context & context, WriteCallback callback,
    const std::optional<FormatSettings> & _format_settings) const
{
    const auto & output_getter = getCreators(name).output_processor_creator;
    if (!output_getter)
        throw Exception("Format " + name + " is not suitable for output", ErrorCodes::FORMAT_IS_NOT_SUITABLE_FOR_OUTPUT);

    RowOutputFormatParams params;
    params.callback = std::move(callback);

    auto format_settings = _format_settings
        ? *_format_settings : getFormatSettings(context);

    /** TODO: Materialization is needed, because formats can use the functions `IDataType`,
      *  which only work with full columns.
      */
    auto format = output_getter(buf, sample, params, format_settings);

    /// Enable auto-flush for streaming mode. Currently it is needed by INSERT WATCH query.
    if (format_settings.enable_streaming)
        format->setAutoFlush();

    /// It's a kludge. Because I cannot remove context from MySQL format.
    if (auto * mysql = typeid_cast<MySQLOutputFormat *>(format.get()))
        mysql->setContext(context);

    return format;
}


void FormatFactory::registerInputFormat(const String & name, InputCreator input_creator)
{
    auto & target = dict[name].input_creator;
    if (target)
        throw Exception("FormatFactory: Input format " + name + " is already registered", ErrorCodes::LOGICAL_ERROR);
    target = std::move(input_creator);
}

void FormatFactory::registerOutputFormat(const String & name, OutputCreator output_creator)
{
    auto & target = dict[name].output_creator;
    if (target)
        throw Exception("FormatFactory: Output format " + name + " is already registered", ErrorCodes::LOGICAL_ERROR);
    target = std::move(output_creator);
}

void FormatFactory::registerInputFormatProcessor(const String & name, InputProcessorCreator input_creator)
{
    auto & target = dict[name].input_processor_creator;
    if (target)
        throw Exception("FormatFactory: Input format " + name + " is already registered", ErrorCodes::LOGICAL_ERROR);
    target = std::move(input_creator);
}

void FormatFactory::registerOutputFormatProcessor(const String & name, OutputProcessorCreator output_creator)
{
    auto & target = dict[name].output_processor_creator;
    if (target)
        throw Exception("FormatFactory: Output format " + name + " is already registered", ErrorCodes::LOGICAL_ERROR);
    target = std::move(output_creator);
}

void FormatFactory::registerFileSegmentationEngine(const String & name, FileSegmentationEngine file_segmentation_engine)
{
    auto & target = dict[name].file_segmentation_engine;
    if (target)
        throw Exception("FormatFactory: File segmentation engine " + name + " is already registered", ErrorCodes::LOGICAL_ERROR);
    target = std::move(file_segmentation_engine);
}

FormatFactory & FormatFactory::instance()
{
    static FormatFactory ret;
    return ret;
}

}
