#include <Storages/StorageExecutable.h>

#include <filesystem>

#include <Common/ShellCommand.h>
#include <Common/filesystemHelpers.h>

#include <Core/Block.h>

#include <IO/ReadHelpers.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTCreateQuery.h>

#include <QueryPipeline/Pipe.h>
#include <Processors/ISimpleTransform.h>
#include <Processors/Executors/CompletedPipelineExecutor.h>
#include <Processors/Formats/IOutputFormat.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterSelectWithUnionQuery.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Storages/StorageFactory.h>

#include <boost/algorithm/string/split.hpp>


namespace DB
{

namespace ErrorCodes
{
    extern const int UNSUPPORTED_METHOD;
    extern const int LOGICAL_ERROR;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int TIMEOUT_EXCEEDED;
}

StorageExecutable::StorageExecutable(
    const StorageID & table_id_,
    const String & format,
    const ExecutableSettings & settings_,
    const std::vector<ASTPtr> & input_queries_,
    const ColumnsDescription & columns,
    const ConstraintsDescription & constraints)
    : IStorage(table_id_)
    , settings(settings_)
    , input_queries(input_queries_)
    , log(settings.is_executable_pool ? &Poco::Logger::get("StorageExecutablePool") : &Poco::Logger::get("StorageExecutable"))
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns);
    storage_metadata.setConstraints(constraints);
    setInMemoryMetadata(storage_metadata);

    ShellCommandCoordinator::Configuration configuration
    {
        .format = format,

        .pool_size = settings.pool_size,
        .command_termination_timeout = settings.command_termination_timeout,
        .max_command_execution_time = settings.max_command_execution_time,

        .is_executable_pool = settings.is_executable_pool,
        .send_chunk_header = settings.send_chunk_header,
        .execute_direct = true
    };

    coordinator = std::make_unique<ShellCommandCoordinator>(std::move(configuration));
}

Pipe StorageExecutable::read(
    const Names & /*column_names*/,
    const StorageMetadataPtr & metadata_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr context,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    unsigned /*threads*/)
{
    auto & script_name = settings.script_name;

    auto user_scripts_path = context->getUserScriptsPath();
    auto script_path = user_scripts_path + '/' + script_name;

    if (!pathStartsWith(script_path, user_scripts_path))
        throw Exception(ErrorCodes::UNSUPPORTED_METHOD,
            "Executable file {} must be inside user scripts folder {}",
            script_name,
            user_scripts_path);

    if (!std::filesystem::exists(std::filesystem::path(script_path)))
         throw Exception(ErrorCodes::UNSUPPORTED_METHOD,
            "Executable file {} does not exist inside user scripts folder {}",
            script_name,
            user_scripts_path);

    Pipes inputs;
    inputs.reserve(input_queries.size());

    for (auto & input_query : input_queries)
    {
        InterpreterSelectWithUnionQuery interpreter(input_query, context, {});
        inputs.emplace_back(QueryPipelineBuilder::getPipe(interpreter.buildQueryPipeline()));
    }

    auto sample_block = metadata_snapshot->getSampleBlock();

    ShellCommandSourceConfiguration configuration;
    configuration.max_block_size = max_block_size;

    if (coordinator->getConfiguration().is_executable_pool)
    {
        configuration.read_fixed_number_of_rows = true;
        configuration.read_number_of_rows_from_process_output = true;
    }

    /// TODO: Filter by column_names
    return coordinator->createPipe(settings.script_name, settings.script_arguments, std::move(inputs), std::move(sample_block), context, configuration);
}

void registerStorageExecutable(StorageFactory & factory)
{
    auto register_storage = [](const StorageFactory::Arguments & args, bool is_executable_pool) -> StoragePtr
    {
        auto local_context = args.getLocalContext();

        if (args.engine_args.size() < 2)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "StorageExecutable requires minimum 2 arguments: script_name, format, [input_query...]");

        for (size_t i = 0; i < 2; ++i)
            args.engine_args[i] = evaluateConstantExpressionOrIdentifierAsLiteral(args.engine_args[i], local_context);

        auto scipt_name_with_arguments_value = args.engine_args[0]->as<ASTLiteral &>().value.safeGet<String>();

        std::vector<String> script_name_with_arguments;
        boost::split(script_name_with_arguments, scipt_name_with_arguments_value, [](char c) { return c == ' '; });

        auto script_name = script_name_with_arguments[0];
        script_name_with_arguments.erase(script_name_with_arguments.begin());
        auto format = args.engine_args[1]->as<ASTLiteral &>().value.safeGet<String>();

        std::vector<ASTPtr> input_queries;
        for (size_t i = 2; i < args.engine_args.size(); ++i)
        {
            ASTPtr query = args.engine_args[i]->children.at(0);
            if (!query->as<ASTSelectWithUnionQuery>())
                throw Exception(
                    ErrorCodes::UNSUPPORTED_METHOD, "StorageExecutable argument is invalid input query {}",
                    query->formatForErrorMessage());

            input_queries.emplace_back(std::move(query));
        }

        const auto & columns = args.columns;
        const auto & constraints = args.constraints;

        ExecutableSettings settings;
        settings.script_name = script_name;
        settings.script_arguments = script_name_with_arguments;
        settings.is_executable_pool = is_executable_pool;

        if (is_executable_pool)
        {
            size_t max_command_execution_time = 10;

            size_t max_execution_time_seconds = static_cast<size_t>(args.getContext()->getSettings().max_execution_time.totalSeconds());
            if (max_execution_time_seconds != 0 && max_command_execution_time > max_execution_time_seconds)
                max_command_execution_time = max_execution_time_seconds;

            settings.max_command_execution_time = max_command_execution_time;
            if (args.storage_def->settings)
                settings.loadFromQuery(*args.storage_def);
        }

        auto global_context = args.getContext()->getGlobalContext();
        return StorageExecutable::create(args.table_id, format, settings, input_queries, columns, constraints);
    };

    factory.registerStorage("Executable", [&](const StorageFactory::Arguments & args)
    {
        return register_storage(args, false /*is_executable_pool*/);
    });

    factory.registerStorage("ExecutablePool", [&](const StorageFactory::Arguments & args)
    {
        return register_storage(args, true /*is_executable_pool*/);
    });
}

};

