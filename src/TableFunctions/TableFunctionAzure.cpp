#include "config.h"

#if USE_AZURE_BLOB_STORAGE

//#include <IO/S3Common.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Interpreters/Context.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/TableFunctionS3.h>
#include <TableFunctions/TableFunctionAzure.h>
#include <Interpreters/parseColumnsListForTableFunction.h>
#include <Access/Common/AccessFlags.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTFunction.h>
#include <Storages/checkAndGetLiteralArgument.h>
#include <Storages/StorageAzure.h>
#include <Storages/StorageS3.h>
#include <Storages/StorageURL.h>
#include <Storages/NamedCollectionsHelpers.h>
#include <Formats/FormatFactory.h>
#include "registerTableFunctions.h"
#include <Disks/ObjectStorages/AzureBlobStorage/AzureObjectStorage.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>

#include <boost/algorithm/string.hpp>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int LOGICAL_ERROR;
}

namespace
{

bool isConnectionString(const std::string & candidate)
{
    return candidate.starts_with("DefaultEndpointsProtocol");
}

}

StorageAzure::Configuration TableFunctionAzure::parseArgumentsImpl(ASTs & engine_args, const ContextPtr & local_context, bool get_format_from_file)
{
    StorageAzure::Configuration configuration;

    /// Supported signatures:
    ///
    /// Azure(connection_string|storage_account_url, container_name, blobpath, [account_name, account_key, format, compression, structure])
    ///

    if (auto named_collection = tryGetNamedCollectionWithOverrides(engine_args, local_context))
    {
        StorageAzure::processNamedCollectionResult(configuration, *named_collection);

        configuration.blobs_paths = {configuration.blob_path};

        if (configuration.format == "auto" && get_format_from_file)
            configuration.format = FormatFactory::instance().getFormatFromFileName(configuration.blob_path, true);

        return configuration;
    }

    if (engine_args.size() < 3 || engine_args.size() > 8)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                        "Storage Azure requires 3 to 7 arguments: "
                        "Azure(connection_string|storage_account_url, container_name, blobpath, [account_name, account_key, format, compression, structure])");

    for (auto & engine_arg : engine_args)
        engine_arg = evaluateConstantExpressionOrIdentifierAsLiteral(engine_arg, local_context);

    std::unordered_map<std::string_view, size_t> engine_args_to_idx;

    configuration.connection_url = checkAndGetLiteralArgument<String>(engine_args[0], "connection_string/storage_account_url");
    configuration.is_connection_string = isConnectionString(configuration.connection_url);

    configuration.container = checkAndGetLiteralArgument<String>(engine_args[1], "container");
    configuration.blob_path = checkAndGetLiteralArgument<String>(engine_args[2], "blobpath");

    auto is_format_arg = [] (const std::string & s) -> bool
    {
        return s == "auto" || FormatFactory::instance().getAllFormats().contains(s);
    };

    if (engine_args.size() == 4)
    {
        auto fourth_arg = checkAndGetLiteralArgument<String>(engine_args[3], "format/account_name/structure");
        if (is_format_arg(fourth_arg))
        {
            configuration.format = fourth_arg;
        }
        else
        {
            configuration.structure = fourth_arg;
        }
    }
    else if (engine_args.size() == 5)
    {
        auto fourth_arg = checkAndGetLiteralArgument<String>(engine_args[3], "format/account_name");
        if (is_format_arg(fourth_arg))
        {
            configuration.format = fourth_arg;
            configuration.compression_method = checkAndGetLiteralArgument<String>(engine_args[4], "compression");
        }
        else
        {
            configuration.account_name = fourth_arg;
            configuration.account_key = checkAndGetLiteralArgument<String>(engine_args[4], "account_key");
        }
    }
    else if (engine_args.size() == 6)
    {
        auto fourth_arg = checkAndGetLiteralArgument<String>(engine_args[3], "format/account_name");
        if (is_format_arg(fourth_arg))
        {
            configuration.format = fourth_arg;
            configuration.compression_method = checkAndGetLiteralArgument<String>(engine_args[4], "compression");
            configuration.structure = checkAndGetLiteralArgument<String>(engine_args[5], "structure");
        }
        else
        {
            configuration.account_name = fourth_arg;
            configuration.account_key = checkAndGetLiteralArgument<String>(engine_args[4], "account_key");
            auto sixth_arg = checkAndGetLiteralArgument<String>(engine_args[5], "format/account_name");
            if (!is_format_arg(sixth_arg))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unknown format {}", sixth_arg);
            configuration.format = sixth_arg;
        }
    }
    else if (engine_args.size() == 7)
    {
        auto fourth_arg = checkAndGetLiteralArgument<String>(engine_args[3], "format/account_name");
        if (is_format_arg(fourth_arg))
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Format, compression and structure must be last arguments");
        }
        else
        {
            configuration.account_name = fourth_arg;
            configuration.account_key = checkAndGetLiteralArgument<String>(engine_args[4], "account_key");
            auto sixth_arg = checkAndGetLiteralArgument<String>(engine_args[5], "format/account_name");
            if (!is_format_arg(sixth_arg))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unknown format {}", sixth_arg);
            configuration.format = sixth_arg;
            configuration.compression_method = checkAndGetLiteralArgument<String>(engine_args[6], "compression");
        }
    }
    else if (engine_args.size() == 8)
    {

        auto fourth_arg = checkAndGetLiteralArgument<String>(engine_args[3], "format/account_name");
        if (is_format_arg(fourth_arg))
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Format and compression must be last arguments");
        }
        else
        {
            configuration.account_name = fourth_arg;
            configuration.account_key = checkAndGetLiteralArgument<String>(engine_args[4], "account_key");
            auto sixth_arg = checkAndGetLiteralArgument<String>(engine_args[5], "format/account_name");
            if (!is_format_arg(sixth_arg))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unknown format {}", sixth_arg);
            configuration.format = sixth_arg;
            configuration.compression_method = checkAndGetLiteralArgument<String>(engine_args[6], "compression");
            configuration.structure = checkAndGetLiteralArgument<String>(engine_args[7], "structure");
        }
    }

    configuration.blobs_paths = {configuration.blob_path};

    if (configuration.format == "auto" && get_format_from_file)
        configuration.format = FormatFactory::instance().getFormatFromFileName(configuration.blob_path, true);

    return configuration;
}

void TableFunctionAzure::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    /// Clone ast function, because we can modify its arguments like removing headers.
    auto ast_copy = ast_function->clone();

    ASTs & args_func = ast_function->children;

    if (args_func.size() != 1)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Table function '{}' must have arguments.", getName());

    auto & args = args_func.at(0)->children;

    parseArgumentsImpl(args, context);
}

ColumnsDescription TableFunctionAzure::getActualTableStructure(ContextPtr context) const
{
    if (configuration.structure == "auto")
    {
        context->checkAccess(getSourceAccessType());
        auto client = StorageAzure::createClient(configuration);
        auto settings = StorageAzure::createSettings(context);

        auto object_storage = std::make_unique<AzureObjectStorage>("AzureTableFunction", std::move(client), std::move(settings));
        return StorageAzure::getTableStructureFromData(object_storage.get(), configuration, std::nullopt, context);
    }

    return parseColumnsListFromString(configuration.structure, context);
}

bool TableFunctionAzure::supportsReadingSubsetOfColumns()
{
    return FormatFactory::instance().checkIfFormatSupportsSubsetOfColumns(configuration.format);
}

StoragePtr TableFunctionAzure::executeImpl(const ASTPtr & /*ast_function*/, ContextPtr context, const std::string & table_name, ColumnsDescription /*cached_columns*/) const
{
    configuration.is_connection_string = true;
    configuration.blobs_paths = {configuration.blob_path};

    auto client = StorageAzure::createClient(configuration);
    auto settings = StorageAzure::createSettings(context);

    ColumnsDescription columns;
    if (configuration.structure != "auto")
        columns = parseColumnsListFromString(configuration.structure, context);
    else if (!structure_hint.empty())
        columns = structure_hint;

    StoragePtr storage = std::make_shared<StorageAzure>(
        configuration,
        std::make_unique<AzureObjectStorage>(table_name, std::move(client), std::move(settings)),
        context,
        StorageID(getDatabaseName(), table_name),
        columns,
        ConstraintsDescription{},
        String{},
        /// No format_settings for table function Azure
        std::nullopt,
        nullptr);

    storage->startup();

    return storage;
}

void registerTableFunctionAzure(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionAzure>(
        {.documentation
         = {.description=R"(The table function can be used to read the data stored on Azure Blob Storage.)",
            .examples{{"azure_blob_storage", "SELECT * FROM  azure_blob_storage(connection, container, blob_path, format, structure)", ""}}},
         .allow_readonly = false});
}

}

#endif
