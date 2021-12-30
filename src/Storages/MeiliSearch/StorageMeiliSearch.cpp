#include <Storages/StorageFactory.h>
#include <Storages/MeiliSearch/StorageMeiliSearch.h>
#include <Storages/IStorage.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <Common/parseAddress.h>
#include "Core/Types.h"
#include "Parsers/ASTFunction.h"
#include "Parsers/IAST_fwd.h"
#include <Parsers/ASTSelectQuery.h>
#include <base/logger_useful.h>
#include <Storages/transformQueryForExternalDatabase.h>
#include <QueryPipeline/Pipe.h>
#include <Storages/SelectQueryInfo.h>
#include <Storages/MeiliSearch/MeiliSearchConnection.h>
#include <Storages/MeiliSearch/SourceMeiliSearch.h>
#include <Storages/MeiliSearch/SinkMeiliSearch.h>
#include <iostream>

namespace DB 
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_QUERY_PARAMETER;
}

StorageMeiliSearch::StorageMeiliSearch(
        const StorageID & table_id,
        const MeiliSearchConfiguration& config_,
        const ColumnsDescription &  columns_,
        const ConstraintsDescription &  constraints_,
        const String &  comment) 
        : IStorage(table_id)
        , config{config_}
        , log(&Poco::Logger::get("StorageMeiliSearch (" + table_id.table_name + ")")) {
            StorageInMemoryMetadata storage_metadata;
            storage_metadata.setColumns(columns_);
            storage_metadata.setConstraints(constraints_);
            storage_metadata.setComment(comment);
            setInMemoryMetadata(storage_metadata);
        }

void printAST(ASTPtr ptr) {
    WriteBufferFromOwnString out;
    IAST::FormatSettings settings(out, true);
    settings.identifier_quoting_style = IdentifierQuotingStyle::BackticksMySQL;
    settings.always_quote_identifiers = IdentifierQuotingStyle::BackticksMySQL != IdentifierQuotingStyle::None;
    ptr->format(settings);
}

std::string convertASTtoStr(ASTPtr ptr) {
    WriteBufferFromOwnString out;
    IAST::FormatSettings settings(out, true);
    settings.identifier_quoting_style = IdentifierQuotingStyle::BackticksMySQL;
    settings.always_quote_identifiers = IdentifierQuotingStyle::BackticksMySQL != IdentifierQuotingStyle::None;
    ptr->format(settings);
    return out.str();
}

ASTPtr getFunctionParams(ASTPtr node, const std::string& name) {
    if (!node) {
        return nullptr;
    }
    const auto* ptr = node->as<ASTFunction>();
    if (ptr && ptr->name == name) {
        if (node->children.size() == 1) {
            return node->children[0];
        } else {
            return nullptr;
        }
    }
    for (const auto& next : node->children) {
        auto res = getFunctionParams(next, name);
        if (res != nullptr) {
            return res;
        }
    }
    return nullptr;
}

Pipe StorageMeiliSearch::read(
    const Names & column_names,
    const StorageMetadataPtr & metadata_snapshot,
    SelectQueryInfo & query_info,
    ContextPtr /*context*/,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    unsigned)
{
    metadata_snapshot->check(column_names, getVirtuals(), getStorageID());

    ASTPtr original_where = query_info.query->clone()->as<ASTSelectQuery &>().where();
    ASTPtr query_params = getFunctionParams(original_where, "meiliMatch");
    

    std::unordered_map<String, String> kv_pairs_params;
    if (query_params) {
        LOG_TRACE(log, "Query params: {}", convertASTtoStr(query_params));
        for (const auto& el : query_params->children) {
            auto str = el->getColumnName();
            auto it = find(str.begin(), str.end(), '=');
            if (it == str.end()) {
                throw Exception(
                    "meiliMatch function must have parameters of the form \'key=value\'",
                    ErrorCodes::BAD_QUERY_PARAMETER);
            }
            String key(str.begin() + 1, it);
            String value(it + 1, str.end() - 1);
            kv_pairs_params[key] = value;
        }
    } else {
        LOG_TRACE(log, "Query params: none");
    }

    for (const auto& el : kv_pairs_params) {
        LOG_TRACE(log, "Parsed parameter: key = " + el.first + ", value = " + el.second);
    }

    Block sample_block;
    for (const String & column_name : column_names)
    {
        auto column_data = metadata_snapshot->getColumns().getPhysical(column_name);
        sample_block.insert({ column_data.type, column_data.name });
    }

    return Pipe(std::make_shared<MeiliSearchSource>(config, sample_block, max_block_size, 0, kv_pairs_params));
}

SinkToStoragePtr StorageMeiliSearch::write(
    const ASTPtr & /*query*/, 
    const StorageMetadataPtr& metadata_snapshot, 
    ContextPtr /*local_context*/)
{
    LOG_TRACE(log, "Trying update index: " + config.index);
    return std::make_shared<SinkMeiliSearch>(
        config,
        metadata_snapshot->getSampleBlock(),
        20000);
}

MeiliSearchConfiguration getConfiguration(ASTs engine_args) 
{

    if (engine_args.size() != 3) {
        throw Exception(
                "Storage MeiliSearch requires 3 parameters: MeiliSearch('url/host:port', index, key).",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
    }

    // 7700 - default port
    try {
        auto parsed_host_port = parseAddress(engine_args[0]->as<ASTLiteral &>().value.safeGet<String>(), 7700);

        String host = parsed_host_port.first;
        UInt16 port = parsed_host_port.second;
        String index = engine_args[1]->as<ASTLiteral &>().value.safeGet<String>();
        String key = engine_args[2]->as<ASTLiteral &>().value.safeGet<String>();
        
        return MeiliSearchConfiguration(host, port, index, key);
    } catch(...) {
        String url = engine_args[0]->as<ASTLiteral &>().value.safeGet<String>();
        String index = engine_args[1]->as<ASTLiteral &>().value.safeGet<String>();
        String key = engine_args[2]->as<ASTLiteral &>().value.safeGet<String>();
        
        return MeiliSearchConfiguration(url, index, key);
    }
}

void registerStorageMeiliSearch(StorageFactory & factory)
{
    factory.registerStorage("MeiliSearch", [](const StorageFactory::Arguments & args) 
    {

        auto config = getConfiguration(args.engine_args);

        return StorageMeiliSearch::create(
            args.table_id,
            config,
            args.columns,
            args.constraints,
            args.comment);
    },
    {
        .source_access_type = AccessType::MEILISEARCH,
    });
}


}

