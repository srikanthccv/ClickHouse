#include "config.h"

#if USE_HDFS

#include <Storages/HDFS/StorageHDFSCluster.h>

#include <Client/Connection.h>
#include <Core/QueryProcessingStage.h>
#include <DataTypes/DataTypeString.h>
#include <Interpreters/Context.h>
#include <Interpreters/getHeaderForProcessingStage.h>
#include <Interpreters/SelectQueryOptions.h>
#include <Interpreters/InterpreterSelectQuery.h>
#include <Interpreters/getTableExpressions.h>
#include <QueryPipeline/narrowPipe.h>
#include <QueryPipeline/Pipe.h>
#include <QueryPipeline/RemoteQueryExecutor.h>

#include <Processors/Transforms/AddingDefaultsTransform.h>

#include <Processors/Sources/RemoteSource.h>
#include <Parsers/queryToString.h>
#include <Parsers/ASTTablesInSelectQuery.h>

#include <Storages/IStorage.h>
#include <Storages/SelectQueryInfo.h>
#include <Storages/HDFS/HDFSCommon.h>
#include <Storages/StorageDictionary.h>

#include <memory>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

static ASTPtr addColumnsStructureToQuery(const ASTPtr & query, const String & structure)
{
    /// Add argument with table structure to hdfsCluster table function in select query.
    auto result_query = query->clone();
    ASTExpressionList * expression_list = extractTableFunctionArgumentsFromSelectQuery(result_query);
    if (!expression_list)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected SELECT query from table function hdfsCluster, got '{}'", queryToString(query));

    auto structure_literal = std::make_shared<ASTLiteral>(structure);

    if (expression_list->children.size() != 2 && expression_list->children.size() != 3)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected 2 or 3 arguments in hdfsCluster table functions, got {}", expression_list->children.size());

    if (expression_list->children.size() == 2)
    {
        auto format_literal = std::make_shared<ASTLiteral>("auto");
        expression_list->children.push_back(format_literal);
    }

    expression_list->children.push_back(structure_literal);
    return result_query;
}

StorageHDFSCluster::StorageHDFSCluster(
    ContextPtr context_,
    String cluster_name_,
    const String & uri_,
    const StorageID & table_id_,
    const String & format_name_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    const String & compression_method_)
    : IStorage(table_id_)
    , cluster_name(cluster_name_)
    , uri(uri_)
    , format_name(format_name_)
    , compression_method(compression_method_)
{
    context_->getRemoteHostFilter().checkURL(Poco::URI(uri_));
    checkHDFSURL(uri_);

    StorageInMemoryMetadata storage_metadata;

    if (columns_.empty())
    {
        auto columns = StorageHDFS::getTableStructureFromData(format_name, uri_, compression_method, context_);
        storage_metadata.setColumns(columns);
        need_to_add_structure_to_query = true;
    }
    else
        storage_metadata.setColumns(columns_);

    storage_metadata.setConstraints(constraints_);
    setInMemoryMetadata(storage_metadata);
}

/// The code executes on initiator
Pipe StorageHDFSCluster::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & query_info,
    ContextPtr context,
    QueryProcessingStage::Enum processed_stage,
    size_t /*max_block_size*/,
    unsigned /*num_streams*/)
{
    auto cluster = context->getCluster(cluster_name)->getClusterWithReplicasAsShards(context->getSettingsRef());

    auto iterator = std::make_shared<HDFSSource::DisclosedGlobIterator>(context, uri);
    auto callback = std::make_shared<HDFSSource::IteratorWrapper>([iterator]() mutable -> String
    {
        return iterator->next();
    });

    /// Calculate the header. This is significant, because some columns could be thrown away in some cases like query with count(*)
    Block header =
        InterpreterSelectQuery(query_info.query, context, SelectQueryOptions(processed_stage).analyze()).getSampleBlock();

    const Scalars & scalars = context->hasQueryContext() ? context->getQueryContext()->getScalars() : Scalars{};

    Pipes pipes;

    const bool add_agg_info = processed_stage == QueryProcessingStage::WithMergeableState;

    auto query_to_send = query_info.original_query;
    if (need_to_add_structure_to_query)
        query_to_send = addColumnsStructureToQuery(
            query_to_send, StorageDictionary::generateNamesAndTypesDescription(storage_snapshot->metadata->getColumns().getAll()));

    for (const auto & replicas : cluster->getShardsAddresses())
    {
        /// There will be only one replica, because we consider each replica as a shard
        for (const auto & node : replicas)
        {
            auto connection = std::make_shared<Connection>(
                node.host_name, node.port, context->getGlobalContext()->getCurrentDatabase(),
                node.user, node.password, node.quota_key, node.cluster, node.cluster_secret,
                "HDFSClusterInititiator",
                node.compression,
                node.secure
            );


            /// For unknown reason global context is passed to IStorage::read() method
            /// So, task_identifier is passed as constructor argument. It is more obvious.
            auto remote_query_executor = std::make_shared<RemoteQueryExecutor>(
                connection,
                queryToString(query_to_send),
                header,
                context,
                /*throttler=*/nullptr,
                scalars,
                Tables(),
                processed_stage,
                RemoteQueryExecutor::Extension{.task_iterator = callback});

            pipes.emplace_back(std::make_shared<RemoteSource>(remote_query_executor, add_agg_info, false));
        }
    }

    storage_snapshot->check(column_names);
    return Pipe::unitePipes(std::move(pipes));
}

QueryProcessingStage::Enum StorageHDFSCluster::getQueryProcessingStage(
    ContextPtr context, QueryProcessingStage::Enum to_stage, const StorageSnapshotPtr &, SelectQueryInfo &) const
{
    /// Initiator executes query on remote node.
    if (context->getClientInfo().query_kind == ClientInfo::QueryKind::INITIAL_QUERY)
        if (to_stage >= QueryProcessingStage::Enum::WithMergeableState)
            return QueryProcessingStage::Enum::WithMergeableState;

    /// Follower just reads the data.
    return QueryProcessingStage::Enum::FetchColumns;
}


NamesAndTypesList StorageHDFSCluster::getVirtuals() const
{
    return NamesAndTypesList{
        {"_path", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())},
        {"_file", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())}};
}


}

#endif
