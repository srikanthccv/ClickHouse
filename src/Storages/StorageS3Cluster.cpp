#include "Storages/StorageS3Cluster.h"

#include "config.h"

#if USE_AWS_S3

#include "Common/Exception.h"
#include "Client/Connection.h"
#include "Core/QueryProcessingStage.h"
#include <DataTypes/DataTypeString.h>
#include <IO/WriteBufferFromS3.h>
#include <IO/WriteHelpers.h>
#include <Interpreters/Context.h>
#include <Interpreters/getHeaderForProcessingStage.h>
#include <Interpreters/SelectQueryOptions.h>
#include <Interpreters/InterpreterSelectQuery.h>
#include <Processors/Transforms/AddingDefaultsTransform.h>
#include <QueryPipeline/narrowPipe.h>
#include <QueryPipeline/Pipe.h>
#include <Processors/Sources/RemoteSource.h>
#include <QueryPipeline/RemoteQueryExecutor.h>
#include <Parsers/queryToString.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Parsers/ASTLiteral.h>
#include <Storages/IStorage.h>
#include <Storages/SelectQueryInfo.h>
#include <Storages/getVirtualsForStorage.h>
#include <Storages/StorageDictionary.h>
#include <Storages/addColumnsStructureToQueryWithClusterEngine.h>
#include <Common/logger_useful.h>

#include <aws/core/auth/AWSCredentials.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListObjectsV2Request.h>

#include <memory>
#include <string>

namespace DB
{

StorageS3Cluster::StorageS3Cluster(
    const StorageS3ClusterConfiguration & configuration_,
    const StorageID & table_id_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    ContextPtr context_)
    : IStorage(table_id_)
    , s3_configuration{configuration_.url, configuration_.auth_settings, configuration_.rw_settings, configuration_.headers}
    , filename(configuration_.url)
    , cluster_name(configuration_.cluster_name)
    , format_name(configuration_.format)
    , compression_method(configuration_.compression_method)
{
    context_->getGlobalContext()->getRemoteHostFilter().checkURL(Poco::URI{filename});
    StorageInMemoryMetadata storage_metadata;
    StorageS3::updateS3Configuration(context_, s3_configuration);

    if (columns_.empty())
    {
        const bool is_key_with_globs = filename.find_first_of("*?{") != std::string::npos;

        /// `distributed_processing` is set to false, because this code is executed on the initiator, so there is no callback set
        /// for asking for the next tasks.
        /// `format_settings` is set to std::nullopt, because StorageS3Cluster is used only as table function
        auto columns = StorageS3::getTableStructureFromDataImpl(format_name, s3_configuration, compression_method,
            /*distributed_processing_*/false, is_key_with_globs, /*format_settings=*/std::nullopt, context_);
        storage_metadata.setColumns(columns);
        add_columns_structure_to_query = true;
    }
    else
        storage_metadata.setColumns(columns_);

    storage_metadata.setConstraints(constraints_);
    setInMemoryMetadata(storage_metadata);

    auto default_virtuals = NamesAndTypesList{
        {"_path", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())},
        {"_file", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())}};

    auto columns = storage_metadata.getSampleBlock().getNamesAndTypesList();
    virtual_columns = getVirtualsForStorage(columns, default_virtuals);
    for (const auto & column : virtual_columns)
        virtual_block.insert({column.type->createColumn(), column.type, column.name});
}

/// The code executes on initiator
Pipe StorageS3Cluster::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & query_info,
    ContextPtr context,
    QueryProcessingStage::Enum processed_stage,
    size_t /*max_block_size*/,
    unsigned /*num_streams*/)
{
    StorageS3::updateS3Configuration(context, s3_configuration);

    auto cluster = context->getCluster(cluster_name)->getClusterWithReplicasAsShards(context->getSettingsRef());

    auto iterator = std::make_shared<StorageS3Source::DisclosedGlobIterator>(
        *s3_configuration.client, s3_configuration.uri, query_info.query, virtual_block, context);
    auto callback = std::make_shared<StorageS3Source::IteratorWrapper>([iterator]() mutable -> String { return iterator->next(); });

    /// Calculate the header. This is significant, because some columns could be thrown away in some cases like query with count(*)
    Block header =
        InterpreterSelectQuery(query_info.query, context, SelectQueryOptions(processed_stage).analyze()).getSampleBlock();

    const Scalars & scalars = context->hasQueryContext() ? context->getQueryContext()->getScalars() : Scalars{};

    Pipes pipes;

    const bool add_agg_info = processed_stage == QueryProcessingStage::WithMergeableState;

    ASTPtr query_to_send = query_info.original_query->clone();
    if (add_columns_structure_to_query)
        addColumnsStructureToQueryWithClusterEngine(
            query_to_send, StorageDictionary::generateNamesAndTypesDescription(storage_snapshot->metadata->getColumns().getAll()), 5, getName());

    for (const auto & replicas : cluster->getShardsAddresses())
    {
        /// There will be only one replica, because we consider each replica as a shard
        for (const auto & node : replicas)
        {
            auto connection = std::make_shared<Connection>(
                node.host_name, node.port, context->getGlobalContext()->getCurrentDatabase(),
                node.user, node.password, node.quota_key, node.cluster, node.cluster_secret,
                "S3ClusterInititiator",
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

QueryProcessingStage::Enum StorageS3Cluster::getQueryProcessingStage(
    ContextPtr context, QueryProcessingStage::Enum to_stage, const StorageSnapshotPtr &, SelectQueryInfo &) const
{
    /// Initiator executes query on remote node.
    if (context->getClientInfo().query_kind == ClientInfo::QueryKind::INITIAL_QUERY)
        if (to_stage >= QueryProcessingStage::Enum::WithMergeableState)
            return QueryProcessingStage::Enum::WithMergeableState;

    /// Follower just reads the data.
    return QueryProcessingStage::Enum::FetchColumns;
}


NamesAndTypesList StorageS3Cluster::getVirtuals() const
{
    return virtual_columns;
}


}

#endif
