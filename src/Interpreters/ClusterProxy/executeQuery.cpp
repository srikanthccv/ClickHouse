#include <Core/QueryProcessingStage.h>
#include <Core/Settings.h>
#include <Core/UUID.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/ObjectUtils.h>
#include <Interpreters/Cluster.h>
#include <Interpreters/ClusterProxy/SelectStreamFactory.h>
#include <Interpreters/ClusterProxy/executeQuery.h>
#include <Interpreters/Context.h>
#include <Interpreters/IInterpreter.h>
#include <Interpreters/InterpreterSelectQueryAnalyzer.h>
#include <Interpreters/OptimizeShardingKeyRewriteInVisitor.h>
#include <Interpreters/ProcessList.h>
#include <Interpreters/getCustomKeyFilterForParallelReplicas.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/queryToString.h>
#include <Planner/Utils.h>
#include <Processors/QueryPlan/DistributedCreateLocalPlan.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/ReadFromPreparedSource.h>
#include <Processors/QueryPlan/ReadFromRemote.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Processors/ResizeProcessor.h>
#include <Processors/Sources/NullSource.h>
#include <QueryPipeline/Pipe.h>
#include <Storages/Distributed/DistributedSettings.h>
#include <Storages/SelectQueryInfo.h>
#include <Storages/StorageReplicatedMergeTree.h>
#include <Storages/StorageSnapshot.h>
#include <Storages/buildQueryTreeForShard.h>
#include <Storages/getStructureOfRemoteTable.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TOO_LARGE_DISTRIBUTED_DEPTH;
    extern const int LOGICAL_ERROR;
    extern const int UNEXPECTED_CLUSTER;
}

namespace ClusterProxy
{

ContextMutablePtr updateSettingsAndClientInfoForCluster(const Cluster & cluster,
    bool is_remote_function,
    ContextPtr context,
    const Settings & settings,
    const StorageID & main_table,
    ASTPtr additional_filter_ast,
    LoggerPtr log,
    const DistributedSettings * distributed_settings)
{
    ClientInfo new_client_info = context->getClientInfo();
    Settings new_settings = settings;
    new_settings.queue_max_wait_ms = Cluster::saturate(new_settings.queue_max_wait_ms, settings.max_execution_time);

    /// In case of interserver mode we should reset initial_user for remote() function to use passed user from the query.
    if (is_remote_function)
    {
        const auto & address = cluster.getShardsAddresses().front().front();
        new_client_info.initial_user = address.user;
    }

    /// If "secret" (in remote_servers) is not in use,
    /// user on the shard is not the same as the user on the initiator,
    /// hence per-user limits should not be applied.
    const bool interserver_mode = !cluster.getSecret().empty();
    if (!interserver_mode)
    {
        /// Does not matter on remote servers, because queries are sent under different user.
        new_settings.max_concurrent_queries_for_user = 0;
        new_settings.max_memory_usage_for_user = 0;

        /// Set as unchanged to avoid sending to remote server.
        new_settings.max_concurrent_queries_for_user.changed = false;
        new_settings.max_memory_usage_for_user.changed = false;
    }

    if (settings.force_optimize_skip_unused_shards_nesting && settings.force_optimize_skip_unused_shards)
    {
        if (new_settings.force_optimize_skip_unused_shards_nesting == 1)
        {
            new_settings.force_optimize_skip_unused_shards = false;
            new_settings.force_optimize_skip_unused_shards.changed = false;

            if (log)
                LOG_TRACE(log, "Disabling force_optimize_skip_unused_shards for nested queries (force_optimize_skip_unused_shards_nesting exceeded)");
        }
        else
        {
            --new_settings.force_optimize_skip_unused_shards_nesting.value;
            new_settings.force_optimize_skip_unused_shards_nesting.changed = true;

            if (log)
                LOG_TRACE(log, "force_optimize_skip_unused_shards_nesting is now {}", new_settings.force_optimize_skip_unused_shards_nesting);
        }
    }

    if (settings.optimize_skip_unused_shards_nesting && settings.optimize_skip_unused_shards)
    {
        if (new_settings.optimize_skip_unused_shards_nesting == 1)
        {
            new_settings.optimize_skip_unused_shards = false;
            new_settings.optimize_skip_unused_shards.changed = false;

            if (log)
                LOG_TRACE(log, "Disabling optimize_skip_unused_shards for nested queries (optimize_skip_unused_shards_nesting exceeded)");
        }
        else
        {
            --new_settings.optimize_skip_unused_shards_nesting.value;
            new_settings.optimize_skip_unused_shards_nesting.changed = true;

            if (log)
                LOG_TRACE(log, "optimize_skip_unused_shards_nesting is now {}", new_settings.optimize_skip_unused_shards_nesting);
        }
    }

    if (!settings.skip_unavailable_shards.changed && distributed_settings)
    {
        new_settings.skip_unavailable_shards = distributed_settings->skip_unavailable_shards.value;
        new_settings.skip_unavailable_shards.changed = true;
    }

    if (settings.offset)
    {
        new_settings.offset = 0;
        new_settings.offset.changed = false;
    }
    if (settings.limit)
    {
        new_settings.limit = 0;
        new_settings.limit.changed = false;
    }

    /// Setting additional_table_filters may be applied to Distributed table.
    /// In case if query is executed up to WithMergableState on remote shard, it is impossible to filter on initiator.
    /// We need to propagate the setting, but change the table name from distributed to source.
    ///
    /// Here we don't try to analyze setting again. In case if query_info->additional_filter_ast is not empty, some filter was applied.
    /// It's just easier to add this filter for a source table.
    if (additional_filter_ast)
    {
        Tuple tuple;
        tuple.push_back(main_table.getShortName());
        tuple.push_back(queryToString(additional_filter_ast));
        new_settings.additional_table_filters.value.push_back(std::move(tuple));
    }

    /// disable parallel replicas if cluster contains only shards with 1 replica
    if (context->canUseTaskBasedParallelReplicas())
    {
        bool disable_parallel_replicas = true;
        for (const auto & shard : cluster.getShardsInfo())
        {
            if (shard.getAllNodeCount() > 1)
            {
                disable_parallel_replicas = false;
                break;
            }
        }
        if (disable_parallel_replicas)
            new_settings.allow_experimental_parallel_reading_from_replicas = 0;
    }

    if (settings.max_execution_time_leaf.value > 0)
    {
        /// Replace 'max_execution_time' of this sub-query with 'max_execution_time_leaf' and 'timeout_overflow_mode'
        /// with 'timeout_overflow_mode_leaf'
        new_settings.max_execution_time = settings.max_execution_time_leaf;
        new_settings.timeout_overflow_mode = settings.timeout_overflow_mode_leaf;
    }

    /// in case of parallel replicas custom key use round robing load balancing
    /// so custom key partitions will be spread over nodes in round-robin fashion
    if (context->canUseParallelReplicasCustomKeyForCluster(cluster) && !settings.load_balancing.changed)
    {
        new_settings.load_balancing = LoadBalancing::ROUND_ROBIN;
    }

    auto new_context = Context::createCopy(context);
    new_context->setSettings(new_settings);
    new_context->setClientInfo(new_client_info);

    if (context->canUseParallelReplicasCustomKeyForCluster(cluster))
        new_context->disableOffsetParallelReplicas();

    return new_context;
}

ContextMutablePtr updateSettingsForCluster(const Cluster & cluster, ContextPtr context, const Settings & settings, const StorageID & main_table)
{
    return updateSettingsAndClientInfoForCluster(cluster,
        /* is_remote_function= */ false,
        context,
        settings,
        main_table,
        /* additional_filter_ast= */ {},
        /* log= */ {},
        /* distributed_settings= */ {});
}


static ThrottlerPtr getThrottler(const ContextPtr & context)
{
    const Settings & settings = context->getSettingsRef();

    ThrottlerPtr user_level_throttler;
    if (auto process_list_element = context->getProcessListElement())
        user_level_throttler = process_list_element->getUserNetworkThrottler();

    /// Network bandwidth limit, if needed.
    ThrottlerPtr throttler;
    if (settings.max_network_bandwidth || settings.max_network_bytes)
    {
        throttler = std::make_shared<Throttler>(
                settings.max_network_bandwidth,
                settings.max_network_bytes,
                "Limit for bytes to send or receive over network exceeded.",
                user_level_throttler);
    }
    else
        throttler = user_level_throttler;

    return throttler;
}

AdditionalShardFilterGenerator
getShardFilterGeneratorForCustomKey(const Cluster & cluster, ContextPtr context, const ColumnsDescription & columns)
{
    if (!context->canUseParallelReplicasCustomKeyForCluster(cluster))
        return {};

    const auto & settings = context->getSettingsRef();
    auto custom_key_ast = parseCustomKeyForTable(settings.parallel_replicas_custom_key, *context);
    if (custom_key_ast == nullptr)
        return {};

    return [my_custom_key_ast = std::move(custom_key_ast),
            column_description = columns,
            custom_key_type = settings.parallel_replicas_custom_key_filter_type.value,
            custom_key_range_lower = settings.parallel_replicas_custom_key_range_lower.value,
            custom_key_range_upper = settings.parallel_replicas_custom_key_range_upper.value,
            query_context = context,
            replica_count = cluster.getShardsInfo().front().per_replica_pools.size()](uint64_t replica_num) -> ASTPtr
    {
        return getCustomKeyFilterForParallelReplica(
            replica_count,
            replica_num - 1,
            my_custom_key_ast,
            {custom_key_type, custom_key_range_lower, custom_key_range_upper},
            column_description,
            query_context);
    };
}


void executeQuery(
    QueryPlan & query_plan,
    const Block & header,
    QueryProcessingStage::Enum processed_stage,
    const StorageID & main_table,
    const ASTPtr & table_func_ptr,
    SelectStreamFactory & stream_factory,
    LoggerPtr log,
    ContextPtr context,
    const SelectQueryInfo & query_info,
    const ExpressionActionsPtr & sharding_key_expr,
    const std::string & sharding_key_column_name,
    const DistributedSettings & distributed_settings,
    AdditionalShardFilterGenerator shard_filter_generator,
    bool is_remote_function)
{
    const Settings & settings = context->getSettingsRef();

    if (settings.max_distributed_depth && context->getClientInfo().distributed_depth >= settings.max_distributed_depth)
        throw Exception(ErrorCodes::TOO_LARGE_DISTRIBUTED_DEPTH, "Maximum distributed depth exceeded");

    const ClusterPtr & not_optimized_cluster = query_info.cluster;

    std::vector<QueryPlanPtr> plans;
    SelectStreamFactory::Shards remote_shards;

    auto cluster = query_info.getCluster();
    auto new_context = updateSettingsAndClientInfoForCluster(*cluster, is_remote_function, context,
        settings, main_table, query_info.additional_filter_ast, log, &distributed_settings);
    if (context->getSettingsRef().allow_experimental_parallel_reading_from_replicas
        && context->getSettingsRef().allow_experimental_parallel_reading_from_replicas.value
           != new_context->getSettingsRef().allow_experimental_parallel_reading_from_replicas.value)
    {
        LOG_TRACE(
            log,
            "Parallel reading from replicas is disabled for cluster. There are no shards with more than 1 replica: cluster={}",
            cluster->getName());
    }

    new_context->increaseDistributedDepth();

    const size_t shards = cluster->getShardCount();

    if (context->getSettingsRef().allow_experimental_analyzer)
    {
        for (size_t i = 0, s = cluster->getShardsInfo().size(); i < s; ++i)
        {
            const auto & shard_info = cluster->getShardsInfo()[i];

            auto query_for_shard = query_info.query_tree->clone();
            if (sharding_key_expr &&
                query_info.optimized_cluster &&
                settings.optimize_skip_unused_shards_rewrite_in &&
                shards > 1 &&
                /// TODO: support composite sharding key
                sharding_key_expr->getRequiredColumns().size() == 1)
            {
                OptimizeShardingKeyRewriteInVisitor::Data visitor_data{
                    sharding_key_expr,
                    sharding_key_column_name,
                    shard_info,
                    not_optimized_cluster->getSlotToShard(),
                };
                optimizeShardingKeyRewriteIn(query_for_shard, std::move(visitor_data), new_context);
            }

            // decide for each shard if parallel reading from replicas should be enabled
            // according to settings and number of replicas declared per shard
            const auto & addresses = cluster->getShardsAddresses().at(i);
            bool parallel_replicas_enabled = addresses.size() > 1 && context->canUseTaskBasedParallelReplicas();

            stream_factory.createForShard(
                shard_info,
                query_for_shard,
                main_table,
                table_func_ptr,
                new_context,
                plans,
                remote_shards,
                static_cast<UInt32>(shards),
                parallel_replicas_enabled,
                shard_filter_generator);
        }
    }
    else
    {
        for (size_t i = 0, s = cluster->getShardsInfo().size(); i < s; ++i)
        {
            const auto & shard_info = cluster->getShardsInfo()[i];

            ASTPtr query_ast_for_shard = query_info.query->clone();
            if (sharding_key_expr &&
                query_info.optimized_cluster &&
                settings.optimize_skip_unused_shards_rewrite_in &&
                shards > 1 &&
                /// TODO: support composite sharding key
                sharding_key_expr->getRequiredColumns().size() == 1)
            {
                OptimizeShardingKeyRewriteInVisitor::Data visitor_data{
                    sharding_key_expr,
                    sharding_key_column_name,
                    shard_info,
                    not_optimized_cluster->getSlotToShard(),
                };
                OptimizeShardingKeyRewriteInVisitor visitor(visitor_data);
                visitor.visit(query_ast_for_shard);
            }

            // decide for each shard if parallel reading from replicas should be enabled
            // according to settings and number of replicas declared per shard
            const auto & addresses = cluster->getShardsAddresses().at(i);
            bool parallel_replicas_enabled = addresses.size() > 1 && context->canUseTaskBasedParallelReplicas();

            stream_factory.createForShard(
                shard_info,
                query_ast_for_shard,
                main_table,
                table_func_ptr,
                new_context,
                plans,
                remote_shards,
                static_cast<UInt32>(shards),
                parallel_replicas_enabled,
                shard_filter_generator);
        }
    }

    if (!remote_shards.empty())
    {
        Scalars scalars = context->hasQueryContext() ? context->getQueryContext()->getScalars() : Scalars{};
        scalars.emplace(
            "_shard_count", Block{{DataTypeUInt32().createColumnConst(1, shards), std::make_shared<DataTypeUInt32>(), "_shard_count"}});
        auto external_tables = context->getExternalTables();

        auto plan = std::make_unique<QueryPlan>();
        auto read_from_remote = std::make_unique<ReadFromRemote>(
            std::move(remote_shards),
            header,
            processed_stage,
            main_table,
            table_func_ptr,
            new_context,
            getThrottler(context),
            std::move(scalars),
            std::move(external_tables),
            log,
            shards,
            query_info.storage_limits,
            not_optimized_cluster->getName());

        read_from_remote->setStepDescription("Read from remote replica");
        plan->addStep(std::move(read_from_remote));
        plan->addInterpreterContext(new_context);
        plans.emplace_back(std::move(plan));
    }

    if (plans.empty())
        return;

    if (plans.size() == 1)
    {
        query_plan = std::move(*plans.front());
        return;
    }

    DataStreams input_streams;
    input_streams.reserve(plans.size());
    for (auto & plan : plans)
        input_streams.emplace_back(plan->getCurrentDataStream());

    auto union_step = std::make_unique<UnionStep>(std::move(input_streams));
    query_plan.unitePlans(std::move(union_step), std::move(plans));
}


void executeQueryWithParallelReplicas(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    const Block & header,
    QueryProcessingStage::Enum processed_stage,
    const ASTPtr & query_ast,
    ContextPtr context,
    std::shared_ptr<const StorageLimitsList> storage_limits)
{
    auto logger = getLogger("executeQueryWithParallelReplicas");
    LOG_DEBUG(logger, "Executing read from {}, header {}, query ({}), stage {} with parallel replicas",
        storage_id.getNameForLogs(), header.dumpStructure(), query_ast->formatForLogging(), processed_stage);

    const auto & settings = context->getSettingsRef();

    /// check cluster for parallel replicas
    auto not_optimized_cluster = context->getClusterForParallelReplicas();

    auto new_context = Context::createCopy(context);

    /// check hedged connections setting
    if (settings.use_hedged_requests.value)
    {
        if (settings.use_hedged_requests.changed)
        {
            LOG_WARNING(
                getLogger("executeQueryWithParallelReplicas"),
                "Setting 'use_hedged_requests' explicitly with enabled 'allow_experimental_parallel_reading_from_replicas' has no effect. "
                "Hedged connections are not used for parallel reading from replicas");
        }
        else
        {
            LOG_INFO(
                getLogger("executeQueryWithParallelReplicas"),
                "Disabling 'use_hedged_requests' in favor of 'allow_experimental_parallel_reading_from_replicas'. Hedged connections are "
                "not used for parallel reading from replicas");
        }

        /// disable hedged connections -> parallel replicas uses own logic to choose replicas
        new_context->setSetting("use_hedged_requests", Field{false});
    }

    auto scalars = new_context->hasQueryContext() ? new_context->getQueryContext()->getScalars() : Scalars{};

    UInt64 shard_num = 0; /// shard_num is 1-based, so 0 - no shard specified
    const auto it = scalars.find("_shard_num");
    if (it != scalars.end())
    {
        const Block & block = it->second;
        const auto & column = block.safeGetByPosition(0).column;
        shard_num = column->getUInt(0);
    }

    const auto shard_count = not_optimized_cluster->getShardCount();
    ClusterPtr new_cluster = not_optimized_cluster;
    /// if got valid shard_num from query initiator, then parallel replicas scope is the specified shard
    /// shards are numbered in order of appearance in the cluster config
    if (shard_num > 0)
    {
        if (shard_num > shard_count)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Shard number is greater than shard count: shard_num={} shard_count={} cluster={}",
                shard_num,
                shard_count,
                not_optimized_cluster->getName());

        chassert(shard_count == not_optimized_cluster->getShardsAddresses().size());

        LOG_DEBUG(getLogger("executeQueryWithParallelReplicas"), "Parallel replicas query in shard scope: shard_num={} cluster={}",
                  shard_num, not_optimized_cluster->getName());

        // get cluster for shard specified by shard_num
        // shard_num is 1-based, but getClusterWithSingleShard expects 0-based index
        new_cluster = not_optimized_cluster->getClusterWithSingleShard(shard_num - 1);
    }
    else
    {
        if (not_optimized_cluster->getShardCount() > 1)
            throw DB::Exception(
                ErrorCodes::UNEXPECTED_CLUSTER,
                "`cluster_for_parallel_replicas` setting refers to cluster with several shards. Expected a cluster with one shard");
    }

    auto external_tables = new_context->getExternalTables();
    auto read_from_remote = std::make_unique<ReadFromParallelRemoteReplicasStep>(
        query_ast,
        new_cluster,
        storage_id,
        header,
        processed_stage,
        new_context,
        getThrottler(new_context),
        std::move(scalars),
        std::move(external_tables),
        getLogger("ReadFromParallelRemoteReplicasStep"),
        std::move(storage_limits));

    query_plan.addStep(std::move(read_from_remote));
}

void executeQueryWithParallelReplicas(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    QueryProcessingStage::Enum processed_stage,
    const QueryTreeNodePtr & query_tree,
    const PlannerContextPtr & planner_context,
    ContextPtr context,
    std::shared_ptr<const StorageLimitsList> storage_limits)
{
    QueryTreeNodePtr modified_query_tree = query_tree->clone();
    rewriteJoinToGlobalJoin(modified_query_tree, context);
    modified_query_tree = buildQueryTreeForShard(planner_context, modified_query_tree);

    auto header
        = InterpreterSelectQueryAnalyzer::getSampleBlock(modified_query_tree, context, SelectQueryOptions(processed_stage).analyze());
    auto modified_query_ast = queryNodeToDistributedSelectQuery(modified_query_tree);

    executeQueryWithParallelReplicas(query_plan, storage_id, header, processed_stage, modified_query_ast, context, storage_limits);
}

void executeQueryWithParallelReplicas(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    QueryProcessingStage::Enum processed_stage,
    const ASTPtr & query_ast,
    ContextPtr context,
    std::shared_ptr<const StorageLimitsList> storage_limits)
{
    auto modified_query_ast = ClusterProxy::rewriteSelectQuery(
        context, query_ast, storage_id.database_name, storage_id.table_name, /*remote_table_function_ptr*/ nullptr);
    auto header = InterpreterSelectQuery(modified_query_ast, context, SelectQueryOptions(processed_stage).analyze()).getSampleBlock();

    executeQueryWithParallelReplicas(query_plan, storage_id, header, processed_stage, modified_query_ast, context, storage_limits);
}

void executeQueryWithParallelReplicasCustomKey(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    const SelectQueryInfo & query_info,
    const ColumnsDescription & columns,
    const StorageSnapshotPtr & snapshot,
    QueryProcessingStage::Enum processed_stage,
    const Block & header,
    ContextPtr context)
{
    /// Return directly (with correct header) if no shard to query.
    if (query_info.getCluster()->getShardsInfo().empty())
    {
        if (context->getSettingsRef().allow_experimental_analyzer)
            return;

        Pipe pipe(std::make_shared<NullSource>(header));
        auto read_from_pipe = std::make_unique<ReadFromPreparedSource>(std::move(pipe));
        read_from_pipe->setStepDescription("Read from NullSource (Distributed)");
        query_plan.addStep(std::move(read_from_pipe));
        return;
    }

    ColumnsDescriptionByShardNum columns_object;
    if (hasDynamicSubcolumns(columns))
        columns_object = getExtendedObjectsOfRemoteTables(*query_info.cluster, storage_id, columns, context);

    ClusterProxy::SelectStreamFactory select_stream_factory
        = ClusterProxy::SelectStreamFactory(header, columns_object, snapshot, processed_stage);

    auto shard_filter_generator = getShardFilterGeneratorForCustomKey(*query_info.getCluster(), context, columns);

    ClusterProxy::executeQuery(
        query_plan,
        header,
        processed_stage,
        storage_id,
        /*table_func_ptr=*/nullptr,
        select_stream_factory,
        getLogger("executeQueryWithParallelReplicasCustomKey"),
        context,
        query_info,
        /*sharding_key_expr=*/nullptr,
        /*sharding_key_column_name=*/{},
        /*distributed_settings=*/{},
        shard_filter_generator,
        /*is_remote_function=*/false);
}

void executeQueryWithParallelReplicasCustomKey(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    const SelectQueryInfo & query_info,
    const ColumnsDescription & columns,
    const StorageSnapshotPtr & snapshot,
    QueryProcessingStage::Enum processed_stage,
    const QueryTreeNodePtr & query_tree,
    ContextPtr context)
{
    auto header = InterpreterSelectQueryAnalyzer::getSampleBlock(query_tree, context, SelectQueryOptions(processed_stage).analyze());
    executeQueryWithParallelReplicasCustomKey(query_plan, storage_id, query_info, columns, snapshot, processed_stage, header, context);
}

void executeQueryWithParallelReplicasCustomKey(
    QueryPlan & query_plan,
    const StorageID & storage_id,
    SelectQueryInfo query_info,
    const ColumnsDescription & columns,
    const StorageSnapshotPtr & snapshot,
    QueryProcessingStage::Enum processed_stage,
    const ASTPtr & query_ast,
    ContextPtr context)
{
    auto header = InterpreterSelectQuery(query_ast, context, SelectQueryOptions(processed_stage).analyze()).getSampleBlock();
    query_info.query = ClusterProxy::rewriteSelectQuery(
        context, query_info.query, storage_id.getDatabaseName(), storage_id.getTableName(), /*table_function_ptr=*/nullptr);
    executeQueryWithParallelReplicasCustomKey(query_plan, storage_id, query_info, columns, snapshot, processed_stage, header, context);
}
}

}
