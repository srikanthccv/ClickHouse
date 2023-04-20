#include <Common/ConcurrentBoundedQueue.h>
#include <QueryPipeline/RemoteQueryExecutor.h>
#include <QueryPipeline/RemoteQueryExecutorReadContext.h>

#include <Columns/ColumnConst.h>
#include <Common/CurrentThread.h>
#include "Core/Protocol.h"
#include <Processors/QueryPlan/BuildQueryPipelineSettings.h>
#include <Processors/QueryPlan/Optimizations/QueryPlanOptimizationSettings.h>
#include <Processors/Sources/SourceFromSingleChunk.h>
#include <Processors/Transforms/LimitsCheckingTransform.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Storages/IStorage.h>
#include <Storages/SelectQueryInfo.h>
#include <Interpreters/castColumn.h>
#include <Interpreters/Cluster.h>
#include <Interpreters/Context.h>
#include <Interpreters/InternalTextLogsQueue.h>
#include <IO/ConnectionTimeouts.h>
#include <Client/MultiplexedConnections.h>
#include <Client/HedgedConnections.h>
#include <Storages/MergeTree/MergeTreeDataPartUUID.h>
#include <Storages/StorageMemory.h>


namespace ProfileEvents
{
    extern const Event ReadTaskRequestsReceived;
    extern const Event MergeTreeReadTaskRequestsReceived;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int UNKNOWN_PACKET_FROM_SERVER;
    extern const int DUPLICATED_PART_UUIDS;
    extern const int SYSTEM_ERROR;
}

RemoteQueryExecutor::RemoteQueryExecutor(
    const String & query_, const Block & header_, ContextPtr context_,
    const Scalars & scalars_, const Tables & external_tables_,
    QueryProcessingStage::Enum stage_, std::optional<Extension> extension_)
    : header(header_), query(query_), context(context_), scalars(scalars_)
    , external_tables(external_tables_), stage(stage_)
    , task_iterator(extension_ ? extension_->task_iterator : nullptr)
    , parallel_reading_coordinator(extension_ ? extension_->parallel_reading_coordinator : nullptr)
{}

RemoteQueryExecutor::RemoteQueryExecutor(
    Connection & connection,
    const String & query_, const Block & header_, ContextPtr context_,
    ThrottlerPtr throttler, const Scalars & scalars_, const Tables & external_tables_,
    QueryProcessingStage::Enum stage_, std::optional<Extension> extension_)
    : RemoteQueryExecutor(query_, header_, context_, scalars_, external_tables_, stage_, extension_)
{
    create_connections = [this, &connection, throttler, extension_]()
    {
        auto res = std::make_unique<MultiplexedConnections>(connection, context->getSettingsRef(), throttler);
        if (extension_ && extension_->replica_info)
            res->setReplicaInfo(*extension_->replica_info);
        return res;
    };
}

RemoteQueryExecutor::RemoteQueryExecutor(
    std::shared_ptr<Connection> connection_ptr,
    const String & query_, const Block & header_, ContextPtr context_,
    ThrottlerPtr throttler, const Scalars & scalars_, const Tables & external_tables_,
    QueryProcessingStage::Enum stage_, std::optional<Extension> extension_)
    : RemoteQueryExecutor(query_, header_, context_, scalars_, external_tables_, stage_, extension_)
{
    create_connections = [this, connection_ptr, throttler, extension_]()
    {
        auto res = std::make_unique<MultiplexedConnections>(connection_ptr, context->getSettingsRef(), throttler);
        if (extension_ && extension_->replica_info)
            res->setReplicaInfo(*extension_->replica_info);
        return res;
    };
}

RemoteQueryExecutor::RemoteQueryExecutor(
    std::vector<IConnectionPool::Entry> && connections_,
    const String & query_, const Block & header_, ContextPtr context_,
    const ThrottlerPtr & throttler, const Scalars & scalars_, const Tables & external_tables_,
    QueryProcessingStage::Enum stage_, std::optional<Extension> extension_)
    : header(header_), query(query_), context(context_)
    , scalars(scalars_), external_tables(external_tables_), stage(stage_)
    , task_iterator(extension_ ? extension_->task_iterator : nullptr)
    , parallel_reading_coordinator(extension_ ? extension_->parallel_reading_coordinator : nullptr)
{
    create_connections = [this, connections_, throttler, extension_]() mutable {
        auto res = std::make_unique<MultiplexedConnections>(std::move(connections_), context->getSettingsRef(), throttler);
        if (extension_ && extension_->replica_info)
            res->setReplicaInfo(*extension_->replica_info);
        return res;
    };
}

RemoteQueryExecutor::RemoteQueryExecutor(
    const ConnectionPoolWithFailoverPtr & pool,
    const String & query_, const Block & header_, ContextPtr context_,
    const ThrottlerPtr & throttler, const Scalars & scalars_, const Tables & external_tables_,
    QueryProcessingStage::Enum stage_, std::optional<Extension> extension_)
    : header(header_), query(query_), context(context_)
    , scalars(scalars_), external_tables(external_tables_), stage(stage_)
    , task_iterator(extension_ ? extension_->task_iterator : nullptr)
    , parallel_reading_coordinator(extension_ ? extension_->parallel_reading_coordinator : nullptr)
{
    create_connections = [this, pool, throttler, extension_]()->std::unique_ptr<IConnections>
    {
        const Settings & current_settings = context->getSettingsRef();
        auto timeouts = ConnectionTimeouts::getTCPTimeoutsWithFailover(current_settings);

#if defined(OS_LINUX)
        if (current_settings.use_hedged_requests)
        {
            std::shared_ptr<QualifiedTableName> table_to_check = nullptr;
            if (main_table)
                table_to_check = std::make_shared<QualifiedTableName>(main_table.getQualifiedName());

            auto res = std::make_unique<HedgedConnections>(pool, context, timeouts, throttler, pool_mode, table_to_check);
            if (extension_ && extension_->replica_info)
                res->setReplicaInfo(*extension_->replica_info);
            return res;
        }
#endif

        std::vector<IConnectionPool::Entry> connection_entries;
        if (main_table)
        {
            auto try_results = pool->getManyChecked(timeouts, &current_settings, pool_mode, main_table.getQualifiedName());
            connection_entries.reserve(try_results.size());
            for (auto & try_result : try_results)
                connection_entries.emplace_back(std::move(try_result.entry));
        }
        else
            connection_entries = pool->getMany(timeouts, &current_settings, pool_mode);

        auto res = std::make_unique<MultiplexedConnections>(std::move(connection_entries), current_settings, throttler);
        if (extension_ && extension_->replica_info)
            res->setReplicaInfo(*extension_->replica_info);
        return res;
    };
}

RemoteQueryExecutor::~RemoteQueryExecutor()
{
    /** If interrupted in the middle of the loop of communication with replicas, then interrupt
      * all connections, then read and skip the remaining packets to make sure
      * these connections did not remain hanging in the out-of-sync state.
      */
    if (established || isQueryPending())
        connections->disconnect();
}

/** If we receive a block with slightly different column types, or with excessive columns,
  *  we will adapt it to expected structure.
  */
static Block adaptBlockStructure(const Block & block, const Block & header)
{
    /// Special case when reader doesn't care about result structure. Deprecated and used only in Benchmark, PerformanceTest.
    if (!header)
        return block;

    Block res;
    res.info = block.info;

    for (const auto & elem : header)
    {
        ColumnPtr column;

        if (elem.column && isColumnConst(*elem.column))
        {
            /// We expect constant column in block.
            /// If block is not empty, then get value for constant from it,
            /// because it may be different for remote server for functions like version(), uptime(), ...
            if (block.rows() > 0 && block.has(elem.name))
            {
                /// Const column is passed as materialized. Get first value from it.
                ///
                /// TODO: check that column contains the same value.
                /// TODO: serialize const columns.
                auto col = block.getByName(elem.name);
                col.column = block.getByName(elem.name).column->cut(0, 1);

                column = castColumn(col, elem.type);

                if (!isColumnConst(*column))
                    column = ColumnConst::create(column, block.rows());
                else
                    /// It is not possible now. Just in case we support const columns serialization.
                    column = column->cloneResized(block.rows());
            }
            else
                column = elem.column->cloneResized(block.rows());
        }
        else
            column = castColumn(block.getByName(elem.name), elem.type);

        res.insert({column, elem.type, elem.name});
    }
    return res;
}

void RemoteQueryExecutor::sendQuery(ClientInfo::QueryKind query_kind)
{
    if (sent_query)
        return;

    connections = create_connections();

    const auto & settings = context->getSettingsRef();
    if (settings.skip_unavailable_shards && 0 == connections->size())
    {
        /// To avoid sending the query again in the read(), we need to update the following flags:
        std::lock_guard guard(was_cancelled_mutex);
        was_cancelled = true;
        finished = true;
        sent_query = true;

        return;
    }

    /// Query cannot be canceled in the middle of the send query,
    /// since there are multiple packets:
    /// - Query
    /// - Data (multiple times)
    ///
    /// And after the Cancel packet none Data packet can be sent, otherwise the remote side will throw:
    ///
    ///     Unexpected packet Data received from client
    ///
    std::lock_guard guard(was_cancelled_mutex);

    established = true;
    was_cancelled = false;

    auto timeouts = ConnectionTimeouts::getTCPTimeoutsWithFailover(settings);
    ClientInfo modified_client_info = context->getClientInfo();
    modified_client_info.query_kind = query_kind;

    if (!duplicated_part_uuids.empty())
        connections->sendIgnoredPartUUIDs(duplicated_part_uuids);

    connections->sendQuery(timeouts, query, query_id, stage, modified_client_info, true);

    established = false;
    sent_query = true;

    if (settings.enable_scalar_subquery_optimization)
        sendScalars();
    sendExternalTables();
}


Block RemoteQueryExecutor::readBlock()
{
    while (true)
    {
        auto res = read();

        if (res.getType() == ReadResult::Type::Data)
            return res.getBlock();
    }
}


RemoteQueryExecutor::ReadResult RemoteQueryExecutor::read()
{
    if (!sent_query)
    {
        sendQuery();

        if (context->getSettingsRef().skip_unavailable_shards && (0 == connections->size()))
            return ReadResult(Block());
    }

    while (true)
    {
        std::lock_guard lock(was_cancelled_mutex);
        if (was_cancelled)
            return ReadResult(Block());

        auto packet = connections->receivePacket();
        auto anything = processPacket(std::move(packet));

        if (anything.getType() == ReadResult::Type::Data || anything.getType() == ReadResult::Type::ParallelReplicasToken)
            return anything;

        if (got_duplicated_part_uuids)
            return restartQueryWithoutDuplicatedUUIDs();
    }
}

RemoteQueryExecutor::ReadResult RemoteQueryExecutor::read(std::unique_ptr<ReadContext> & read_context [[maybe_unused]])
{
#if defined(OS_LINUX)
    if (!sent_query)
    {
        sendQuery();

        if (context->getSettingsRef().skip_unavailable_shards && (0 == connections->size()))
            return ReadResult(Block());
    }

    if (!read_context || resent_query)
    {
        std::lock_guard lock(was_cancelled_mutex);
        if (was_cancelled)
            return ReadResult(Block());

        read_context = std::make_unique<ReadContext>(*connections);
    }

    while (true)
    {
        if (!read_context->resumeRoutine())
            return ReadResult(Block());

        if (read_context->is_read_in_progress.load(std::memory_order_relaxed))
        {
            read_context->setTimer();
            return ReadResult(read_context->epoll.getFileDescriptor());
        }
        else
        {
            /// We need to check that query was not cancelled again,
            /// to avoid the race between cancel() thread and read() thread.
            /// (since cancel() thread will steal the fiber and may update the packet).
            if (was_cancelled)
                return ReadResult(Block());

            auto anything = processPacket(std::move(read_context->packet));

            if (anything.getType() == ReadResult::Type::Data || anything.getType() == ReadResult::Type::ParallelReplicasToken)
                return anything;

            if (got_duplicated_part_uuids)
                return restartQueryWithoutDuplicatedUUIDs(&read_context);
        }
    }
#else
    return read();
#endif
}


RemoteQueryExecutor::ReadResult RemoteQueryExecutor::restartQueryWithoutDuplicatedUUIDs(std::unique_ptr<ReadContext> * read_context)
{
    /// Cancel previous query and disconnect before retry.
    cancel(read_context);
    connections->disconnect();

    /// Only resend once, otherwise throw an exception
    if (!resent_query)
    {
        if (log)
            LOG_DEBUG(log, "Found duplicate UUIDs, will retry query without those parts");

        resent_query = true;
        sent_query = false;
        got_duplicated_part_uuids = false;
        /// Consecutive read will implicitly send query first.
        if (!read_context)
            return read();
        else
            return read(*read_context);
    }
    throw Exception(ErrorCodes::DUPLICATED_PART_UUIDS, "Found duplicate uuids while processing query");
}

RemoteQueryExecutor::ReadResult RemoteQueryExecutor::processPacket(Packet packet)
{
    switch (packet.type)
    {
        case Protocol::Server::MergeTreeReadTaskRequest:
            processMergeTreeReadTaskRequest(packet.request);
            return ReadResult(ReadResult::Type::ParallelReplicasToken);

        case Protocol::Server::MergeTreeAllRangesAnnounecement:
            processMergeTreeInitialReadAnnounecement(packet.announcement);
            return ReadResult(ReadResult::Type::ParallelReplicasToken);

        case Protocol::Server::ReadTaskRequest:
            processReadTaskRequest();
            break;
        case Protocol::Server::PartUUIDs:
            if (!setPartUUIDs(packet.part_uuids))
                got_duplicated_part_uuids = true;
            break;
        case Protocol::Server::Data:
            /// Note: `packet.block.rows() > 0` means it's a header block.
            /// We can actually return it, and the first call to RemoteQueryExecutor::read
            /// will return earlier. We should consider doing it.
            if (packet.block && (packet.block.rows() > 0))
                return ReadResult(adaptBlockStructure(packet.block, header));
            break;  /// If the block is empty - we will receive other packets before EndOfStream.

        case Protocol::Server::Exception:
            got_exception_from_replica = true;
            packet.exception->rethrow();
            break;

        case Protocol::Server::EndOfStream:
            if (!connections->hasActiveConnections())
            {
                finished = true;
                /// TODO: Replace with Type::Finished
                return ReadResult(Block{});
            }
            break;

        case Protocol::Server::Progress:
            /** We use the progress from a remote server.
              * We also include in ProcessList,
              * and we use it to check
              * constraints (for example, the minimum speed of query execution)
              * and quotas (for example, the number of lines to read).
              */
            if (progress_callback)
                progress_callback(packet.progress);
            break;

        case Protocol::Server::ProfileInfo:
            /// Use own (client-side) info about read bytes, it is more correct info than server-side one.
            if (profile_info_callback)
                profile_info_callback(packet.profile_info);
            break;

        case Protocol::Server::Totals:
            totals = packet.block;
            if (totals)
                totals = adaptBlockStructure(totals, header);
            break;

        case Protocol::Server::Extremes:
            extremes = packet.block;
            if (extremes)
                extremes = adaptBlockStructure(packet.block, header);
            break;

        case Protocol::Server::Log:
            /// Pass logs from remote server to client
            if (auto log_queue = CurrentThread::getInternalTextLogsQueue())
                log_queue->pushBlock(std::move(packet.block));
            break;

        case Protocol::Server::ProfileEvents:
            /// Pass profile events from remote server to client
            if (auto profile_queue = CurrentThread::getInternalProfileEventsQueue())
                if (!profile_queue->emplace(std::move(packet.block)))
                    throw Exception(ErrorCodes::SYSTEM_ERROR, "Could not push into profile queue");
            break;

        case Protocol::Server::TimezoneUpdate:
            break;

        default:
            got_unknown_packet_from_replica = true;
            throw Exception(
                ErrorCodes::UNKNOWN_PACKET_FROM_SERVER,
                "Unknown packet {} from one of the following replicas: {}",
                packet.type,
                connections->dumpAddresses());
    }

    return ReadResult(ReadResult::Type::Nothing);
}

bool RemoteQueryExecutor::setPartUUIDs(const std::vector<UUID> & uuids)
{
    auto query_context = context->getQueryContext();
    auto duplicates = query_context->getPartUUIDs()->add(uuids);

    if (!duplicates.empty())
    {
        duplicated_part_uuids.insert(duplicated_part_uuids.begin(), duplicates.begin(), duplicates.end());
        return false;
    }
    return true;
}

void RemoteQueryExecutor::processReadTaskRequest()
{
    if (!task_iterator)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Distributed task iterator is not initialized");

    ProfileEvents::increment(ProfileEvents::ReadTaskRequestsReceived);
    auto response = (*task_iterator)();
    connections->sendReadTaskResponse(response);
}

void RemoteQueryExecutor::processMergeTreeReadTaskRequest(ParallelReadRequest request)
{
    if (!parallel_reading_coordinator)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Coordinator for parallel reading from replicas is not initialized");

    ProfileEvents::increment(ProfileEvents::MergeTreeReadTaskRequestsReceived);
    auto response = parallel_reading_coordinator->handleRequest(std::move(request));
    connections->sendMergeTreeReadTaskResponse(response);
}

void RemoteQueryExecutor::processMergeTreeInitialReadAnnounecement(InitialAllRangesAnnouncement announcement)
{
    if (!parallel_reading_coordinator)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Coordinator for parallel reading from replicas is not initialized");

    parallel_reading_coordinator->handleInitialAllRangesAnnouncement(announcement);
}

void RemoteQueryExecutor::finish(std::unique_ptr<ReadContext> * read_context)
{
    /** If one of:
      * - nothing started to do;
      * - received all packets before EndOfStream;
      * - received exception from one replica;
      * - received an unknown packet from one replica;
      * then you do not need to read anything.
      */
    if (!isQueryPending() || hasThrownException())
        return;

    /** If you have not read all the data yet, but they are no longer needed.
      * This may be due to the fact that the data is sufficient (for example, when using LIMIT).
      */

    /// Send the request to abort the execution of the request, if not already sent.
    tryCancel("Cancelling query because enough data has been read", read_context);

    /// Get the remaining packets so that there is no out of sync in the connections to the replicas.
    Packet packet = connections->drain();
    switch (packet.type)
    {
        case Protocol::Server::EndOfStream:
            finished = true;
            break;

        case Protocol::Server::Log:
            /// Pass logs from remote server to client
            if (auto log_queue = CurrentThread::getInternalTextLogsQueue())
                log_queue->pushBlock(std::move(packet.block));
            break;

        case Protocol::Server::Exception:
            got_exception_from_replica = true;
            packet.exception->rethrow();
            break;

        case Protocol::Server::ProfileEvents:
            /// Pass profile events from remote server to client
            if (auto profile_queue = CurrentThread::getInternalProfileEventsQueue())
                if (!profile_queue->emplace(std::move(packet.block)))
                    throw Exception(ErrorCodes::SYSTEM_ERROR, "Could not push into profile queue");
            break;

        case Protocol::Server::TimezoneUpdate:
            break;

        default:
            got_unknown_packet_from_replica = true;
            throw Exception(ErrorCodes::UNKNOWN_PACKET_FROM_SERVER, "Unknown packet {} from one of the following replicas: {}",
                toString(packet.type),
                connections->dumpAddresses());
    }
}

void RemoteQueryExecutor::cancel(std::unique_ptr<ReadContext> * read_context)
{
    {
        std::lock_guard lock(external_tables_mutex);

        /// Stop sending external data.
        for (auto & vec : external_tables_data)
            for (auto & elem : vec)
                elem->is_cancelled = true;
    }

    if (!isQueryPending() || hasThrownException())
        return;

    tryCancel("Cancelling query", read_context);
}

void RemoteQueryExecutor::sendScalars()
{
    connections->sendScalarsData(scalars);
}

void RemoteQueryExecutor::sendExternalTables()
{
    size_t count = connections->size();

    {
        std::lock_guard lock(external_tables_mutex);

        external_tables_data.clear();
        external_tables_data.reserve(count);

        StreamLocalLimits limits;
        const auto & settings = context->getSettingsRef();
        limits.mode = LimitsMode::LIMITS_TOTAL;
        limits.speed_limits.max_execution_time = settings.max_execution_time;
        limits.timeout_overflow_mode = settings.timeout_overflow_mode;

        for (size_t i = 0; i < count; ++i)
        {
            ExternalTablesData res;
            for (const auto & table : external_tables)
            {
                StoragePtr cur = table.second;
                /// Send only temporary tables with StorageMemory
                if (!std::dynamic_pointer_cast<StorageMemory>(cur))
                    continue;

                auto data = std::make_unique<ExternalTableData>();
                data->table_name = table.first;
                data->creating_pipe_callback = [cur, limits, context = this->context]()
                {
                    SelectQueryInfo query_info;
                    auto metadata_snapshot = cur->getInMemoryMetadataPtr();
                    auto storage_snapshot = cur->getStorageSnapshot(metadata_snapshot, context);
                    QueryProcessingStage::Enum read_from_table_stage = cur->getQueryProcessingStage(
                        context, QueryProcessingStage::Complete, storage_snapshot, query_info);

                    QueryPlan plan;
                    cur->read(
                        plan,
                        metadata_snapshot->getColumns().getNamesOfPhysical(),
                        storage_snapshot, query_info, context,
                        read_from_table_stage, DEFAULT_BLOCK_SIZE, 1);

                    auto builder = plan.buildQueryPipeline(
                        QueryPlanOptimizationSettings::fromContext(context),
                        BuildQueryPipelineSettings::fromContext(context));

                    builder->resize(1);
                    builder->addTransform(std::make_shared<LimitsCheckingTransform>(builder->getHeader(), limits));

                    return builder;
                };

                data->pipe = data->creating_pipe_callback();
                res.emplace_back(std::move(data));
            }
            external_tables_data.push_back(std::move(res));
        }
    }

    connections->sendExternalTablesData(external_tables_data);
}

void RemoteQueryExecutor::tryCancel(const char * reason, std::unique_ptr<ReadContext> * read_context)
{
    /// Flag was_cancelled is atomic because it is checked in read(),
    /// in case of packet had been read by fiber (async_socket_for_remote).
    std::lock_guard guard(was_cancelled_mutex);

    if (was_cancelled)
        return;

    was_cancelled = true;

    if (read_context && *read_context)
    {
        /// The timer should be set for query cancellation to avoid query cancellation hung.
        ///
        /// Since in case the remote server will abnormally terminated, neither
        /// FIN nor RST packet will be sent, and the initiator will not know that
        /// the connection died (unless tcp_keep_alive_timeout > 0).
        ///
        /// Also note that it is possible to get this situation even when
        /// enough data already had been read.
        (*read_context)->setTimer();
        (*read_context)->cancel();
    }

    connections->sendCancel();

    if (log)
        LOG_TRACE(log, "({}) {}", connections->dumpAddresses(), reason);
}

bool RemoteQueryExecutor::isQueryPending() const
{
    return sent_query && !finished;
}

bool RemoteQueryExecutor::hasThrownException() const
{
    return got_exception_from_replica || got_unknown_packet_from_replica;
}

}
