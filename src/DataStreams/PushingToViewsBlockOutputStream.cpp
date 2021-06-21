#include <DataStreams/ConvertingBlockInputStream.h>
#include <DataStreams/MaterializingBlockInputStream.h>
#include <DataStreams/OneBlockInputStream.h>
#include <DataStreams/PushingToViewsBlockOutputStream.h>
#include <DataStreams/SquashingBlockInputStream.h>
#include <DataStreams/copyData.h>
#include <DataTypes/NestedUtils.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterInsertQuery.h>
#include <Interpreters/InterpreterSelectQuery.h>
#include <Parsers/ASTInsertQuery.h>
#include <Storages/LiveView/StorageLiveView.h>
#include <Storages/MergeTree/ReplicatedMergeTreeBlockOutputStream.h>
#include <Storages/StorageValues.h>
#include <Storages/StorageMaterializedView.h>
#include <Common/CurrentThread.h>
#include <Common/MemoryTracker.h>
#include <Common/ThreadPool.h>
#include <Common/ThreadStatus.h>
#include <Common/checkStackSize.h>
#include <Common/setThreadName.h>
#include <common/logger_useful.h>
#include <DataStreams/PushingToSinkBlockOutputStream.h>

#include <atomic>
#include <chrono>

#include <Common/ProfileEvents.h>

namespace ProfileEvents
{
extern const Event SlowRead;
extern const Event MergedRows;
extern const Event ZooKeeperTransactions;
}

namespace DB
{

PushingToViewsBlockOutputStream::PushingToViewsBlockOutputStream(
    const StoragePtr & storage_,
    const StorageMetadataPtr & metadata_snapshot_,
    ContextPtr context_,
    const ASTPtr & query_ptr_,
    bool no_destination)
    : WithContext(context_)
    , storage(storage_)
    , metadata_snapshot(metadata_snapshot_)
    , log(&Poco::Logger::get("PushingToViewsBlockOutputStream"))
    , query_ptr(query_ptr_)
{
    checkStackSize();

    /** TODO This is a very important line. At any insertion into the table one of streams should own lock.
      * Although now any insertion into the table is done via PushingToViewsBlockOutputStream,
      *  but it's clear that here is not the best place for this functionality.
      */
    addTableLock(
        storage->lockForShare(getContext()->getInitialQueryId(), getContext()->getSettingsRef().lock_acquire_timeout));

    /// If the "root" table deduplicates blocks, there are no need to make deduplication for children
    /// Moreover, deduplication for AggregatingMergeTree children could produce false positives due to low size of inserting blocks
    bool disable_deduplication_for_children = false;
    if (!getContext()->getSettingsRef().deduplicate_blocks_in_dependent_materialized_views)
        disable_deduplication_for_children = !no_destination && storage->supportsDeduplication();

    auto table_id = storage->getStorageID();
    Dependencies dependencies = DatabaseCatalog::instance().getDependencies(table_id);

    /// We need special context for materialized views insertions
    if (!dependencies.empty())
    {
        select_context = Context::createCopy(context);
        insert_context = Context::createCopy(context);

        const auto & insert_settings = insert_context->getSettingsRef();

        // Do not deduplicate insertions into MV if the main insertion is Ok
        if (disable_deduplication_for_children)
            insert_context->setSetting("insert_deduplicate", Field{false});

        // Separate min_insert_block_size_rows/min_insert_block_size_bytes for children
        if (insert_settings.min_insert_block_size_rows_for_materialized_views)
            insert_context->setSetting("min_insert_block_size_rows", insert_settings.min_insert_block_size_rows_for_materialized_views.value);
        if (insert_settings.min_insert_block_size_bytes_for_materialized_views)
            insert_context->setSetting("min_insert_block_size_bytes", insert_settings.min_insert_block_size_bytes_for_materialized_views.value);
    }

    auto thread_group = CurrentThread::getGroup();

    for (const auto & database_table : dependencies)
    {
        auto dependent_table = DatabaseCatalog::instance().getTable(database_table, getContext());
        auto dependent_metadata_snapshot = dependent_table->getInMemoryMetadataPtr();

        ASTPtr query;
        BlockOutputStreamPtr out;
        QueryViewsLogElement::ViewType type = QueryViewsLogElement::ViewType::DEFAULT;
        String target_name = database_table.getNameForLogs();

        if (auto * materialized_view = dynamic_cast<StorageMaterializedView *>(dependent_table.get()))
        {
            type = QueryViewsLogElement::ViewType::MATERIALIZED;
            addTableLock(
                materialized_view->lockForShare(getContext()->getInitialQueryId(), getContext()->getSettingsRef().lock_acquire_timeout));

            StoragePtr inner_table = materialized_view->getTargetTable();
            auto inner_table_id = inner_table->getStorageID();
            auto inner_metadata_snapshot = inner_table->getInMemoryMetadataPtr();
            query = dependent_metadata_snapshot->getSelectQuery().inner_query;
            target_name = inner_table_id.getNameForLogs();

            std::unique_ptr<ASTInsertQuery> insert = std::make_unique<ASTInsertQuery>();
            insert->table_id = inner_table_id;

            /// Get list of columns we get from select query.
            auto header = InterpreterSelectQuery(query, select_context, SelectQueryOptions().analyze())
                .getSampleBlock();

            /// Insert only columns returned by select.
            auto list = std::make_shared<ASTExpressionList>();
            const auto & inner_table_columns = inner_metadata_snapshot->getColumns();
            for (const auto & column : header)
            {
                /// But skip columns which storage doesn't have.
                if (inner_table_columns.hasPhysical(column.name))
                    list->children.emplace_back(std::make_shared<ASTIdentifier>(column.name));
            }

            insert->columns = std::move(list);

            ASTPtr insert_query_ptr(insert.release());
            InterpreterInsertQuery interpreter(insert_query_ptr, insert_context);
            BlockIO io = interpreter.execute();
            out = io.out;
        }
        else if (const auto * live_view = dynamic_cast<const StorageLiveView *>(dependent_table.get()))
        {
            type = QueryViewsLogElement::ViewType::LIVE;
            query = live_view->getInnerQuery(); // TODO: Optimize this
            out = std::make_shared<PushingToViewsBlockOutputStream>(
                dependent_table, dependent_metadata_snapshot, insert_context, ASTPtr(), true);
        }
        else
            out = std::make_shared<PushingToViewsBlockOutputStream>(
                dependent_table, dependent_metadata_snapshot, insert_context, ASTPtr());

        auto main_thread = current_thread;
        auto thread_status = std::make_shared<ThreadStatus>();
        current_thread = main_thread;
        thread_status->attachQueryContext(getContext());

        QueryViewsLogElement::ViewRuntimeStats runtime_stats{
            target_name, type, thread_status, 0, std::chrono::system_clock::now(), QueryViewsLogElement::Status::QUERY_START};
        views.emplace_back(ViewInfo{std::move(query), database_table, std::move(out), nullptr, std::move(runtime_stats)});
    }

    /// Do not push to destination table if the flag is set
    if (!no_destination)
    {
        auto sink = storage->write(query_ptr, storage->getInMemoryMetadataPtr(), getContext());

        metadata_snapshot->check(sink->getPort().getHeader().getColumnsWithTypeAndName());

        replicated_output = dynamic_cast<ReplicatedMergeTreeSink *>(sink.get());
        output = std::make_shared<PushingToSinkBlockOutputStream>(std::move(sink));
    }
    ProfileEvents::increment(ProfileEvents::ZooKeeperTransactions, 100);
}


Block PushingToViewsBlockOutputStream::getHeader() const
{
    /// If we don't write directly to the destination
    /// then expect that we're inserting with precalculated virtual columns
    if (output)
        return metadata_snapshot->getSampleBlock();
    else
        return metadata_snapshot->getSampleBlockWithVirtuals(storage->getVirtuals());
}


void PushingToViewsBlockOutputStream::write(const Block & block)
{
    /** Throw an exception if the sizes of arrays - elements of nested data structures doesn't match.
      * We have to make this assertion before writing to table, because storage engine may assume that they have equal sizes.
      * NOTE It'd better to do this check in serialization of nested structures (in place when this assumption is required),
      * but currently we don't have methods for serialization of nested structures "as a whole".
      */
    Nested::validateArraySizes(block);

    if (auto * live_view = dynamic_cast<StorageLiveView *>(storage.get()))
    {
        StorageLiveView::writeIntoLiveView(*live_view, block, getContext());
    }
    else
    {
        if (output)
            /// TODO: to support virtual and alias columns inside MVs, we should return here the inserted block extended
            ///       with additional columns directly from storage and pass it to MVs instead of raw block.
            output->write(block);
    }

    if (views.empty())
        return;

    /// Don't process materialized views if this block is duplicate
    if (!getContext()->getSettingsRef().deduplicate_blocks_in_dependent_materialized_views && replicated_output && replicated_output->lastBlockIsDuplicate())
        return;

    // Push to each view. Only parallel if available
    const Settings & settings = getContext()->getSettingsRef();
    const size_t max_threads = settings.parallel_view_processing ? settings.max_threads : 1;
    ThreadPool pool(std::min(max_threads, views.size()));
    for (auto & view : views)
    {
        pool.scheduleOrThrowOnError([=, &view, this] {

            setThreadName("PushingToViews");
            current_thread = view.runtime_stats.thread_status.get();
            ProfileEvents::increment(ProfileEvents::ZooKeeperTransactions, 1);
            LOG_WARNING(log, "WRITE THREAD");

            Stopwatch watch;
            try
            {
                process(block, view);
            }
            catch (...)
            {
                view.exception = std::current_exception();
            }
            /* process might have set view.exception without throwing */
            if (view.exception)
                view.runtime_stats.setStatus(QueryViewsLogElement::Status::EXCEPTION_WHILE_PROCESSING);
            view.runtime_stats.elapsed_ms += watch.elapsedMilliseconds();
        });
    }
    // Wait for concurrent view processing
    pool.wait();
    check_exceptions_in_views();
}

void PushingToViewsBlockOutputStream::writePrefix()
{
    if (output)
        output->writePrefix();

    for (auto & view : views)
    {
        Stopwatch watch;
        try
        {
            view.out->writePrefix();
        }
        catch (Exception & ex)
        {
            ex.addMessage("while write prefix to view " + view.table_id.getNameForLogs());
            view.exception = std::current_exception();
            view.runtime_stats.setStatus(QueryViewsLogElement::Status::EXCEPTION_WHILE_PROCESSING);
            log_query_views();
            throw;
        }
        view.runtime_stats.elapsed_ms += watch.elapsedMilliseconds();
    }
}

void PushingToViewsBlockOutputStream::writeSuffix()
{
    if (output)
        output->writeSuffix();

    if (views.empty())
        return;
    std::exception_ptr first_exception;
    const Settings & settings = getContext()->getSettingsRef();

    /// Run writeSuffix() for views in separate thread pool.
    /// In could have been done in PushingToViewsBlockOutputStream::process, however
    /// it is not good if insert into main table fail but into view succeed.
    const size_t max_threads = settings.parallel_view_processing ? settings.max_threads : 1;
    ThreadPool pool(std::min(max_threads, views.size()));
    auto thread_group = CurrentThread::getGroup();

    for (auto & view : views)
    {
        if (view.exception)
            continue;

        pool.scheduleOrThrowOnError([&] {
            setThreadName("PushingToViews");
            current_thread = view.runtime_stats.thread_status.get();
            ProfileEvents::increment(ProfileEvents::ZooKeeperTransactions, 1);
            LOG_WARNING(log, "WRITE SUFFIX THREAD");

            Stopwatch watch;
            try
            {
                view.out->writeSuffix();
                view.runtime_stats.setStatus(QueryViewsLogElement::Status::QUERY_FINISH);
            }
            catch (...)
            {
                view.exception = std::current_exception();
            }
            view.runtime_stats.elapsed_ms += watch.elapsedMilliseconds();
            LOG_TRACE(
                log,
                "Pushing from {} to {} took {} ms.",
                storage->getStorageID().getNameForLogs(),
                view.table_id.getNameForLogs(),
                view.runtime_stats.elapsed_ms);
        });
    }
    // Wait for concurrent view processing
    pool.wait();
    check_exceptions_in_views();

    UInt64 milliseconds = main_watch.elapsedMilliseconds();
    if (views.size() > 1)
    {
        LOG_DEBUG(log, "Pushing from {} to {} views took {} ms.",
            storage->getStorageID().getNameForLogs(), views.size(),
            milliseconds);
    }
    log_query_views();
}

void PushingToViewsBlockOutputStream::flush()
{
    if (output)
        output->flush();

    for (auto & view : views)
        view.out->flush();
}

void PushingToViewsBlockOutputStream::process(const Block & block, ViewInfo & view)
{
    try
    {
        BlockInputStreamPtr in;

        /// We need keep InterpreterSelectQuery, until the processing will be finished, since:
        ///
        /// - We copy Context inside InterpreterSelectQuery to support
        ///   modification of context (Settings) for subqueries
        /// - InterpreterSelectQuery lives shorter than query pipeline.
        ///   It's used just to build the query pipeline and no longer needed
        /// - ExpressionAnalyzer and then, Functions, that created in InterpreterSelectQuery,
        ///   **can** take a reference to Context from InterpreterSelectQuery
        ///   (the problem raises only when function uses context from the
        ///    execute*() method, like FunctionDictGet do)
        /// - These objects live inside query pipeline (DataStreams) and the reference become dangling.
        std::optional<InterpreterSelectQuery> select;

        if (view.query)
        {
            /// We create a table with the same name as original table and the same alias columns,
            ///  but it will contain single block (that is INSERT-ed into main table).
            /// InterpreterSelectQuery will do processing of alias columns.

            auto local_context = Context::createCopy(select_context);
            local_context->addViewSource(
                StorageValues::create(storage->getStorageID(), metadata_snapshot->getColumns(), block, storage->getVirtuals()));
            select.emplace(view.query, local_context, SelectQueryOptions());
            in = std::make_shared<MaterializingBlockInputStream>(select->execute().getInputStream());

            /// Squashing is needed here because the materialized view query can generate a lot of blocks
            /// even when only one block is inserted into the parent table (e.g. if the query is a GROUP BY
            /// and two-level aggregation is triggered).
            in = std::make_shared<SquashingBlockInputStream>(
                    in, getContext()->getSettingsRef().min_insert_block_size_rows, getContext()->getSettingsRef().min_insert_block_size_bytes);
            in = std::make_shared<ConvertingBlockInputStream>(in, view.out->getHeader(), ConvertingBlockInputStream::MatchColumnsMode::Name);
        }
        else
            in = std::make_shared<OneBlockInputStream>(block);

        in->readPrefix();

        while (Block result_block = in->read())
        {
            Nested::validateArraySizes(result_block);
            view.out->write(result_block);
        }

        in->readSuffix();
    }
    catch (Exception & ex)
    {
        ex.addMessage("while pushing to view " + view.table_id.getNameForLogs());
        view.exception = std::current_exception();
    }
    catch (...)
    {
        view.exception = std::current_exception();
    }
}

void PushingToViewsBlockOutputStream::check_exceptions_in_views()
{
    for (auto & view : views)
    {
        if (view.exception)
        {
            log_query_views();
            std::rethrow_exception(view.exception);
        }
    }
}

void PushingToViewsBlockOutputStream::log_query_views()
{
    const auto & settings = getContext()->getSettingsRef();
    const UInt64 min_query_duration = settings.log_queries_min_query_duration_ms.totalMilliseconds();
    if (views.empty() || !settings.log_queries || !settings.log_query_views)
        return;

    for (auto & view : views)
    {
        if (view.runtime_stats.event_status == QueryViewsLogElement::Status::QUERY_START)
            view.runtime_stats.setStatus(QueryViewsLogElement::Status::EXCEPTION_WHILE_PROCESSING);

        if (min_query_duration && view.runtime_stats.elapsed_ms <= min_query_duration)
            continue;

        try
        {
            view.runtime_stats.thread_status->logToQueryViewsLog(view);
        }
        catch (...)
        {
            LOG_WARNING(log, getCurrentExceptionMessage(true));
        }
    }
}
}
