#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterInsertQuery.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTCreateQuery.h>
#include <Parsers/ASTExpressionList.h>
#include <Parsers/ASTInsertQuery.h>
#include <Parsers/ASTLiteral.h>
#include <Processors/Executors/PullingPipelineExecutor.h>
#include <Processors/Pipe.h>
#include <Processors/Sources/SourceFromInputStream.h>
#include <Processors/Transforms/ExpressionTransform.h>
#include <Storages/FileLog/FileLogSource.h>
#include <Storages/FileLog/ReadBufferFromFileLog.h>
#include <Storages/FileLog/StorageFileLog.h>
#include <Storages/StorageFactory.h>
#include <Storages/StorageMaterializedView.h>
#include <Common/Exception.h>
#include <Common/Macros.h>
#include <Common/getNumberOfPhysicalCPUCores.h>
#include <Common/quoteString.h>
#include <Common/typeid_cast.h>
#include <common/logger_useful.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

namespace
{
    const auto RESCHEDULE_MS = 500;
    const auto BACKOFF_TRESHOLD = 32000;
    const auto MAX_THREAD_WORK_DURATION_MS = 60000;
}

StorageFileLog::StorageFileLog(
    const StorageID & table_id_,
    ContextPtr context_,
    const ColumnsDescription & columns_,
    const String & relative_path_,
    const String & format_name_,
    std::unique_ptr<FileLogSettings> settings)
    : IStorage(table_id_)
    , WithContext(context_->getGlobalContext())
    , filelog_settings(std::move(settings))
    , path(getContext()->getUserFilesPath() + "/" + relative_path_)
    , format_name(format_name_)
    , log(&Poco::Logger::get("StorageFileLog (" + table_id_.table_name + ")"))
    , milliseconds_to_wait(RESCHEDULE_MS)
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    setInMemoryMetadata(storage_metadata);

    try
    {
        init();
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

void StorageFileLog::init()
{
    if (std::filesystem::is_regular_file(path))
    {
        auto normal_path = std::filesystem::path(path).lexically_normal().native();
        file_statuses[normal_path] = FileContext{};
        file_names.push_back(normal_path);
    }
    else if (std::filesystem::is_directory(path))
    {
        /// Just consider file with depth 1
        for (const auto & dir_entry : std::filesystem::directory_iterator{path})
        {
            if (dir_entry.is_regular_file())
            {
                auto normal_path = std::filesystem::path(dir_entry.path()).lexically_normal().native();
                file_statuses[normal_path] = FileContext{};
                file_names.push_back(normal_path);
            }
        }
    }
    else
    {
        throw Exception("The path neither a regular file, nor a directory", ErrorCodes::BAD_ARGUMENTS);
    }

    directory_watch = std::make_unique<FileLogDirectoryWatcher>(path);

    auto thread = getContext()->getMessageBrokerSchedulePool().createTask(log->name(), [this] { threadFunc(); });
    task = std::make_shared<TaskContext>(std::move(thread));
}

Pipe StorageFileLog::read(
    const Names & column_names,
    const StorageMetadataPtr & metadata_snapshot,
    SelectQueryInfo & /* query_info */,
    ContextPtr local_context,
    QueryProcessingStage::Enum /* processed_stage */,
    size_t /* max_block_size */,
    unsigned /* num_streams */)
{
    std::lock_guard<std::mutex> lock(status_mutex);

    updateFileStatuses();

    /// No files to parse
    if (file_names.empty())
    {
        return Pipe{};
    }

    auto modified_context = Context::createCopy(local_context);

    auto max_streams_number = std::min<UInt64>(filelog_settings->filelog_max_threads, file_names.size());

    Pipes pipes;
    pipes.reserve(max_streams_number);
    for (size_t stream_number = 0; stream_number < max_streams_number; ++stream_number)
    {
        pipes.emplace_back(std::make_shared<FileLogSource>(
            *this,
            metadata_snapshot,
            modified_context,
            getMaxBlockSize(),
            getPollTimeoutMillisecond(),
            stream_number,
            max_streams_number));
    }

    auto pipe = Pipe::unitePipes(std::move(pipes));

    auto convert_actions_dag = ActionsDAG::makeConvertingActions(
        pipe.getHeader().getColumnsWithTypeAndName(),
        metadata_snapshot->getSampleBlockForColumns(column_names, getVirtuals(), getStorageID()).getColumnsWithTypeAndName(),
        ActionsDAG::MatchColumnsMode::Name);

    auto actions = std::make_shared<ExpressionActions>(
        convert_actions_dag, ExpressionActionsSettings::fromContext(getContext(), CompileExpressions::yes));

    pipe.addSimpleTransform([&](const Block & stream_header) { return std::make_shared<ExpressionTransform>(stream_header, actions); });

    return pipe;
}

void StorageFileLog::startup()
{
    if (task)
    {
        task->holder->activateAndSchedule();
    }
}


void StorageFileLog::shutdown()
{
    if (task)
    {
        task->stream_cancelled = true;

        LOG_TRACE(log, "Waiting for cleanup");
        task->holder->deactivate();
    }
}

size_t StorageFileLog::getMaxBlockSize() const
{
    return filelog_settings->filelog_max_block_size.changed ? filelog_settings->filelog_max_block_size.value
                                                            : getContext()->getSettingsRef().max_insert_block_size.value;
}

size_t StorageFileLog::getPollMaxBatchSize() const
{
    size_t batch_size = filelog_settings->filelog_poll_max_batch_size.changed ? filelog_settings->filelog_poll_max_batch_size.value
                                                                              : getContext()->getSettingsRef().max_block_size.value;
    return std::min(batch_size, getMaxBlockSize());
}

size_t StorageFileLog::getPollTimeoutMillisecond() const
{
    return filelog_settings->filelog_poll_timeout_ms.changed ? filelog_settings->filelog_poll_timeout_ms.totalMilliseconds()
                                                             : getContext()->getSettingsRef().stream_poll_timeout_ms.totalMilliseconds();
}


bool StorageFileLog::checkDependencies(const StorageID & table_id)
{
    // Check if all dependencies are attached
    auto dependencies = DatabaseCatalog::instance().getDependencies(table_id);
    if (dependencies.empty())
        return true;

    for (const auto & storage : dependencies)
    {
        auto table = DatabaseCatalog::instance().tryGetTable(storage, getContext());
        if (!table)
            return false;

        // If it materialized view, check it's target table
        auto * materialized_view = dynamic_cast<StorageMaterializedView *>(table.get());
        if (materialized_view && !materialized_view->tryGetTargetTable())
            return false;

        // Check all its dependencies
        if (!checkDependencies(storage))
            return false;
    }

    return true;
}

void StorageFileLog::threadFunc()
{
    try
    {
        auto table_id = getStorageID();
        // Check if at least one direct dependency is attached
        size_t dependencies_count = DatabaseCatalog::instance().getDependencies(table_id).size();
        if (dependencies_count)
        {
            auto start_time = std::chrono::steady_clock::now();

            // Keep streaming as long as there are attached views and streaming is not cancelled
            while (!task->stream_cancelled)
            {
                if (!checkDependencies(table_id))
                    break;

                LOG_DEBUG(log, "Started streaming to {} attached views", dependencies_count);

                if (streamToViews())
                {
                    LOG_TRACE(log, "Stream stalled. Reschedule.");
                    if (milliseconds_to_wait < BACKOFF_TRESHOLD)
                        milliseconds_to_wait *= 2;
                    break;
                }
                else
                {
                    milliseconds_to_wait = RESCHEDULE_MS;
                }

                auto ts = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ts-start_time);
                if (duration.count() > MAX_THREAD_WORK_DURATION_MS)
                {
                    LOG_TRACE(log, "Thread work duration limit exceeded. Reschedule.");
                    break;
                }
                updateFileStatuses();
            }
        }
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }

    // Wait for attached views
    if (!task->stream_cancelled)
        task->holder->scheduleAfter(milliseconds_to_wait);
}


bool StorageFileLog::streamToViews()
{
    std::lock_guard<std::mutex> lock(status_mutex);
    Stopwatch watch;

    auto table_id = getStorageID();
    auto table = DatabaseCatalog::instance().getTable(table_id, getContext());
    if (!table)
        throw Exception("Engine table " + table_id.getNameForLogs() + " doesn't exist.", ErrorCodes::LOGICAL_ERROR);
    auto metadata_snapshot = getInMemoryMetadataPtr();

    auto max_streams_number = std::min<UInt64>(filelog_settings->filelog_max_threads.value, file_names.size());
    /// No files to parse
    if (max_streams_number == 0)
    {
        return false;
    }

    // Create an INSERT query for streaming data
    auto insert = std::make_shared<ASTInsertQuery>();
    insert->table_id = table_id;

    auto new_context = Context::createCopy(getContext());

    InterpreterInsertQuery interpreter(insert, new_context, false, true, true);
    auto block_io = interpreter.execute();

    Pipes pipes;
    pipes.reserve(max_streams_number);
    for (size_t stream_number = 0; stream_number < max_streams_number; ++stream_number)
    {
        pipes.emplace_back(std::make_shared<FileLogSource>(
            *this,
            metadata_snapshot,
            new_context,
            getPollMaxBatchSize(),
            getPollTimeoutMillisecond(),
            stream_number,
            max_streams_number));
    }
    auto pipe = Pipe::unitePipes(std::move(pipes));

    auto convert_actions_dag = ActionsDAG::makeConvertingActions(
        pipe.getHeader().getColumnsWithTypeAndName(),
        block_io.out->getHeader().getColumnsWithTypeAndName(),
        ActionsDAG::MatchColumnsMode::Name);

    auto actions = std::make_shared<ExpressionActions>(
        convert_actions_dag, ExpressionActionsSettings::fromContext(getContext(), CompileExpressions::yes));

    pipe.addSimpleTransform([&](const Block & stream_header) { return std::make_shared<ExpressionTransform>(stream_header, actions); });

    QueryPipeline pipeline;
    pipeline.init(std::move(pipe));

    assertBlocksHaveEqualStructure(pipeline.getHeader(), block_io.out->getHeader(), "StorageFileLog streamToViews");

    size_t rows = 0;

    PullingPipelineExecutor executor(pipeline);
    Block block;
    block_io.out->writePrefix();
    while (executor.pull(block))
    {
        block_io.out->write(block);
        rows += block.rows();
    }
    block_io.out->writeSuffix();

    UInt64 milliseconds = watch.elapsedMilliseconds();
    LOG_DEBUG(log, "Pushing {} rows to {} took {} ms.",
        formatReadableQuantity(rows), table_id.getNameForLogs(), milliseconds);

    return updateFileStatuses();
}

void registerStorageFileLog(StorageFactory & factory)
{
    auto creator_fn = [](const StorageFactory::Arguments & args)
    {
        ASTs & engine_args = args.engine_args;
        size_t args_count = engine_args.size();

        bool has_settings = args.storage_def->settings;

        auto filelog_settings = std::make_unique<FileLogSettings>();
        if (has_settings)
        {
            filelog_settings->loadFromQuery(*args.storage_def);
        }

        auto physical_cpu_cores = getNumberOfPhysicalCPUCores();
        auto num_threads = filelog_settings->filelog_max_threads.value;

        if (num_threads > physical_cpu_cores)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Number of threads to parse files can not be bigger than {}", physical_cpu_cores);
        }
        else if (num_threads < 1)
        {
            throw Exception("Number of threads to parse files can not be lower than 1", ErrorCodes::BAD_ARGUMENTS);
        }

        if (filelog_settings->filelog_max_block_size.changed && filelog_settings->filelog_max_block_size.value < 1)
        {
            throw Exception("filelog_max_block_size can not be lower than 1", ErrorCodes::BAD_ARGUMENTS);
        }

        if (filelog_settings->filelog_poll_max_batch_size.changed && filelog_settings->filelog_poll_max_batch_size.value < 1)
        {
            throw Exception("filelog_poll_max_batch_size can not be lower than 1", ErrorCodes::BAD_ARGUMENTS);
        }

        if (args_count != 2)
            throw Exception(
                "Arguments size of StorageFileLog should be 2, path and format name", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        auto path_ast = evaluateConstantExpressionAsLiteral(engine_args[0], args.getContext());
        auto format_ast = evaluateConstantExpressionAsLiteral(engine_args[1], args.getContext());

        auto path = path_ast->as<ASTLiteral &>().value.safeGet<String>();
        auto format = format_ast->as<ASTLiteral &>().value.safeGet<String>();

        return StorageFileLog::create(args.table_id, args.getContext(), args.columns, path, format, std::move(filelog_settings));
    };

    factory.registerStorage(
        "FileLog",
        creator_fn,
        StorageFactory::StorageFeatures{
            .supports_settings = true,
        });
}

bool StorageFileLog::updateFileStatuses()
{
    if (!directory_watch)
        return false;
    /// Do not need to hold file_status lock, since it will be holded
    /// by caller when call this function
    auto error = directory_watch->getErrorAndReset();
    if (error.has_error)
        LOG_ERROR(log, "Error happened during watching directory {}: {}", directory_watch->getPath(), error.error_msg);

    auto events = directory_watch->getEventsAndReset();

    for (const auto & event : events)
    {
        switch (event.type)
        {
            case Poco::DirectoryWatcher::DW_ITEM_ADDED: {
                auto normal_path = std::filesystem::path(event.path).lexically_normal().native();
                LOG_TRACE(log, "New event {} watched, path: {}", event.callback, event.path);
                if (std::filesystem::is_regular_file(normal_path))
                {
                    file_statuses[event.path] = FileContext{};
                    file_names.push_back(normal_path);
                }
                break;
            }

            case Poco::DirectoryWatcher::DW_ITEM_MODIFIED: {
                auto normal_path = std::filesystem::path(event.path).lexically_normal().native();
                LOG_TRACE(log, "New event {} watched, path: {}", event.callback, normal_path);
                if (std::filesystem::is_regular_file(normal_path) && file_statuses.contains(normal_path))
                {
                    file_statuses[normal_path].status = FileStatus::UPDATED;
                }
                break;
            }

            case Poco::DirectoryWatcher::DW_ITEM_REMOVED:
            case Poco::DirectoryWatcher::DW_ITEM_MOVED_TO:
            case Poco::DirectoryWatcher::DW_ITEM_MOVED_FROM: {
                auto normal_path = std::filesystem::path(event.path).lexically_normal().native();
                LOG_TRACE(log, "New event {} watched, path: {}", event.callback, normal_path);
                if (std::filesystem::is_regular_file(normal_path) && file_statuses.contains(normal_path))
                {
                    file_statuses[normal_path].status = FileStatus::REMOVED;
                }
            }
        }
    }
    std::vector<String> valid_files;
    for (const auto & file_name : file_names)
    {
        if (file_statuses.at(file_name).status == FileStatus::REMOVED || file_statuses.at(file_name).status == FileStatus::BROKEN)
        {
            file_statuses.erase(file_name);
        }
        else
        {
            valid_files.push_back(file_name);
        }
    }

    file_names.swap(valid_files);

    return events.empty() || file_names.empty();
}

NamesAndTypesList StorageFileLog::getVirtuals() const
{
    return NamesAndTypesList{{"_file_name", std::make_shared<DataTypeString>()}, {"_offset", std::make_shared<DataTypeUInt64>()}};
}

Names StorageFileLog::getVirtualColumnNames()
{
    return {"_file_name", "_offset"};
}
}
