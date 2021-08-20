#include <Client/ClientBase.h>

#include <iostream>
#include <iomanip>
#include <filesystem>

#include <common/argsToConfig.h>
#include <common/DateLUT.h>
#include <common/LocalDate.h>
#include <common/LineReader.h>
#include <common/scope_guard_safe.h>

#if !defined(ARCADIA_BUILD)
#    include <Common/config_version.h>
#endif
#include <Common/UTF8Helpers.h>
#include <Common/TerminalSize.h>
#include <Common/clearPasswordFromCommandLine.h>
#include <Common/StringUtils/StringUtils.h>
#include <Common/filesystemHelpers.h>
#include <Common/Config/configReadClient.h>
#include <Common/InterruptListener.h>
#include <Common/NetException.h>
#include <Storages/ColumnsDescription.h>

#include <Client/ClientBaseHelpers.h>
#include <Client/TestHint.h>

#include <Parsers/parseQuery.h>
#include <Parsers/ParserQuery.h>
#include <Parsers/formatAST.h>
#include <Parsers/ASTInsertQuery.h>
#include <Parsers/ASTCreateQuery.h>
#include <Parsers/ASTDropQuery.h>
#include <Parsers/ASTSetQuery.h>
#include <Parsers/ASTUseQuery.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTQueryWithOutput.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTIdentifier.h>

#include <Formats/FormatFactory.h>
#include <Processors/Formats/IInputFormat.h>
#include <Processors/QueryPipeline.h>
#include <Processors/Executors/PullingAsyncPipelineExecutor.h>
#include <Processors/Transforms/AddingDefaultsTransform.h>
#include <Interpreters/ReplaceQueryParameterVisitor.h>
#include <IO/WriteBufferFromOStream.h>
#include <IO/UseSSL.h>
#include <IO/CompressionMethod.h>

#include <DataStreams/NullBlockOutputStream.h>
#include <DataStreams/InternalTextLogsRowOutputStream.h>

namespace fs = std::filesystem;


namespace DB
{

static const NameSet exit_strings{"exit", "quit", "logout", "учше", "йгше", "дщпщге", "exit;", "quit;", "logout;", "учшеж", "йгшеж", "дщпщгеж", "q", "й", "\\q", "\\Q", "\\й", "\\Й", ":q", "Жй"};

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int DEADLOCK_AVOIDED;
    extern const int CLIENT_OUTPUT_FORMAT_SPECIFIED;
    extern const int UNKNOWN_PACKET_FROM_SERVER;
    extern const int INVALID_USAGE_OF_INPUT;
    extern const int NO_DATA_TO_INSERT;
    extern const int UNEXPECTED_PACKET_FROM_SERVER;
}

}

namespace DB
{

ASTPtr ClientBase::parseQuery(const char *& pos, const char * end, bool allow_multi_statements) const
{
    ParserQuery parser(end);
    ASTPtr res;

    const auto & settings = global_context->getSettingsRef();
    size_t max_length = 0;

    if (!allow_multi_statements)
        max_length = settings.max_query_size;

    if (is_interactive || ignore_error)
    {
        String message;
        res = tryParseQuery(parser, pos, end, message, true, "", allow_multi_statements, max_length, settings.max_parser_depth);

        if (!res)
        {
            std::cerr << std::endl << message << std::endl << std::endl;
            return nullptr;
        }
    }
    else
    {
        res = parseQueryAndMovePosition(parser, pos, end, "", allow_multi_statements, max_length, settings.max_parser_depth);
    }

    if (is_interactive)
    {
        std::cout << std::endl;
        WriteBufferFromOStream res_buf(std::cout, 4096);
        formatAST(*res, res_buf);
        res_buf.next();
        std::cout << std::endl << std::endl;
    }

    return res;
}


// Consumes trailing semicolons and tries to consume the same-line trailing
// comment.
static void adjustQueryEnd(const char *& this_query_end, const char * all_queries_end, int max_parser_depth)
{
    // We have to skip the trailing semicolon that might be left
    // after VALUES parsing or just after a normal semicolon-terminated query.
    Tokens after_query_tokens(this_query_end, all_queries_end);
    IParser::Pos after_query_iterator(after_query_tokens, max_parser_depth);
    while (after_query_iterator.isValid() && after_query_iterator->type == TokenType::Semicolon)
    {
        this_query_end = after_query_iterator->end;
        ++after_query_iterator;
    }

    // Now we have to do some extra work to add the trailing
    // same-line comment to the query, but preserve the leading
    // comments of the next query. The trailing comment is important
    // because the test hints are usually written this way, e.g.:
    // select nonexistent_column; -- { serverError 12345 }.
    // The token iterator skips comments and whitespace, so we have
    // to find the newline in the string manually. If it's earlier
    // than the next significant token, it means that the text before
    // newline is some trailing whitespace or comment, and we should
    // add it to our query. There are also several special cases
    // that are described below.
    const auto * newline = find_first_symbols<'\n'>(this_query_end, all_queries_end);
    const char * next_query_begin = after_query_iterator->begin;

    // We include the entire line if the next query starts after
    // it. This is a generic case of trailing in-line comment.
    // The "equals" condition is for case of end of input (they both equal
    // all_queries_end);
    if (newline <= next_query_begin)
    {
        assert(newline >= this_query_end);
        this_query_end = newline;
    }
    else
    {
        // Many queries on one line, can't do anything. By the way, this
        // syntax is probably going to work as expected:
        // select nonexistent /* { serverError 12345 } */; select 1
    }
}


/// Convert external tables to ExternalTableData and send them using the connection.
void ClientBase::sendExternalTables(ASTPtr parsed_query)
{
    const auto * select = parsed_query->as<ASTSelectWithUnionQuery>();
    if (!select && !external_tables.empty())
        throw Exception("External tables could be sent only with select query", ErrorCodes::BAD_ARGUMENTS);

    std::vector<ExternalTableDataPtr> data;
    for (auto & table : external_tables)
        data.emplace_back(table.getData(global_context));

    connection->sendExternalTablesData(data);
}


void ClientBase::onData(Block & block, ASTPtr parsed_query)
{
    if (!block)
        return;

    processed_rows += block.rows();

    /// Even if all blocks are empty, we still need to initialize the output stream to write empty resultset.
    initBlockOutputStream(block, parsed_query);

    /// The header block containing zero rows was used to initialize
    /// block_out_stream, do not output it.
    /// Also do not output too much data if we're fuzzing.
    if (block.rows() == 0 || (query_fuzzer_runs != 0 && processed_rows >= 100))
        return;

    if (need_render_progress)
        progress_indication.clearProgressOutput();

    block_out_stream->write(block);
    written_first_block = true;

    /// Received data block is immediately displayed to the user.
    block_out_stream->flush();

    /// Restore progress bar after data block.
    if (need_render_progress)
        progress_indication.writeProgress();
}


void ClientBase::onLogData(Block & block)
{
    initLogsOutputStream();
    progress_indication.clearProgressOutput();
    logs_out_stream->write(block);
    logs_out_stream->flush();
}


void ClientBase::onTotals(Block & block, ASTPtr parsed_query)
{
    initBlockOutputStream(block, parsed_query);
    block_out_stream->setTotals(block);
}


void ClientBase::onExtremes(Block & block, ASTPtr parsed_query)
{
    initBlockOutputStream(block, parsed_query);
    block_out_stream->setExtremes(block);
}


void ClientBase::onReceiveExceptionFromServer(std::unique_ptr<Exception> && e)
{
    have_error = true;
    server_exception = std::move(e);
    resetOutput();
}


void ClientBase::onProfileInfo(const BlockStreamProfileInfo & profile_info)
{
    if (profile_info.hasAppliedLimit() && block_out_stream)
        block_out_stream->setRowsBeforeLimit(profile_info.getRowsBeforeLimit());
}


void ClientBase::initBlockOutputStream(const Block & block, ASTPtr parsed_query)
{
    if (!block_out_stream)
    {
        /// Ignore all results when fuzzing as they can be huge.
        if (query_fuzzer_runs)
        {
            block_out_stream = std::make_shared<NullBlockOutputStream>(block);
            return;
        }

        WriteBuffer * out_buf = nullptr;
        String pager = config().getString("pager", "");
        if (!pager.empty())
        {
            signal(SIGPIPE, SIG_IGN);
            pager_cmd = ShellCommand::execute(pager, true);
            out_buf = &pager_cmd->in;
        }
        else
        {
            out_buf = &std_out;
        }

        String current_format = format;

        /// The query can specify output format or output file.
        /// FIXME: try to prettify this cast using `as<>()`
        if (const auto * query_with_output = dynamic_cast<const ASTQueryWithOutput *>(parsed_query.get()))
        {
            if (query_with_output->out_file)
            {
                const auto & out_file_node = query_with_output->out_file->as<ASTLiteral &>();
                const auto & out_file = out_file_node.value.safeGet<std::string>();

                out_file_buf = wrapWriteBufferWithCompressionMethod(
                    std::make_unique<WriteBufferFromFile>(out_file, DBMS_DEFAULT_BUFFER_SIZE, O_WRONLY | O_EXCL | O_CREAT),
                    chooseCompressionMethod(out_file, ""),
                    /* compression level = */ 3
                );

                // We are writing to file, so default format is the same as in non-interactive mode.
                if (is_interactive && is_default_format)
                    current_format = "TabSeparated";
            }
            if (query_with_output->format != nullptr)
            {
                if (has_vertical_output_suffix)
                    throw Exception("Output format already specified", ErrorCodes::CLIENT_OUTPUT_FORMAT_SPECIFIED);
                const auto & id = query_with_output->format->as<ASTIdentifier &>();
                current_format = id.name();
            }
        }

        if (has_vertical_output_suffix)
            current_format = "Vertical";

        /// It is not clear how to write progress with parallel formatting. It may increase code complexity significantly.
        if (!need_render_progress)
            block_out_stream = global_context->getOutputStreamParallelIfPossible(current_format, out_file_buf ? *out_file_buf : *out_buf, block);
        else
            block_out_stream = global_context->getOutputStream(current_format, out_file_buf ? *out_file_buf : *out_buf, block);

        block_out_stream->writePrefix();
    }
}


void ClientBase::initLogsOutputStream()
{
    if (!logs_out_stream)
    {
        WriteBuffer * wb = out_logs_buf.get();

        if (!out_logs_buf)
        {
            if (server_logs_file.empty())
            {
                /// Use stderr by default
                out_logs_buf = std::make_unique<WriteBufferFromFileDescriptor>(STDERR_FILENO);
                wb = out_logs_buf.get();
            }
            else if (server_logs_file == "-")
            {
                /// Use stdout if --server_logs_file=- specified
                wb = &std_out;
            }
            else
            {
                out_logs_buf
                    = std::make_unique<WriteBufferFromFile>(server_logs_file, DBMS_DEFAULT_BUFFER_SIZE, O_WRONLY | O_APPEND | O_CREAT);
                wb = out_logs_buf.get();
            }
        }

        logs_out_stream = std::make_shared<InternalTextLogsRowOutputStream>(*wb, stdout_is_a_tty);
        logs_out_stream->writePrefix();
    }
}


void ClientBase::processTextAsSingleQuery(const String & full_query)
{
    /// Some parts of a query (result output and formatting) are executed
    /// client-side. Thus we need to parse the query.
    const char * begin = full_query.data();
    auto parsed_query = parseQuery(begin, begin + full_query.size(), false);

    if (!parsed_query)
        return;

    String query_to_execute;

    // An INSERT query may have the data that follow query text. Remove the
    /// Send part of query without data, because data will be sent separately.
    auto * insert = parsed_query->as<ASTInsertQuery>();
    if (insert && insert->data)
        query_to_execute = full_query.substr(0, insert->data - full_query.data());
    else
        query_to_execute = full_query;

    processParsedSingleQuery(full_query, query_to_execute, parsed_query);

    if (have_error)
        processError(full_query);
}


void ClientBase::processOrdinaryQuery(const String & query_to_execute, ASTPtr parsed_query)
{
    /// Rewrite query only when we have query parameters.
    /// Note that if query is rewritten, comments in query are lost.
    /// But the user often wants to see comments in server logs, query log, processlist, etc.
    auto query = query_to_execute;
    if (!query_parameters.empty())
    {
        /// Replace ASTQueryParameter with ASTLiteral for prepared statements.
        ReplaceQueryParameterVisitor visitor(query_parameters);
        visitor.visit(parsed_query);

        /// Get new query after substitutions. Note that it cannot be done for INSERT query with embedded data.
        query = serializeAST(*parsed_query);
    }

    int retries_left = 10;
    for (;;)
    {
        assert(retries_left > 0);

        try
        {
            connection->sendQuery(
                connection_parameters.timeouts,
                query,
                global_context->getCurrentQueryId(),
                query_processing_stage,
                &global_context->getSettingsRef(),
                &global_context->getClientInfo(),
                true);

            sendExternalTables(parsed_query);
            receiveResult(parsed_query);

            break;
        }
        catch (const Exception & e)
        {
            std::cerr << getCurrentExceptionMessage(true);
            /// Retry when the server said "Client should retry" and no rows
            /// has been received yet.
            if (processed_rows == 0 && e.code() == ErrorCodes::DEADLOCK_AVOIDED && --retries_left)
            {
                std::cerr << "Got a transient error from the server, will"
                        << " retry (" << retries_left << " retries left)";
            }
            else
            {
                throw;
            }
        }
    }
}


/// Receives and processes packets coming from server.
/// Also checks if query execution should be cancelled.
void ClientBase::receiveResult(ASTPtr parsed_query)
{
    InterruptListener interrupt_listener;
    bool cancelled = false;

    // TODO: get the poll_interval from commandline.
    const auto receive_timeout = connection_parameters.timeouts.receive_timeout;
    constexpr size_t default_poll_interval = 1000000; /// in microseconds
    constexpr size_t min_poll_interval = 5000; /// in microseconds
    const size_t poll_interval
        = std::max(min_poll_interval, std::min<size_t>(receive_timeout.totalMicroseconds(), default_poll_interval));

    while (true)
    {
        Stopwatch receive_watch(CLOCK_MONOTONIC_COARSE);

        while (true)
        {
            /// Has the Ctrl+C been pressed and thus the query should be cancelled?
            /// If this is the case, inform the server about it and receive the remaining packets
            /// to avoid losing sync.
            if (!cancelled)
            {
                auto cancel_query = [&] {
                    connection->sendCancel();
                    cancelled = true;
                    if (is_interactive)
                    {
                        progress_indication.clearProgressOutput();
                        std::cout << "Cancelling query." << std::endl;
                    }

                    /// Pressing Ctrl+C twice results in shut down.
                    interrupt_listener.unblock();
                };

                if (interrupt_listener.check())
                {
                    cancel_query();
                }
                else
                {
                    double elapsed = receive_watch.elapsedSeconds();
                    if (elapsed > receive_timeout.totalSeconds())
                    {
                        std::cout << "Timeout exceeded while receiving data from server."
                                    << " Waited for " << static_cast<size_t>(elapsed) << " seconds,"
                                    << " timeout is " << receive_timeout.totalSeconds() << " seconds." << std::endl;

                        cancel_query();
                    }
                }
            }

            /// Poll for changes after a cancellation check, otherwise it never reached
            /// because of progress updates from server.
            if (connection->poll(poll_interval))
                break;
        }

        if (!receiveAndProcessPacket(parsed_query, cancelled))
            break;
    }

    if (cancelled && is_interactive)
        std::cout << "Query was cancelled." << std::endl;
}


/// Receive a part of the result, or progress info or an exception and process it.
/// Returns true if one should continue receiving packets.
/// Output of result is suppressed if query was cancelled.
bool ClientBase::receiveAndProcessPacket(ASTPtr parsed_query, bool cancelled)
{
    Packet packet = connection->receivePacket();

    switch (packet.type)
    {
        case Protocol::Server::PartUUIDs:
            return true;

        case Protocol::Server::Data:
            if (!cancelled)
                onData(packet.block, parsed_query);
            return true;

        case Protocol::Server::Progress:
            onProgress(packet.progress);
            return true;

        case Protocol::Server::ProfileInfo:
            onProfileInfo(packet.profile_info);
            return true;

        case Protocol::Server::Totals:
            if (!cancelled)
                onTotals(packet.block, parsed_query);
            return true;

        case Protocol::Server::Extremes:
            if (!cancelled)
                onExtremes(packet.block, parsed_query);
            return true;

        case Protocol::Server::Exception:
            onReceiveExceptionFromServer(std::move(packet.exception));
            return false;

        case Protocol::Server::Log:
            onLogData(packet.block);
            return true;

        case Protocol::Server::EndOfStream:
            onEndOfStream();
            return false;

        default:
            throw Exception(
                ErrorCodes::UNKNOWN_PACKET_FROM_SERVER, "Unknown packet {} from server {}", packet.type, connection->getDescription());
    }
}


void ClientBase::onProgress(const Progress & value)
{
    if (!progress_indication.updateProgress(value))
    {
        // Just a keep-alive update.
        return;
    }

    if (block_out_stream)
        block_out_stream->onProgress(value);

    if (need_render_progress)
        progress_indication.writeProgress();
}


void ClientBase::onEndOfStream()
{
    progress_indication.clearProgressOutput();

    if (block_out_stream)
        block_out_stream->writeSuffix();

    if (logs_out_stream)
        logs_out_stream->writeSuffix();

    resetOutput();

    if (is_interactive && !written_first_block)
    {
        progress_indication.clearProgressOutput();
        std::cout << "Ok." << std::endl;
    }
}


/// Flush all buffers.
void ClientBase::resetOutput()
{
    block_out_stream.reset();
    logs_out_stream.reset();

    if (pager_cmd)
    {
        pager_cmd->in.close();
        pager_cmd->wait();
    }
    pager_cmd = nullptr;

    if (out_file_buf)
    {
        out_file_buf->next();
        out_file_buf.reset();
    }

    if (out_logs_buf)
    {
        out_logs_buf->next();
        out_logs_buf.reset();
    }

    std_out.next();
}

/// Receive the block that serves as an example of the structure of table where data will be inserted.
bool ClientBase::receiveSampleBlock(Block & out, ColumnsDescription & columns_description, ASTPtr parsed_query)
{
    while (true)
    {
        Packet packet = connection->receivePacket();

        switch (packet.type)
        {
            case Protocol::Server::Data:
                out = packet.block;
                return true;

            case Protocol::Server::Exception:
                onReceiveExceptionFromServer(std::move(packet.exception));
                return false;

            case Protocol::Server::Log:
                onLogData(packet.block);
                break;

            case Protocol::Server::TableColumns:
                columns_description = ColumnsDescription::parse(packet.multistring_message[1]);
                return receiveSampleBlock(out, columns_description, parsed_query);

            default:
                throw NetException(
                    "Unexpected packet from server (expected Data, Exception or Log, got "
                        + String(Protocol::Server::toString(packet.type)) + ")",
                    ErrorCodes::UNEXPECTED_PACKET_FROM_SERVER);
        }
    }
}


void ClientBase::processInsertQuery(const String & query_to_execute, ASTPtr parsed_query)
{
    /// Process the query that requires transferring data blocks to the server.
    const auto parsed_insert_query = parsed_query->as<ASTInsertQuery &>();
    if ((!parsed_insert_query.data && !parsed_insert_query.infile) && (is_interactive || (!stdin_is_a_tty && std_in.eof())))
        throw Exception("No data to insert", ErrorCodes::NO_DATA_TO_INSERT);

    connection->sendQuery(
        connection_parameters.timeouts,
        query_to_execute,
        global_context->getCurrentQueryId(),
        query_processing_stage,
        &global_context->getSettingsRef(),
        &global_context->getClientInfo(),
        true);

    sendExternalTables(parsed_query);

    /// Receive description of table structure.
    Block sample;
    ColumnsDescription columns_description;
    if (receiveSampleBlock(sample, columns_description, parsed_query))
    {
        /// If structure was received (thus, server has not thrown an exception),
        /// send our data with that structure.
        sendData(sample, columns_description, parsed_query);
        receiveEndOfQuery();
    }
}


void ClientBase::sendData(Block & sample, const ColumnsDescription & columns_description, ASTPtr parsed_query)
{
    /// If INSERT data must be sent.
    auto * parsed_insert_query = parsed_query->as<ASTInsertQuery>();
    if (!parsed_insert_query)
        return;

    if (parsed_insert_query->infile)
    {
        const auto & in_file_node = parsed_insert_query->infile->as<ASTLiteral &>();
        const auto in_file = in_file_node.value.safeGet<std::string>();

        auto in_buffer = wrapReadBufferWithCompressionMethod(std::make_unique<ReadBufferFromFile>(in_file), chooseCompressionMethod(in_file, ""));

        try
        {
            sendDataFrom(*in_buffer, sample, columns_description, parsed_query);
        }
        catch (Exception & e)
        {
            e.addMessage("data for INSERT was parsed from file");
            throw;
        }
    }
    else if (parsed_insert_query->data)
    {
        /// Send data contained in the query.
        ReadBufferFromMemory data_in(parsed_insert_query->data, parsed_insert_query->end - parsed_insert_query->data);
        try
        {
            sendDataFrom(data_in, sample, columns_description, parsed_query);
        }
        catch (Exception & e)
        {
            /// The following query will use data from input
            //      "INSERT INTO data FORMAT TSV\n " < data.csv
            //  And may be pretty hard to debug, so add information about data source to make it easier.
            e.addMessage("data for INSERT was parsed from query");
            throw;
        }
        // Remember where the data ended. We use this info later to determine
        // where the next query begins.
        parsed_insert_query->end = parsed_insert_query->data + data_in.count();
    }
    else if (!is_interactive)
    {
        /// Send data read from stdin.
        try
        {
            if (need_render_progress)
            {
                /// Set total_bytes_to_read for current fd.
                FileProgress file_progress(0, std_in.size());
                progress_indication.updateProgress(Progress(file_progress));

                /// Set callback to be called on file progress.
                progress_indication.setFileProgressCallback(global_context, true);

                /// Add callback to track reading from fd.
                std_in.setProgressCallback(global_context);
            }

            sendDataFrom(std_in, sample, columns_description, parsed_query);
        }
        catch (Exception & e)
        {
            e.addMessage("data for INSERT was parsed from stdin");
            throw;
        }
    }
    else
        throw Exception("No data to insert", ErrorCodes::NO_DATA_TO_INSERT);
}


void ClientBase::sendDataFrom(ReadBuffer & buf, Block & sample, const ColumnsDescription & columns_description, ASTPtr parsed_query)
{
    String current_format = insert_format;

    /// Data format can be specified in the INSERT query.
    if (const auto * insert = parsed_query->as<ASTInsertQuery>())
    {
        if (!insert->format.empty())
            current_format = insert->format;
    }

    auto source = FormatFactory::instance().getInput(current_format, buf, sample, global_context, insert_format_max_block_size);
    Pipe pipe(source);

    if (columns_description.hasDefaults())
    {
        pipe.addSimpleTransform([&](const Block & header)
        {
            return std::make_shared<AddingDefaultsTransform>(header, columns_description, *source, global_context);
        });
    }

    QueryPipeline pipeline;
    pipeline.init(std::move(pipe));
    PullingAsyncPipelineExecutor executor(pipeline);

    Block block;
    while (executor.pull(block))
    {
        /// Check if server send Log packet
        receiveLogs(parsed_query);

        /// Check if server send Exception packet
        auto packet_type = connection->checkPacket(/* timeout_milliseconds */0);
        if (packet_type && *packet_type == Protocol::Server::Exception)
        {
            /*
                * We're exiting with error, so it makes sense to kill the
                * input stream without waiting for it to complete.
                */
            executor.cancel();
            return;
        }

        if (block)
        {
            connection->sendData(block, /* name */"", /* scalar */false);
            processed_rows += block.rows();
        }
    }

    connection->sendData({}, "", false);
}


/// Process Log packets, used when inserting data by blocks
void ClientBase::receiveLogs(ASTPtr parsed_query)
{
    auto packet_type = connection->checkPacket(0);

    while (packet_type && *packet_type == Protocol::Server::Log)
    {
        receiveAndProcessPacket(parsed_query, false);
        packet_type = connection->checkPacket(/* timeout_milliseconds */0);
    }
}


/// Process Log packets, exit when receive Exception or EndOfStream
bool ClientBase::receiveEndOfQuery()
{
    while (true)
    {
        Packet packet = connection->receivePacket();

        switch (packet.type)
        {
            case Protocol::Server::EndOfStream:
                onEndOfStream();
                return true;

            case Protocol::Server::Exception:
                onReceiveExceptionFromServer(std::move(packet.exception));
                return false;

            case Protocol::Server::Log:
                onLogData(packet.block);
                break;

            default:
                throw NetException(
                    "Unexpected packet from server (expected Exception, EndOfStream or Log, got "
                        + String(Protocol::Server::toString(packet.type)) + ")",
                    ErrorCodes::UNEXPECTED_PACKET_FROM_SERVER);
        }
    }
}



void ClientBase::processParsedSingleQuery(const String & full_query, const String & query_to_execute,
        ASTPtr parsed_query, std::optional<bool> echo_query_, bool report_error)
{
    resetOutput();
    have_error = false;

    if (echo_query_ && *echo_query_)
    {
        writeString(full_query, std_out);
        writeChar('\n', std_out);
        std_out.next();
    }

    // if (is_interactive)
    // {
    //     global_context->setCurrentQueryId("");
    //     // Generate a new query_id
    //     for (const auto & query_id_format : query_id_formats)
    //     {
    //         writeString(query_id_format.first, std_out);
    //         writeString(fmt::format(query_id_format.second, fmt::arg("query_id", global_context->getCurrentQueryId())), std_out);
    //         writeChar('\n', std_out);
    //         std_out.next();
    //     }
    // }

    processed_rows = 0;
    written_first_block = false;
    progress_indication.resetProgress();

    executeSingleQuery(query_to_execute, parsed_query);

    if (is_interactive)
    {
        std::cout << std::endl << processed_rows << " rows in set. Elapsed: " << progress_indication.elapsedSeconds() << " sec. ";
        progress_indication.writeFinalProgress();
        std::cout << std::endl << std::endl;
    }
    else if (print_time_to_stderr)
    {
        std::cerr << progress_indication.elapsedSeconds() << "\n";
    }

    if (have_error && report_error)
        processError(full_query);
}


ClientBase::MultiQueryProcessingStage ClientBase::analyzeMultiQueryText(
    const char *& this_query_begin, const char *& this_query_end, const char * all_queries_end,
    String & query_to_execute, ASTPtr & parsed_query, const String & all_queries_text,
    std::optional<Exception> & current_exception)
{
    if (this_query_begin >= all_queries_end)
        return MultiQueryProcessingStage::QUERIES_END;

    // Remove leading empty newlines and other whitespace, because they
    // are annoying to filter in query log. This is mostly relevant for
    // the tests.
    while (this_query_begin < all_queries_end && isWhitespaceASCII(*this_query_begin))
        ++this_query_begin;

    if (this_query_begin >= all_queries_end)
        return MultiQueryProcessingStage::QUERIES_END;

    // If there are only comments left until the end of file, we just
    // stop. The parser can't handle this situation because it always
    // expects that there is some query that it can parse.
    // We can get into this situation because the parser also doesn't
    // skip the trailing comments after parsing a query. This is because
    // they may as well be the leading comments for the next query,
    // and it makes more sense to treat them as such.
    {
        Tokens tokens(this_query_begin, all_queries_end);
        IParser::Pos token_iterator(tokens, global_context->getSettingsRef().max_parser_depth);
        if (!token_iterator.isValid())
            return MultiQueryProcessingStage::QUERIES_END;
    }

    this_query_end = this_query_begin;
    try
    {
        parsed_query = parseQuery(this_query_end, all_queries_end, true);
    }
    catch (Exception & e)
    {
        current_exception.emplace(e);
        return MultiQueryProcessingStage::PARSING_EXCEPTION;
    }

    if (!parsed_query)
    {
        if (ignore_error)
        {
            Tokens tokens(this_query_begin, all_queries_end);
            IParser::Pos token_iterator(tokens, global_context->getSettingsRef().max_parser_depth);
            while (token_iterator->type != TokenType::Semicolon && token_iterator.isValid())
                ++token_iterator;
            this_query_begin = token_iterator->end;

            return MultiQueryProcessingStage::CONTINUE_PARSING;
        }

        return MultiQueryProcessingStage::PARSING_FAILED;
    }

    // INSERT queries may have the inserted data in the query text
    // that follow the query itself, e.g. "insert into t format CSV 1;2".
    // They need special handling. First of all, here we find where the
    // inserted data ends. In multy-query mode, it is delimited by a
    // newline.
    // The VALUES format needs even more handling -- we also allow the
    // data to be delimited by semicolon. This case is handled later by
    // the format parser itself.
    // We can't do multiline INSERTs with inline data, because most
    // row input formats (e.g. TSV) can't tell when the input stops,
    // unlike VALUES.
    auto * insert_ast = parsed_query->as<ASTInsertQuery>();
    if (insert_ast && insert_ast->data)
    {
        this_query_end = find_first_symbols<'\n'>(insert_ast->data, all_queries_end);
        insert_ast->end = this_query_end;
        query_to_execute = all_queries_text.substr(this_query_begin - all_queries_text.data(), insert_ast->data - this_query_begin);
    }
    else
    {
        query_to_execute = all_queries_text.substr(this_query_begin - all_queries_text.data(), this_query_end - this_query_begin);
    }

    // Try to include the trailing comment with test hints. It is just
    // a guess for now, because we don't yet know where the query ends
    // if it is an INSERT query with inline data. We will do it again
    // after we have processed the query. But even this guess is
    // beneficial so that we see proper trailing comments in "echo" and
    // server log.
    adjustQueryEnd(this_query_end, all_queries_end, global_context->getSettingsRef().max_parser_depth);
    return MultiQueryProcessingStage::EXECUTE_QUERY;
}


bool ClientBase::processQueryText(const String & text)
{
    if (exit_strings.end() != exit_strings.find(trim(text, [](char c) { return isWhitespaceASCII(c) || c == ';'; })))
        return false;

    if (!is_multiquery)
    {
        assert(!query_fuzzer_runs);
        processTextAsSingleQuery(text);

        return true;
    }

    if (query_fuzzer_runs)
    {
        processWithFuzzing(text);
        return true;
    }

    return processMultiQuery(text);
}

bool ClientBase::processMultiQuery(const String & all_queries_text)
{
    // It makes sense not to base any control flow on this, so that it is
    // the same in tests and in normal usage. The only difference is that in
    // normal mode we ignore the test hints.
    const bool test_mode = config().has("testmode");
    if (test_mode)
    {
        /// disable logs if expects errors
        TestHint test_hint(test_mode, all_queries_text);
        if (test_hint.clientError() || test_hint.serverError())
            processTextAsSingleQuery("SET send_logs_level = 'fatal'");
    }

    bool echo_query = echo_queries;

    /// Several queries separated by ';'.
    /// INSERT data is ended by the end of line, not ';'.
    /// An exception is VALUES format where we also support semicolon in
    /// addition to end of line.
    const char * this_query_begin = all_queries_text.data();
    const char * this_query_end;
    const char * all_queries_end = all_queries_text.data() + all_queries_text.size();

    String full_query; // full_query is the query + inline INSERT data + trailing comments (the latter is our best guess for now).
    String query_to_execute;
    ASTPtr parsed_query;

    std::optional<Exception> current_exception;
    while (true)
    {
        auto stage = analyzeMultiQueryText(this_query_begin, this_query_end, all_queries_end,
                                           query_to_execute, parsed_query, all_queries_text, current_exception);
        switch (stage)
        {
            case MultiQueryProcessingStage::QUERIES_END:
            case MultiQueryProcessingStage::PARSING_FAILED:
            {
                return true;
            }
            case MultiQueryProcessingStage::CONTINUE_PARSING:
            {
                continue;
            }
            case MultiQueryProcessingStage::PARSING_EXCEPTION:
            {
                this_query_end = find_first_symbols<'\n'>(this_query_end, all_queries_end);

                // Try to find test hint for syntax error. We don't know where
                // the query ends because we failed to parse it, so we consume
                // the entire line.
                TestHint hint(test_mode, String(this_query_begin, this_query_end - this_query_begin));
                if (hint.serverError())
                {
                    // Syntax errors are considered as client errors
                    current_exception->addMessage("\nExpected server error '{}'.", hint.serverError());
                    current_exception->rethrow();
                }

                if (hint.clientError() != current_exception->code())
                {
                    if (hint.clientError())
                        current_exception->addMessage("\nExpected client error: " + std::to_string(hint.clientError()));
                    current_exception->rethrow();
                }

                /// It's expected syntax error, skip the line
                this_query_begin = this_query_end;
                current_exception.reset();

                continue;
            }
            case MultiQueryProcessingStage::EXECUTE_QUERY:
            {
                full_query = all_queries_text.substr(this_query_begin - all_queries_text.data(), this_query_end - this_query_begin);
                if (query_fuzzer_runs)
                {
                    if (!processWithFuzzing(full_query))
                        return false;
                    this_query_begin = this_query_end;
                    continue;
                }

                // Now we know for sure where the query ends.
                // Look for the hint in the text of query + insert data + trailing
                // comments,
                // e.g. insert into t format CSV 'a' -- { serverError 123 }.
                // Use the updated query boundaries we just calculated.
                TestHint test_hint(test_mode, full_query);
                // Echo all queries if asked; makes for a more readable reference
                // file.
                echo_query = test_hint.echoQueries().value_or(echo_query);
                try
                {
                    processParsedSingleQuery(full_query, query_to_execute, parsed_query, echo_query, false);
                }
                catch (...)
                {
                    // Surprisingly, this is a client error. A server error would
                    // have been reported w/o throwing (see onReceiveSeverException()).
                    client_exception = std::make_unique<Exception>(getCurrentExceptionMessage(true), getCurrentExceptionCode());
                    have_error = true;
                }
                // Check whether the error (or its absence) matches the test hints
                // (or their absence).
                bool error_matches_hint = true;
                if (have_error)
                {
                    if (test_hint.serverError())
                    {
                        if (!server_exception)
                        {
                            error_matches_hint = false;
                            fmt::print(stderr, "Expected server error code '{}' but got no server error.\n", test_hint.serverError());
                        }
                        else if (server_exception->code() != test_hint.serverError())
                        {
                            error_matches_hint = false;
                            std::cerr << "Expected server error code: " << test_hint.serverError() << " but got: " << server_exception->code()
                                        << "." << std::endl;
                        }
                    }
                    if (test_hint.clientError())
                    {
                        if (!client_exception)
                        {
                            error_matches_hint = false;
                            fmt::print(stderr, "Expected client error code '{}' but got no client error.\n", test_hint.clientError());
                        }
                        else if (client_exception->code() != test_hint.clientError())
                        {
                            error_matches_hint = false;
                            fmt::print(
                                stderr, "Expected client error code '{}' but got '{}'.\n", test_hint.clientError(), client_exception->code());
                        }
                    }
                    if (!test_hint.clientError() && !test_hint.serverError())
                    {
                        // No error was expected but it still occurred. This is the
                        // default case w/o test hint, doesn't need additional
                        // diagnostics.
                        error_matches_hint = false;
                    }
                }
                else
                {
                    if (test_hint.clientError())
                    {
                        fmt::print(stderr, "The query succeeded but the client error '{}' was expected.\n", test_hint.clientError());
                        error_matches_hint = false;
                    }
                    if (test_hint.serverError())
                    {
                        fmt::print(stderr, "The query succeeded but the server error '{}' was expected.\n", test_hint.serverError());
                        error_matches_hint = false;
                    }
                }
                // If the error is expected, force reconnect and ignore it.
                if (have_error && error_matches_hint)
                {
                    client_exception.reset();
                    server_exception.reset();
                    have_error = false;

                    if (!connection->checkConnected())
                        connect();
                }

                // For INSERTs with inline data: use the end of inline data as
                // reported by the format parser (it is saved in sendData()).
                // This allows us to handle queries like:
                //   insert into t values (1); select 1
                // , where the inline data is delimited by semicolon and not by a
                // newline.
                auto * insert_ast = parsed_query->as<ASTInsertQuery>();
                if (insert_ast && insert_ast->data)
                {
                    this_query_end = insert_ast->end;
                    adjustQueryEnd(this_query_end, all_queries_end, global_context->getSettingsRef().max_parser_depth);
                }

                // Report error.
                if (have_error)
                    processError(full_query);

                // Stop processing queries if needed.
                if (have_error && !ignore_error)
                    return is_interactive;

                this_query_begin = this_query_end;
                break;
            }
        }
    }
}


void ClientBase::runInteractive()
{
    if (config().has("query_id"))
        throw Exception("query_id could be specified only in non-interactive mode", ErrorCodes::BAD_ARGUMENTS);
    if (print_time_to_stderr)
        throw Exception("time option could be specified only in non-interactive mode", ErrorCodes::BAD_ARGUMENTS);

    /// Initialize DateLUT here to avoid counting time spent here as query execution time.
    const auto local_tz = DateLUT::instance().getTimeZone();

    std::optional<Suggest> suggest;
    suggest.emplace();
    loadSuggestionData(*suggest);

    if (home_path.empty())
    {
        const char * home_path_cstr = getenv("HOME");
        if (home_path_cstr)
            home_path = home_path_cstr;
    }

    /// Initialize query_id_formats if any
    if (config().has("query_id_formats"))
    {
        Poco::Util::AbstractConfiguration::Keys keys;
        config().keys("query_id_formats", keys);
        for (const auto & name : keys)
            query_id_formats.emplace_back(name + ":", config().getString("query_id_formats." + name));
    }

    if (query_id_formats.empty())
        query_id_formats.emplace_back("Query id:", " {query_id}\n");

    /// Load command history if present.
    if (config().has("history_file"))
        history_file = config().getString("history_file");
    else
    {
        auto * history_file_from_env = getenv("CLICKHOUSE_HISTORY_FILE");
        if (history_file_from_env)
            history_file = history_file_from_env;
        else if (!home_path.empty())
            history_file = home_path + "/.clickhouse-client-history";
    }

    if (!history_file.empty() && !fs::exists(history_file))
    {
        /// Avoid TOCTOU issue.
        try
        {
            FS::createFile(history_file);
        }
        catch (const ErrnoException & e)
        {
            if (e.getErrno() != EEXIST)
                throw;
        }
    }

    LineReader::Patterns query_extenders = {"\\"};
    LineReader::Patterns query_delimiters = {";", "\\G"};

#if USE_REPLXX
    replxx::Replxx::highlighter_callback_t highlight_callback{};
    if (config().getBool("highlight", true))
        highlight_callback = highlight;

    ReplxxLineReader lr(*suggest, history_file, config().has("multiline"), query_extenders, query_delimiters, highlight_callback);

#elif defined(USE_READLINE) && USE_READLINE
    ReadlineLineReader lr(*suggest, history_file, config().has("multiline"), query_extenders, query_delimiters);
#else
    LineReader lr(history_file, config().has("multiline"), query_extenders, query_delimiters);
#endif

    /// Enable bracketed-paste-mode only when multiquery is enabled and multiline is
    ///  disabled, so that we are able to paste and execute multiline queries in a whole
    ///  instead of erroring out, while be less intrusive.
    if (config().has("multiquery") && !config().has("multiline"))
        lr.enableBracketedPaste();

    do
    {
        auto input = lr.readLine(prompt(), ":-] ");
        if (input.empty())
            break;

        has_vertical_output_suffix = false;
        if (input.ends_with("\\G"))
        {
            input.resize(input.size() - 2);
            has_vertical_output_suffix = true;
        }

        try
        {
            if (!processQueryText(input))
                break;
        }
        catch (const Exception & e)
        {
            /// We don't need to handle the test hints in the interactive mode.
            bool print_stack_trace = config().getBool("stacktrace", false);
            std::cerr << "Exception on client:" << std::endl << getExceptionMessage(e, print_stack_trace, true) << std::endl << std::endl;

            client_exception = std::make_unique<Exception>(e);
        }

        if (client_exception)
        {
            /// client_exception may have been set above or elsewhere.
            /// Client-side exception during query execution can result in the loss of
            /// sync in the connection protocol.
            /// So we reconnect and allow to enter the next query.
            if (!connection->checkConnected())
                connect();
        }
    }
    while (true);

    if (isNewYearMode())
        std::cout << "Happy new year." << std::endl;
    else if (isChineseNewYearMode(local_tz))
        std::cout << "Happy Chinese new year. 春节快乐!" << std::endl;
    else
        std::cout << "Bye." << std::endl;
}


void ClientBase::runNonInteractive()
{
    if (!queries_files.empty())
    {
        auto process_multi_query_from_file = [&](const String & file)
        {
            auto text = getQueryTextPrefix();
            String queries_from_file;

            ReadBufferFromFile in(file);
            readStringUntilEOF(queries_from_file, in);

            text += queries_from_file;
            return processMultiQuery(text);
        };

        /// Read all queries into `text`.
        for (const auto & queries_file : queries_files)
        {
            for (const auto & interleave_file : interleave_queries_files)
                if (!process_multi_query_from_file(interleave_file))
                    return;

            if (!process_multi_query_from_file(queries_file))
                return;
        }

        return;
    }

    String text;
    if (is_multiquery)
        text = getQueryTextPrefix();

    if (config().has("query"))
    {
        text += config().getRawString("query"); /// Poco configuration should not process substitutions in form of ${...} inside query.
    }
    else
    {
        /// If 'query' parameter is not set, read a query from stdin.
        /// The query is read entirely into memory (streaming is disabled).
        ReadBufferFromFileDescriptor in(STDIN_FILENO);
        readStringUntilEOF(text, in);
    }

    if (query_fuzzer_runs)
        processWithFuzzing(text);
    else
        processQueryText(text);
}


static void clearTerminal()
{
    /// Clear from cursor until end of screen.
    /// It is needed if garbage is left in terminal.
    /// Show cursor. It can be left hidden by invocation of previous programs.
    /// A test for this feature: perl -e 'print "x"x100000'; echo -ne '\033[0;0H\033[?25l'; clickhouse-client
    std::cout << "\033[0J"
                    "\033[?25h";
}


static void showClientVersion()
{
    std::cout << DBMS_NAME << " client version " << VERSION_STRING << VERSION_OFFICIAL << "." << std::endl;
}


int ClientBase::main(const std::vector<std::string> & /*args*/)
{
    UseSSL use_ssl;

    std::cout << std::fixed << std::setprecision(3);
    std::cerr << std::fixed << std::setprecision(3);

    if (is_interactive)
    {
        clearTerminal();
        showClientVersion();
    }

    return mainImpl();
}


void ClientBase::init(int argc, char ** argv)
{
    namespace po = boost::program_options;

    /// Don't parse options with Poco library, we prefer neat boost::program_options.
    stopOptionsProcessing();

    stdin_is_a_tty = isatty(STDIN_FILENO);
    stdout_is_a_tty = isatty(STDOUT_FILENO);
    terminal_width = getTerminalWidth();

    Arguments common_arguments{""}; /// 0th argument is ignored.
    std::vector<Arguments> external_tables_arguments;

    readArguments(argc, argv, common_arguments, external_tables_arguments);

    po::variables_map options;
    OptionsDescription options_description;
    addAndCheckOptions(options_description, options, common_arguments);
    po::notify(options);

    if (options.count("version") || options.count("V"))
    {
        showClientVersion();
        exit(0);
    }

    if (options.count("version-clean"))
    {
        std::cout << VERSION_STRING;
        exit(0);
    }

    /// Output of help message.
    if (options.count("help")
        || (options.count("host") && options["host"].as<std::string>() == "elp")) /// If user writes -help instead of --help.
    {
        printHelpMessage(options_description);
        exit(0);
    }

    if (options.count("log-level"))
        Poco::Logger::root().setLevel(options["log-level"].as<std::string>());

    processOptions(options_description, options, external_tables_arguments);
    argsToConfig(common_arguments, config(), 100);
    clearPasswordFromCommandLine(argc, argv);
}

}
