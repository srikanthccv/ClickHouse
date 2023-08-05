#include <Processors/Sources/ShellCommandSource.h>

#include <poll.h>

#include <Common/Epoll.h>
#include <Common/Stopwatch.h>
#include <Common/logger_useful.h>

#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>

#include <QueryPipeline/Pipe.h>
#include <Processors/ISimpleTransform.h>
#include <Processors/Formats/IOutputFormat.h>
#include <Processors/Executors/CompletedPipelineExecutor.h>
#include <Interpreters/Context.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int UNSUPPORTED_METHOD;
    extern const int TIMEOUT_EXCEEDED;
    extern const int CANNOT_READ_FROM_FILE_DESCRIPTOR;
    extern const int CANNOT_WRITE_TO_FILE_DESCRIPTOR;
    extern const int CANNOT_FCNTL;
    extern const int CANNOT_POLL;
}

static bool tryMakeFdNonBlocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (-1 == flags)
        return false;
    if (-1 == fcntl(fd, F_SETFL, flags | O_NONBLOCK))
        return false;

    return true;
}

static void makeFdNonBlocking(int fd)
{
    bool result = tryMakeFdNonBlocking(fd);
    if (!result)
        throwFromErrno("Cannot set non-blocking mode of pipe", ErrorCodes::CANNOT_FCNTL);
}

static bool tryMakeFdBlocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (-1 == flags)
        return false;

    if (-1 == fcntl(fd, F_SETFL, flags & (~O_NONBLOCK)))
        return false;

    return true;
}

static void makeFdBlocking(int fd)
{
    bool result = tryMakeFdBlocking(fd);
    if (!result)
        throwFromErrno("Cannot set blocking mode of pipe", ErrorCodes::CANNOT_FCNTL);
}

static bool pollFd(int fd, size_t timeout_milliseconds, int events)
{
    pollfd pfd;
    pfd.fd = fd;
    pfd.events = events;
    pfd.revents = 0;

    int res;

    while (true)
    {
        Stopwatch watch;
        res = poll(&pfd, 1, static_cast<int>(timeout_milliseconds));

        if (res < 0)
        {
            if (errno != EINTR)
                throwFromErrno("Cannot poll", ErrorCodes::CANNOT_POLL);

            const auto elapsed = watch.elapsedMilliseconds();
            if (timeout_milliseconds <= elapsed)
                break;
            timeout_milliseconds -= elapsed;
        }
        else
        {
            break;
        }
    }

    return res > 0;
}

class TimeoutReadBufferFromFileDescriptor : public BufferWithOwnMemory<ReadBuffer>
{
public:
    explicit TimeoutReadBufferFromFileDescriptor(
        int stdout_fd_,
        int stderr_fd_,
        size_t timeout_milliseconds_,
        ExternalCommandStderrReaction stderr_reaction_,
        ExternalCommandErrorExitReaction error_exit_reaction_)
        : stdout_fd(stdout_fd_)
        , stderr_fd(stderr_fd_)
        , timeout_milliseconds(timeout_milliseconds_)
        , stderr_reaction(stderr_reaction_)
        , error_exit_reaction(error_exit_reaction_)
    {
        makeFdNonBlocking(stdout_fd);
        makeFdNonBlocking(stderr_fd);

#if defined(OS_LINUX)
        epoll.add(stdout_fd);
        if (stderr_reaction != ExternalCommandStderrReaction::NONE || error_exit_reaction != ExternalCommandErrorExitReaction::NONE)
            epoll.add(stderr_fd);
#endif
    }

    bool nextImpl() override
    {
        size_t bytes_read = 0;

#if defined(OS_LINUX)
        static constexpr size_t BUFFER_SIZE = 4_KiB;

        while (!bytes_read)
        {
            epoll_event events[2];
            events[0].data.fd = events[1].data.fd = -1;
            size_t num_events = epoll.getManyReady(2, events, static_cast<int>(timeout_milliseconds));
            if (0 == num_events)
                throw Exception(ErrorCodes::TIMEOUT_EXCEEDED, "Pipe read timeout exceeded {} milliseconds", timeout_milliseconds);

            bool has_stdout = false;
            bool has_stderr = false;
            for (size_t i = 0; i < num_events; ++i)
            {
                if (events[i].data.fd == stdout_fd)
                    has_stdout = true;
                else if (events[i].data.fd == stderr_fd)
                    has_stderr = true;
            }

            if (has_stderr)
            {
                stderr_buf.resize(BUFFER_SIZE);
                ssize_t res = ::read(stderr_fd, stderr_buf.data(), stderr_buf.size());

                if (res > 0)
                {
                    stderr_buf.resize(res);
                    if (stderr_reaction == ExternalCommandStderrReaction::THROW)
                        throw Exception(ErrorCodes::UNSUPPORTED_METHOD, "Executable generates stderr: {}", stderr_buf);
                    else if (stderr_reaction == ExternalCommandStderrReaction::LOG)
                        LOG_WARNING(
                            &::Poco::Logger::get("TimeoutReadBufferFromFileDescriptor"), "Executable generates stderr: {}", stderr_buf);

                    if (error_exit_reaction == ExternalCommandErrorExitReaction::LOG_FIRST)
                    {
                        if (BUFFER_SIZE - error_exit_buf.size() < size_t(res))
                            res = BUFFER_SIZE - error_exit_buf.size();

                        if (res > 0)
                            error_exit_buf.append(stderr_buf.begin(), stderr_buf.begin() + res);
                    }
                    else if (error_exit_reaction == ExternalCommandErrorExitReaction::LOG_LAST)
                    {
                        if (res + error_exit_buf.size() > BUFFER_SIZE)
                        {
                            std::shift_left(error_exit_buf.begin(), error_exit_buf.end(), res + error_exit_buf.size() - BUFFER_SIZE);
                            error_exit_buf.resize(BUFFER_SIZE - res);
                        }

                        error_exit_buf += stderr_buf;
                    }
                }
            }

            if (has_stdout)
            {
                ssize_t res = ::read(stdout_fd, internal_buffer.begin(), internal_buffer.size());

                if (-1 == res && errno != EINTR)
                    throwFromErrno("Cannot read from pipe", ErrorCodes::CANNOT_READ_FROM_FILE_DESCRIPTOR);

                if (res == 0)
                    break;

                if (res > 0)
                    bytes_read += res;
            }
        }
#else
        while (!bytes_read)
        {
            if (!pollFd(stdout_fd, timeout_milliseconds, POLLIN))
                throw Exception(ErrorCodes::TIMEOUT_EXCEEDED, "Pipe read timeout exceeded {} milliseconds", timeout_milliseconds);

            ssize_t res = ::read(stdout_fd, internal_buffer.begin(), internal_buffer.size());

            if (-1 == res && errno != EINTR)
                throwFromErrno("Cannot read from pipe", ErrorCodes::CANNOT_READ_FROM_FILE_DESCRIPTOR);

            if (res == 0)
                break;

            if (res > 0)
                bytes_read += res;
        }
#endif

        if (bytes_read > 0)
        {
            working_buffer = internal_buffer;
            working_buffer.resize(bytes_read);
        }
        else
        {
            return false;
        }

        return true;
    }

    void reset() const
    {
        makeFdBlocking(stdout_fd);
        makeFdBlocking(stderr_fd);
    }

    ~TimeoutReadBufferFromFileDescriptor() override
    {
        tryMakeFdBlocking(stdout_fd);
        tryMakeFdBlocking(stderr_fd);
    }

    String error_exit_buf;

private:
    int stdout_fd;
    int stderr_fd;
    size_t timeout_milliseconds;
    [[maybe_unused]] ExternalCommandStderrReaction stderr_reaction;
    [[maybe_unused]] ExternalCommandErrorExitReaction error_exit_reaction;

#if defined(OS_LINUX)
    Epoll epoll;
    String stderr_buf;
#endif
};

class TimeoutWriteBufferFromFileDescriptor : public BufferWithOwnMemory<WriteBuffer>
{
public:
    explicit TimeoutWriteBufferFromFileDescriptor(int fd_, size_t timeout_milliseconds_)
        : fd(fd_), timeout_milliseconds(timeout_milliseconds_)
    {
        makeFdNonBlocking(fd);
    }

    void nextImpl() override
    {
        if (!offset())
            return;

        size_t bytes_written = 0;

        while (bytes_written != offset())
        {
            if (!pollFd(fd, timeout_milliseconds, POLLOUT))
                throw Exception(ErrorCodes::TIMEOUT_EXCEEDED, "Pipe write timeout exceeded {} milliseconds", timeout_milliseconds);

            ssize_t res = ::write(fd, working_buffer.begin() + bytes_written, offset() - bytes_written);

            if ((-1 == res || 0 == res) && errno != EINTR)
                throwFromErrno("Cannot write into pipe", ErrorCodes::CANNOT_WRITE_TO_FILE_DESCRIPTOR);

            if (res > 0)
                bytes_written += res;
        }
    }

    void reset() const
    {
        makeFdBlocking(fd);
    }

    ~TimeoutWriteBufferFromFileDescriptor() override
    {
        tryMakeFdBlocking(fd);
    }

private:
    int fd;
    size_t timeout_milliseconds;
};

class ShellCommandHolder
{
public:
    using ShellCommandBuilderFunc = std::function<std::unique_ptr<ShellCommand>()>;

    explicit ShellCommandHolder(ShellCommandBuilderFunc && func_)
        : func(std::move(func_))
    {}

    std::unique_ptr<ShellCommand> buildCommand()
    {
        if (returned_command)
            return std::move(returned_command);

        return func();
    }

    void returnCommand(std::unique_ptr<ShellCommand> command)
    {
        returned_command = std::move(command);
    }

private:
    std::unique_ptr<ShellCommand> returned_command;
    ShellCommandBuilderFunc func;
};

namespace
{
    /** A stream, that get child process and sends data using tasks in background threads.
    * For each send data task background thread is created. Send data task must send data to process input pipes.
    * ShellCommandPoolSource receives data from process stdout.
    *
    * If process_pool is passed in constructor then after source is destroyed process is returned to pool.
    */
    class ShellCommandSource final : public ISource
    {
    public:

        using SendDataTask = std::function<void(void)>;

        ShellCommandSource(
            ContextPtr context_,
            const std::string & format_,
            size_t command_read_timeout_milliseconds,
            ExternalCommandStderrReaction stderr_reaction,
            ExternalCommandErrorExitReaction error_exit_reaction,
            bool check_exit_code_,
            const Block & sample_block_,
            std::unique_ptr<ShellCommand> && command_,
            std::vector<SendDataTask> && send_data_tasks = {},
            const ShellCommandSourceConfiguration & configuration_ = {},
            std::unique_ptr<ShellCommandHolder> && command_holder_ = nullptr,
            std::shared_ptr<ProcessPool> process_pool_ = nullptr)
            : ISource(sample_block_)
            , context(context_)
            , format(format_)
            , sample_block(sample_block_)
            , command(std::move(command_))
            , configuration(configuration_)
            , timeout_command_out(
                  command->out.getFD(), command->err.getFD(), command_read_timeout_milliseconds, stderr_reaction, error_exit_reaction)
            , command_holder(std::move(command_holder_))
            , process_pool(process_pool_)
            , check_exit_code(check_exit_code_ || error_exit_reaction != ExternalCommandErrorExitReaction::NONE)
        {
            for (auto && send_data_task : send_data_tasks)
            {
                send_data_threads.emplace_back([task = std::move(send_data_task), this]()
                {
                    try
                    {
                        task();
                    }
                    catch (...)
                    {
                        std::lock_guard lock(send_data_lock);
                        exception_during_send_data = std::current_exception();
                    }
                });
            }

            size_t max_block_size = configuration.max_block_size;

            if (configuration.read_fixed_number_of_rows)
            {
                /** Currently parallel parsing input format cannot read exactly max_block_size rows from input,
                  * so it will be blocked on ReadBufferFromFileDescriptor because this file descriptor represent pipe that does not have eof.
                  */
                auto context_for_reading = Context::createCopy(context);
                context_for_reading->setSetting("input_format_parallel_parsing", false);
                context = context_for_reading;

                if (configuration.read_number_of_rows_from_process_output)
                {
                    /// Initialize executor in generate
                    return;
                }

                max_block_size = configuration.number_of_rows_to_read;
            }

            pipeline = QueryPipeline(Pipe(context->getInputFormat(format, timeout_command_out, sample_block, max_block_size)));
            executor = std::make_unique<PullingPipelineExecutor>(pipeline);
        }

        ~ShellCommandSource() override
        {
            for (auto & thread : send_data_threads)
                if (thread.joinable())
                    thread.join();

            if (command_is_invalid)
            {
                command = nullptr;
                if (!timeout_command_out.error_exit_buf.empty())
                    LOG_ERROR(
                        &::Poco::Logger::get("ShellCommandSource"), "Executable fails with stderr: {}", timeout_command_out.error_exit_buf);
            }

            if (command_holder && process_pool)
            {
                bool valid_command = configuration.read_fixed_number_of_rows && current_read_rows >= configuration.number_of_rows_to_read;

                if (command && valid_command)
                    command_holder->returnCommand(std::move(command));

                process_pool->returnObject(std::move(command_holder));
            }
        }

    protected:

        Chunk generate() override
        {
            rethrowExceptionDuringSendDataIfNeeded();

            Chunk chunk;

            try
            {
                if (configuration.read_fixed_number_of_rows)
                {
                    if (!executor && configuration.read_number_of_rows_from_process_output)
                    {
                        readText(configuration.number_of_rows_to_read, timeout_command_out);
                        char dummy;
                        readChar(dummy, timeout_command_out);

                        size_t max_block_size = configuration.number_of_rows_to_read;
                        pipeline = QueryPipeline(Pipe(context->getInputFormat(format, timeout_command_out, sample_block, max_block_size)));
                        executor = std::make_unique<PullingPipelineExecutor>(pipeline);
                    }

                    if (current_read_rows >= configuration.number_of_rows_to_read)
                        return {};
                }

                if (!executor->pull(chunk))
                {
                    if (check_exit_code)
                        command->wait();
                    return {};
                }

                current_read_rows += chunk.getNumRows();
            }
            catch (...)
            {
                command_is_invalid = true;
                throw;
            }

            return chunk;
        }

        Status prepare() override
        {
            auto status = ISource::prepare();

            if (status == Status::Finished)
            {
                for (auto & thread : send_data_threads)
                    if (thread.joinable())
                        thread.join();

                rethrowExceptionDuringSendDataIfNeeded();
            }

            return status;
        }

        String getName() const override { return "ShellCommandSource"; }

    private:

        void rethrowExceptionDuringSendDataIfNeeded()
        {
            std::lock_guard lock(send_data_lock);
            if (exception_during_send_data)
            {
                command_is_invalid = true;
                std::rethrow_exception(exception_during_send_data);
            }
        }

        ContextPtr context;
        std::string format;
        Block sample_block;

        std::unique_ptr<ShellCommand> command;
        ShellCommandSourceConfiguration configuration;

        TimeoutReadBufferFromFileDescriptor timeout_command_out;

        size_t current_read_rows = 0;

        ShellCommandHolderPtr command_holder;
        std::shared_ptr<ProcessPool> process_pool;

        bool check_exit_code = false;

        QueryPipeline pipeline;
        std::unique_ptr<PullingPipelineExecutor> executor;

        std::vector<ThreadFromGlobalPool> send_data_threads;

        std::mutex send_data_lock;
        std::exception_ptr exception_during_send_data;

        std::atomic<bool> command_is_invalid {false};
    };

    class SendingChunkHeaderTransform final : public ISimpleTransform
    {
    public:
        SendingChunkHeaderTransform(const Block & header, std::shared_ptr<TimeoutWriteBufferFromFileDescriptor> buffer_)
            : ISimpleTransform(header, header, false)
            , buffer(buffer_)
        {
        }

        String getName() const override { return "SendingChunkHeaderTransform"; }

    protected:

        void transform(Chunk & chunk) override
        {
            writeText(chunk.getNumRows(), *buffer);
            writeChar('\n', *buffer);
        }

    private:
        std::shared_ptr<TimeoutWriteBufferFromFileDescriptor> buffer;
    };

}

ShellCommandSourceCoordinator::ShellCommandSourceCoordinator(const Configuration & configuration_)
    : configuration(configuration_)
{
    if (configuration.is_executable_pool)
        process_pool = std::make_shared<ProcessPool>(configuration.pool_size ? configuration.pool_size : std::numeric_limits<size_t>::max());
}

Pipe ShellCommandSourceCoordinator::createPipe(
    const std::string & command,
    const std::vector<std::string> & arguments,
    std::vector<Pipe> && input_pipes,
    Block sample_block,
    ContextPtr context,
    const ShellCommandSourceConfiguration & source_configuration)
{
    ShellCommand::Config command_config(command);
    command_config.arguments = arguments;
    for (size_t i = 1; i < input_pipes.size(); ++i)
        command_config.write_fds.emplace_back(i + 2);

    std::unique_ptr<ShellCommand> process;
    std::unique_ptr<ShellCommandHolder> process_holder;

    auto destructor_strategy = ShellCommand::DestructorStrategy{true /*terminate_in_destructor*/, SIGTERM, configuration.command_termination_timeout_seconds};
    command_config.terminate_in_destructor_strategy = destructor_strategy;

    bool is_executable_pool = (process_pool != nullptr);
    if (is_executable_pool)
    {
        bool execute_direct = configuration.execute_direct;

        bool result = process_pool->tryBorrowObject(
            process_holder,
            [command_config, execute_direct]()
            {
                ShellCommandHolder::ShellCommandBuilderFunc func = [command_config, execute_direct]() mutable
                {
                    if (execute_direct)
                        return ShellCommand::executeDirect(command_config);
                    else
                        return ShellCommand::execute(command_config);
                };

                return std::make_unique<ShellCommandHolder>(std::move(func));
            },
            configuration.max_command_execution_time_seconds * 10000);

        if (!result)
            throw Exception(
                ErrorCodes::TIMEOUT_EXCEEDED,
                "Could not get process from pool, max command execution timeout exceeded {} seconds",
                configuration.max_command_execution_time_seconds);

        process = process_holder->buildCommand();
    }
    else
    {
        if (configuration.execute_direct)
            process = ShellCommand::executeDirect(command_config);
        else
            process = ShellCommand::execute(command_config);
    }

    std::vector<ShellCommandSource::SendDataTask> tasks;
    tasks.reserve(input_pipes.size());

    for (size_t i = 0; i < input_pipes.size(); ++i)
    {
        WriteBufferFromFile * write_buffer = nullptr;

        if (i == 0)
        {
            write_buffer = &process->in;
        }
        else
        {
            int descriptor = static_cast<int>(i) + 2;
            auto it = process->write_fds.find(descriptor);
            if (it == process->write_fds.end())
                throw Exception(ErrorCodes::UNSUPPORTED_METHOD, "Process does not contain descriptor to write {}", descriptor);

            write_buffer = &it->second;
        }

        int write_buffer_fd = write_buffer->getFD();
        auto timeout_write_buffer
            = std::make_shared<TimeoutWriteBufferFromFileDescriptor>(write_buffer_fd, configuration.command_write_timeout_milliseconds);

        input_pipes[i].resize(1);

        if (configuration.send_chunk_header)
        {
            auto transform = std::make_shared<SendingChunkHeaderTransform>(input_pipes[i].getHeader(), timeout_write_buffer);
            input_pipes[i].addTransform(std::move(transform));
        }

        auto pipeline = std::make_shared<QueryPipeline>(std::move(input_pipes[i]));
        auto out = context->getOutputFormat(configuration.format, *timeout_write_buffer, materializeBlock(pipeline->getHeader()));
        out->setAutoFlush();
        pipeline->complete(std::move(out));

        ShellCommandSource::SendDataTask task = [pipeline, timeout_write_buffer, write_buffer, is_executable_pool]()
        {
            CompletedPipelineExecutor executor(*pipeline);
            executor.execute();

            timeout_write_buffer->finalize();
            timeout_write_buffer->reset();

            if (!is_executable_pool)
            {
                write_buffer->close();
            }
        };

        tasks.emplace_back(std::move(task));
    }

    auto source = std::make_unique<ShellCommandSource>(
        context,
        configuration.format,
        configuration.command_read_timeout_milliseconds,
        configuration.stderr_reaction,
        configuration.error_exit_reaction,
        configuration.check_exit_code,
        std::move(sample_block),
        std::move(process),
        std::move(tasks),
        source_configuration,
        std::move(process_holder),
        process_pool);

    return Pipe(std::move(source));
}

}
