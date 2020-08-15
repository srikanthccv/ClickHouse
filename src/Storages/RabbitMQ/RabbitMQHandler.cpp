#include <common/logger_useful.h>
#include <Common/Exception.h>
#include <Storages/RabbitMQ/RabbitMQHandler.h>

namespace DB
{

/* The object of this class is shared between concurrent consumers (who share the same connection == share the same
 * event loop and handler).
 */
RabbitMQHandler::RabbitMQHandler(uv_loop_t * loop_, Poco::Logger * log_) :
    AMQP::LibUvHandler(loop_),
    loop(loop_),
    log(log_),
    connection_running(false),
    loop_state(Loop::STOP)
{
}

///Method that is called when the connection ends up in an error state.
void RabbitMQHandler::onError(AMQP::TcpConnection * connection, const char * message)
{
    LOG_ERROR(log, "Library error report: {}", message);
    connection_running.store(false);
    if (connection)
        connection->close();
}

void RabbitMQHandler::onReady(AMQP::TcpConnection * /* connection */)
{
    LOG_TRACE(log, "Connection is ready");
    connection_running.store(true);
    loop_state.store(Loop::RUN);
}

void RabbitMQHandler::startLoop()
{
    std::lock_guard lock(startup_mutex);
    while (loop_state.load() == Loop::RUN)
        uv_run(loop, UV_RUN_NOWAIT);
}

void RabbitMQHandler::iterateLoop()
{
    std::unique_lock lock(startup_mutex, std::defer_lock);
    if (lock.try_lock())
        uv_run(loop, UV_RUN_NOWAIT);
}

}
