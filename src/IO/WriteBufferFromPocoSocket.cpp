#include <Poco/Net/NetException.h>

#include <IO/WriteBufferFromPocoSocket.h>

#include <Common/Exception.h>
#include <Common/NetException.h>
#include <Common/Stopwatch.h>
#include <Common/MemoryTracker.h>
#include <Client/TimeoutSetter.h>


namespace ProfileEvents
{
    extern const Event NetworkSendElapsedMicroseconds;
}


namespace DB
{

namespace ErrorCodes
{
    extern const int NETWORK_ERROR;
    extern const int SOCKET_TIMEOUT;
    extern const int CANNOT_WRITE_TO_SOCKET;
}


void WriteBufferFromPocoSocket::nextImpl()
{
    if (!offset())
        return;

    Stopwatch watch;

    size_t bytes_written = 0;
    while (bytes_written < offset())
    {
        ssize_t res = 0;

        /// Add more details to exceptions.
        try
        {
            /// sendBytes in SecureStreamSocket throws TimeoutException after max(receive_timeout, send_timeout),
            /// but we want to get this exception exactly after send_timeout. So, set receive_timeout = send_timeout
            /// before sendBytes.
            std::unique_ptr<TimeoutSetter> timeout_setter = nullptr;
            if (socket.secure())
                timeout_setter = std::make_unique<TimeoutSetter>(dynamic_cast<Poco::Net::StreamSocket &>(socket), socket.getSendTimeout(), socket.getSendTimeout());

            res = socket.impl()->sendBytes(working_buffer.begin() + bytes_written, offset() - bytes_written);
        }
        catch (const Poco::Net::NetException & e)
        {
            throw NetException(e.displayText() + ", while writing to socket (" + peer_address.toString() + ")", ErrorCodes::NETWORK_ERROR);
        }
        catch (const Poco::TimeoutException &)
        {
            throw NetException("Timeout exceeded while writing to socket (" + peer_address.toString() + ")", ErrorCodes::SOCKET_TIMEOUT);
        }
        catch (const Poco::IOException & e)
        {
            throw NetException(e.displayText() + ", while writing to socket (" + peer_address.toString() + ")", ErrorCodes::NETWORK_ERROR);
        }

        if (res < 0)
            throw NetException("Cannot write to socket (" + peer_address.toString() + ")", ErrorCodes::CANNOT_WRITE_TO_SOCKET);

        bytes_written += res;
    }

    ProfileEvents::increment(ProfileEvents::NetworkSendElapsedMicroseconds, watch.elapsedMicroseconds());
}

WriteBufferFromPocoSocket::WriteBufferFromPocoSocket(Poco::Net::Socket & socket_, size_t buf_size)
    : BufferWithOwnMemory<WriteBuffer>(buf_size), socket(socket_), peer_address(socket.peerAddress())
{
}

WriteBufferFromPocoSocket::~WriteBufferFromPocoSocket()
{
    /// FIXME move final flush into the caller
    MemoryTracker::LockExceptionInThread lock;
    next();
}

}
