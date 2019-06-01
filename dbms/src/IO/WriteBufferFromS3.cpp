#include <IO/WriteBufferFromS3.h>

#include <common/logger_useful.h>


namespace DB
{

WriteBufferFromS3::WriteBufferFromS3(
    const Poco::URI & uri, const ConnectionTimeouts & timeouts, size_t buffer_size_)
    : WriteBufferFromOStream(buffer_size_)
    , session{makeHTTPSession(uri, timeouts)}
    , request{Poco::Net::HTTPRequest::HTTP_PUT, uri.getPathAndQuery(), Poco::Net::HTTPRequest::HTTP_1_1}
{
    request.setHost(uri.getHost());
    request.setChunkedTransferEncoding(true);
    request.setExpectContinue(true);

    LOG_TRACE((&Logger::get("WriteBufferFromS3")), "Sending request to " << uri.toString());

    ostr = &session->sendRequest(request);
}

void WriteBufferFromS3::finalize()
{
    receiveResponse(*session, request, response);
    /// TODO: Response body is ignored.
}

}
