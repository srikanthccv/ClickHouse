#pragma once

#include <IO/BufferWithOwnMemory.h>
#include <IO/CompressionMethod.h>
#include <IO/HTTPCommon.h>
#include <IO/Progress.h>
#include <IO/WriteBuffer.h>
#include <IO/WriteBufferFromOStream.h>
#include <Server/HTTP/HTTPServerResponse.h>
#include <Common/NetException.h>
#include <Common/Stopwatch.h>

#include <mutex>
#include <optional>


namespace DB
{

/// The difference from WriteBufferFromOStream is that this buffer gets the underlying std::ostream
/// (using response.send()) only after data is flushed for the first time. This is needed in HTTP
/// servers to change some HTTP headers (e.g. response code) before any data is sent to the client
/// (headers can't be changed after response.send() is called).
///
/// In short, it allows delaying the call to response.send().
///
/// Additionally, supports HTTP response compression (in this case corresponding Content-Encoding
/// header will be set).
///
/// Also this class write and flush special X-ClickHouse-Progress HTTP headers
///  if no data was sent at the time of progress notification.
/// This allows to implement progress bar in HTTP clients.
class WriteBufferFromHTTPServerResponse final : public BufferWithOwnMemory<WriteBuffer>
{
public:
    WriteBufferFromHTTPServerResponse(
        HTTPServerResponse & response_,
        bool is_http_method_head_,
        size_t keep_alive_timeout_,
        bool compress_ = false,        /// If true - set Content-Encoding header and compress the result.
        CompressionMethod compression_method_ = CompressionMethod::None);

    ~WriteBufferFromHTTPServerResponse() override;

    /// Writes progress in repeating HTTP headers.
    void onProgress(const Progress & progress, Int64 peak_memory_usage_);

    /// Turn compression on or off.
    /// The setting has any effect only if HTTP headers haven't been sent yet.
    void setCompression(bool enable_compression)
    {
        compress = enable_compression;
    }

    /// Set compression level if the compression is turned on.
    /// The setting has any effect only if HTTP headers haven't been sent yet.
    void setCompressionLevel(int level)
    {
        compression_level = level;
    }

    /// Turn CORS on or off.
    /// The setting has any effect only if HTTP headers haven't been sent yet.
    void addHeaderCORS(bool enable_cors)
    {
        add_cors_header = enable_cors;
    }

    /// Send progress
    void setSendProgress(bool send_progress_) { send_progress = send_progress_; }

    /// Don't send HTTP headers with progress more frequently.
    void setSendProgressInterval(size_t send_progress_interval_ms_)
    {
        send_progress_interval_ms = send_progress_interval_ms_;
    }

    void setExceptionCode(int exception_code_) { exception_code = exception_code_; }

private:
    /// Send at least HTTP headers if no data has been sent yet.
    /// Use after the data has possibly been sent and no error happened (and thus you do not plan
    /// to change response HTTP code.
    /// This method is idempotent.
    void finalizeImpl() override;

    /// Must be called under locked mutex.
    /// This method send headers, if this was not done already,
    ///  but not finish them with \r\n, allowing to send more headers subsequently.
    void startSendHeaders();

    //  Used for write the header X-ClickHouse-Progress / X-ClickHouse-Summary
    void writeHeaderProgressImpl(const char * header_name);
    // Used for write the header X-ClickHouse-Progress
    void writeHeaderProgress();
    // Used for write the header X-ClickHouse-Summary
    void writeHeaderSummary();
    // Use to write the header X-ClickHouse-Exception-Code even when progress has been sent
    void writeExceptionCode();

    /// This method finish headers with \r\n, allowing to start to send body.
    void finishSendHeaders();

    void nextImpl() override;

    HTTPServerResponse & response;

    bool is_http_method_head;
    bool add_cors_header = false;
    size_t keep_alive_timeout = 0;
    bool compress = false;
    CompressionMethod compression_method;
    int compression_level = 1;

    std::shared_ptr<std::ostream> response_body_ostr;
    std::shared_ptr<std::ostream> response_header_ostr;

    std::unique_ptr<WriteBuffer> out;
    bool initialized = false;

    bool headers_started_sending = false;
    bool headers_finished_sending = false;    /// If true, you could not add any headers.

    Progress accumulated_progress;
    bool send_progress = false;
    size_t send_progress_interval_ms = 100;
    Stopwatch progress_watch;

    int exception_code = 0;

    Int64 peak_memory_usage = 0;

    std::mutex mutex;    /// progress callback could be called from different threads.
};

}
