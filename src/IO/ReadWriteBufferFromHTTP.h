#pragma once

#include <functional>
#include <base/types.h>
#include <base/sleep.h>
#include <IO/ConnectionTimeouts.h>
#include <IO/HTTPCommon.h>
#include <IO/ReadBuffer.h>
#include <IO/ReadBufferFromIStream.h>
#include <IO/ReadHelpers.h>
#include <IO/ReadSettings.h>
#include <Poco/Any.h>
#include <Poco/Net/HTTPBasicCredentials.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/URI.h>
#include <Poco/Version.h>
#include <Common/DNSResolver.h>
#include <Common/RemoteHostFilter.h>
#include <base/logger_useful.h>
#include <Poco/URIStreamFactory.h>

#include <Common/config.h>


namespace DB
{
/** Perform HTTP POST request and provide response to read.
  */

namespace ErrorCodes
{
    extern const int TOO_MANY_REDIRECTS;
    extern const int HTTP_RANGE_NOT_SATISFIABLE;
    extern const int BAD_ARGUMENTS;
}

template <typename SessionPtr>
class UpdatableSessionBase
{
protected:
    SessionPtr session;
    UInt64 redirects { 0 };
    Poco::URI initial_uri;
    const ConnectionTimeouts & timeouts;
    UInt64 max_redirects;

public:
    virtual void buildNewSession(const Poco::URI & uri) = 0;

    explicit UpdatableSessionBase(const Poco::URI uri,
        const ConnectionTimeouts & timeouts_,
        UInt64 max_redirects_)
        : initial_uri { uri }
        , timeouts { timeouts_ }
        , max_redirects { max_redirects_ }
    {
    }

    SessionPtr getSession()
    {
        return session;
    }

    void updateSession(const Poco::URI & uri)
    {
        ++redirects;
        if (redirects <= max_redirects)
        {
            buildNewSession(uri);
        }
        else
        {
            throw Exception(ErrorCodes::TOO_MANY_REDIRECTS, "Too many redirects while trying to access {}", initial_uri.toString());
        }
    }

    virtual ~UpdatableSessionBase() = default;
};


namespace detail
{
    template <typename UpdatableSessionPtr>
    class ReadWriteBufferFromHTTPBase : public ReadBuffer
    {
    public:
        using HTTPHeaderEntry = std::tuple<std::string, std::string>;
        using HTTPHeaderEntries = std::vector<HTTPHeaderEntry>;

    protected:
        Poco::URI uri;
        std::string method;
        std::string content_encoding;

        UpdatableSessionPtr session;
        std::istream * istr; /// owned by session
        std::unique_ptr<ReadBuffer> impl;
        std::function<void(std::ostream &)> out_stream_callback;
        const Poco::Net::HTTPBasicCredentials & credentials;
        std::vector<Poco::Net::HTTPCookie> cookies;
        HTTPHeaderEntries http_header_entries;
        RemoteHostFilter remote_host_filter;
        std::function<void(size_t)> next_callback;

        size_t buffer_size;
        bool use_external_buffer;

        size_t bytes_read = 0;
        /// Read from offset with range header if needed (for disk web).
        size_t start_byte = 0;
        /// Non-empty if content-length header was received.
        std::optional<size_t> total_bytes_to_read;

        /// Delayed exception in case retries with partial content are not satisfiable.
        std::exception_ptr exception;
        bool retry_with_range_header = false;

        ReadSettings settings;
        Poco::Logger * log;

        std::istream * call(Poco::URI uri_, Poco::Net::HTTPResponse & response)
        {
            // With empty path poco will send "POST  HTTP/1.1" its bug.
            if (uri_.getPath().empty())
                uri_.setPath("/");

            Poco::Net::HTTPRequest request(method, uri_.getPathAndQuery(), Poco::Net::HTTPRequest::HTTP_1_1);
            request.setHost(uri_.getHost()); // use original, not resolved host name in header

            if (out_stream_callback)
                request.setChunkedTransferEncoding(true);

            for (auto & http_header_entry: http_header_entries)
            {
                request.set(std::get<0>(http_header_entry), std::get<1>(http_header_entry));
            }

            /**
              * Add range header if we have start offset (for disk web)
              * or if we want to retry GET request on purpose.
              */
            bool with_partial_content = start_byte || retry_with_range_header;
            if (with_partial_content)
                request.set("Range", fmt::format("bytes={}-", start_byte + bytes_read));

            if (!credentials.getUsername().empty())
                credentials.authenticate(request);

            LOG_TRACE(log, "Sending request to {}", uri_.toString());

            auto sess = session->getSession();

            try
            {
                auto & stream_out = sess->sendRequest(request);

                if (out_stream_callback)
                    out_stream_callback(stream_out);

                istr = receiveResponse(*sess, request, response, true);
                response.getCookies(cookies);

                if (with_partial_content && response.getStatus() != Poco::Net::HTTPResponse::HTTPStatus::HTTP_PARTIAL_CONTENT)
                {
                    /// If we retried some request, throw error from that request.
                    if (exception)
                        std::rethrow_exception(exception);
                    throw Exception(ErrorCodes::HTTP_RANGE_NOT_SATISFIABLE, "Cannot read with range: {}", request.get("Range"));
                }

                content_encoding = response.get("Content-Encoding", "");
                return istr;
            }
            catch (const Poco::Exception & e)
            {
                /// We use session data storage as storage for exception text
                /// Depend on it we can deduce to reconnect session or reresolve session host
                sess->attachSessionData(e.message());
                throw;
            }
        }

    public:
        using NextCallback = std::function<void(size_t)>;
        using OutStreamCallback = std::function<void(std::ostream &)>;

        explicit ReadWriteBufferFromHTTPBase(
            UpdatableSessionPtr session_,
            Poco::URI uri_,
            const Poco::Net::HTTPBasicCredentials & credentials_,
            const std::string & method_ = {},
            OutStreamCallback out_stream_callback_ = {},
            size_t buffer_size_ = DBMS_DEFAULT_BUFFER_SIZE,
            const ReadSettings & settings_ = {},
            HTTPHeaderEntries http_header_entries_ = {},
            const RemoteHostFilter & remote_host_filter_ = {},
            bool delay_initialization = false,
            bool use_external_buffer_ = false)
            : ReadBuffer(nullptr, 0)
            , uri {uri_}
            , method {!method_.empty() ? method_ : out_stream_callback_ ? Poco::Net::HTTPRequest::HTTP_POST : Poco::Net::HTTPRequest::HTTP_GET}
            , session {session_}
            , out_stream_callback {out_stream_callback_}
            , credentials {credentials_}
            , http_header_entries {http_header_entries_}
            , remote_host_filter {remote_host_filter_}
            , use_external_buffer {use_external_buffer_}
            , buffer_size {buffer_size_}
            , start_byte {settings_.http_start_offset}
            , settings {settings_}
            , log(&Poco::Logger::get("ReadWriteBufferFromHTTP"))
        {
            if (settings.http_max_tries <= 0 || settings.http_retry_initial_backoff_ms <= 0
                || settings.http_retry_initial_backoff_ms >= settings.http_retry_max_backoff_ms)
                throw Exception(ErrorCodes::BAD_ARGUMENTS,
                                "Invalid setting for http backoff, "
                                "must be http_max_tries >= 1 (current is {}) and "
                                "0 < http_retry_initial_backoff_ms < settings.http_retry_max_backoff_ms (now 0 < {} < {})",
                                settings.http_max_tries, settings.http_retry_initial_backoff_ms, settings.http_retry_max_backoff_ms);

            if (!delay_initialization)
                initialize();
        }

        void initialize()
        {
            Poco::Net::HTTPResponse response;
            istr = call(uri, response);

            while (isRedirect(response.getStatus()))
            {
                Poco::URI uri_redirect(response.get("Location"));
                remote_host_filter.checkURL(uri_redirect);

                session->updateSession(uri_redirect);

                istr = call(uri_redirect, response);
            }

            /// If it is the very first initialization.
            if (!bytes_read && !total_bytes_to_read && response.hasContentLength())
                total_bytes_to_read = response.getContentLength();

            try
            {
                impl = std::make_unique<ReadBufferFromIStream>(*istr, buffer_size);

                if (use_external_buffer)
                {
                    /**
                    * See comment 30 lines below.
                    */
                    impl->set(internal_buffer.begin(), internal_buffer.size());
                    assert(working_buffer.begin() != nullptr);
                    assert(!internal_buffer.empty());
                }
            }
            catch (const Poco::Exception & e)
            {
                /// We use session data storage as storage for exception text
                /// Depend on it we can deduce to reconnect session or reresolve session host
                auto sess = session->getSession();
                sess->attachSessionData(e.message());
                throw;
            }
        }

        bool nextImpl() override
        {
            if (next_callback)
                next_callback(count());

            if (total_bytes_to_read && bytes_read == total_bytes_to_read.value())
                return false;

            if (impl)
            {
                if (use_external_buffer)
                {
                    /**
                    * use_external_buffer -- means we read into the buffer which
                    * was passed to us from somewhere else. We do not check whether
                    * previously returned buffer was read or not (no hasPendingData() check is needed),
                    * because this branch means we are prefetching data,
                    * each nextImpl() call we can fill a different buffer.
                    */
                    impl->set(internal_buffer.begin(), internal_buffer.size());
                    assert(working_buffer.begin() != nullptr);
                    assert(!internal_buffer.empty());
                }
                else
                {
                    /**
                    * impl was initialized before, pass position() to it to make
                    * sure there is no pending data which was not read.
                    */
                    if (!working_buffer.empty())
                        impl->position() = position();
                }
            }

            bool result = false;
            bool successful_read = false;
            size_t milliseconds_to_wait = settings.http_retry_initial_backoff_ms;

            /// Default http_max_tries = 1.
            for (size_t i = 0; i < settings.http_max_tries; ++i)
            {
                try
                {
                    if (!impl)
                    {
                        initialize();

                        if (use_external_buffer)
                        {
                            /// See comment 40 lines above.
                            impl->set(internal_buffer.begin(), internal_buffer.size());
                            assert(working_buffer.begin() != nullptr);
                            assert(!internal_buffer.empty());
                        }
                    }

                    result = impl->next();
                    successful_read = true;
                    break;
                }
                catch (const Poco::Exception & e)
                {
                    /**
                     * Retry request unconditionally if nothing has beed read yet.
                     * Otherwise if it is GET method retry with range header starting from bytes_read.
                     */
                    bool can_retry_request = !bytes_read || method == Poco::Net::HTTPRequest::HTTP_GET;
                    if (!can_retry_request)
                        throw;

                    /**
                     * if total_size is not known, last read can fail if we retry with
                     * bytes_read == total_size and header `bytes=bytes_read-`
                     * (we will get an error code 416 - range not satisfiable).
                     * In this case rethrow previous exception.
                     */
                    if (exception && !total_bytes_to_read.has_value() && e.code() == 416)
                        std::rethrow_exception(exception);

                    LOG_ERROR(log,
                              "HTTP request to `{}` failed at try {}/{} with bytes read: {}. "
                              "Error: {}, code: {}. (Current backoff wait is {}/{} ms)",
                              uri.toString(), i, settings.http_max_tries, bytes_read, e.what(), e.code(),
                              milliseconds_to_wait, settings.http_retry_max_backoff_ms);

                    retry_with_range_header = true;
                    exception = std::current_exception();
                    impl.reset();
                    sleepForMilliseconds(milliseconds_to_wait);
                    milliseconds_to_wait *= 2;
                }

                if (successful_read || milliseconds_to_wait >= settings.http_retry_max_backoff_ms)
                    break;
            }

            if (!successful_read && exception)
                std::rethrow_exception(exception);

            if (!result)
                return false;

            internal_buffer = impl->buffer();
            working_buffer = internal_buffer;
            return true;
        }

        std::string getResponseCookie(const std::string & name, const std::string & def) const
        {
            for (const auto & cookie : cookies)
                if (cookie.getName() == name)
                    return cookie.getValue();
            return def;
        }

        /// Set function to call on each nextImpl, useful when you need to track
        /// progress.
        /// NOTE: parameter on each call is not incremental -- it's all bytes count
        /// passed through the buffer
        void setNextCallback(NextCallback next_callback_)
        {
            next_callback = next_callback_;
            /// Some data maybe already read
            next_callback(count());
        }

        const std::string & getCompressionMethod() const
        {
            return content_encoding;
        }
    };
}

class UpdatableSession : public UpdatableSessionBase<HTTPSessionPtr>
{
    using Parent = UpdatableSessionBase<HTTPSessionPtr>;

public:
    explicit UpdatableSession(
        const Poco::URI uri,
        const ConnectionTimeouts & timeouts_,
        const UInt64 max_redirects_)
        : Parent(uri, timeouts_, max_redirects_)
    {
        session = makeHTTPSession(initial_uri, timeouts);
    }

    void buildNewSession(const Poco::URI & uri) override
    {
        session = makeHTTPSession(uri, timeouts);
    }
};

class ReadWriteBufferFromHTTP : public detail::ReadWriteBufferFromHTTPBase<std::shared_ptr<UpdatableSession>>
{
    using Parent = detail::ReadWriteBufferFromHTTPBase<std::shared_ptr<UpdatableSession>>;

public:
    explicit ReadWriteBufferFromHTTP(
        Poco::URI uri_,
        const std::string & method_,
        OutStreamCallback out_stream_callback_,
        const ConnectionTimeouts & timeouts,
        const Poco::Net::HTTPBasicCredentials & credentials_,
        const UInt64 max_redirects = 0,
        size_t buffer_size_ = DBMS_DEFAULT_BUFFER_SIZE,
        const ReadSettings & settings_ = {},
        const HTTPHeaderEntries & http_header_entries_ = {},
        const RemoteHostFilter & remote_host_filter_ = {},
<<<<<<< HEAD
        bool delay_initialization_ = true)
        : Parent(std::make_shared<UpdatableSession>(uri_, timeouts, max_redirects),
            uri_, credentials_, method_, out_stream_callback_, buffer_size_, settings_, http_header_entries_, remote_host_filter_, delay_initialization_)
=======
        bool use_external_buffer_ = false)
        : Parent(std::make_shared<UpdatableSession>(uri_, timeouts, max_redirects),
                 uri_, method_, out_stream_callback_, credentials_, buffer_size_,
                 settings_, http_header_entries_, remote_host_filter_, use_external_buffer_)
>>>>>>> 11b70a285c05fff189debb2705cb60d896fadb9c
    {
    }
};

class UpdatablePooledSession : public UpdatableSessionBase<PooledHTTPSessionPtr>
{
    using Parent = UpdatableSessionBase<PooledHTTPSessionPtr>;

private:
    size_t per_endpoint_pool_size;

public:
    explicit UpdatablePooledSession(const Poco::URI uri,
        const ConnectionTimeouts & timeouts_,
        const UInt64 max_redirects_,
        size_t per_endpoint_pool_size_)
        : Parent(uri, timeouts_, max_redirects_)
        , per_endpoint_pool_size { per_endpoint_pool_size_ }
    {
        session = makePooledHTTPSession(initial_uri, timeouts, per_endpoint_pool_size);
    }

    void buildNewSession(const Poco::URI & uri) override
    {
       session = makePooledHTTPSession(uri, timeouts, per_endpoint_pool_size);
    }
};

class PooledReadWriteBufferFromHTTP : public detail::ReadWriteBufferFromHTTPBase<std::shared_ptr<UpdatablePooledSession>>
{
    using Parent = detail::ReadWriteBufferFromHTTPBase<std::shared_ptr<UpdatablePooledSession>>;

public:
    explicit PooledReadWriteBufferFromHTTP(Poco::URI uri_,
        const std::string & method_ = {},
        OutStreamCallback out_stream_callback_ = {},
        const ConnectionTimeouts & timeouts_ = {},
        const Poco::Net::HTTPBasicCredentials & credentials_ = {},
        size_t buffer_size_ = DBMS_DEFAULT_BUFFER_SIZE,
        const UInt64 max_redirects = 0,
        size_t max_connections_per_endpoint = DEFAULT_COUNT_OF_HTTP_CONNECTIONS_PER_ENDPOINT)
        : Parent(std::make_shared<UpdatablePooledSession>(uri_, timeouts_, max_redirects, max_connections_per_endpoint),
              uri_,
              credentials_,
              method_,
              out_stream_callback_,
              buffer_size_)
    {
    }
};

}
