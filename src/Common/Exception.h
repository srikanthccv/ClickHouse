#pragma once

#include <cerrno>
#include <vector>
#include <memory>

#include <Poco/Version.h>
#include <Poco/Exception.h>

#include <base/defines.h>
#include <Common/StackTrace.h>
#include <Common/LoggingFormatStringHelpers.h>

#include <fmt/format.h>


namespace Poco { class Logger; }

/// Extract format string from a string literal and constructs consteval fmt::format_string
template <typename... Args>
struct FormatStringHelperImpl
{
    std::string_view message_format_string;
    fmt::format_string<Args...> fmt_str;
    template<typename T>
    consteval FormatStringHelperImpl(T && str) : message_format_string(tryGetStaticFormatString(str)), fmt_str(std::forward<T>(str)) {}
    template<typename T>
    FormatStringHelperImpl(fmt::basic_runtime<T> && str) : message_format_string(), fmt_str(std::forward<fmt::basic_runtime<T>>(str)) {}

    PreformattedMessage format(Args && ...args) const
    {
        return PreformattedMessage{fmt::format(fmt_str, std::forward<Args...>(args)...), message_format_string};
    }
};

template <typename... Args>
using FormatStringHelper = FormatStringHelperImpl<std::type_identity_t<Args>...>;

namespace DB
{

void abortOnFailedAssertion(const String & description);

class Exception : public Poco::Exception
{
public:
    using FramePointers = std::vector<void *>;

    Exception() = default;

    // used to remove the sensitive information from exceptions if query_masking_rules is configured
    struct MessageMasked
    {
        std::string msg;
        MessageMasked(const std::string & msg_);
        MessageMasked(std::string && msg_);
    };

    Exception(const MessageMasked & msg_masked, int code, bool remote_);
    Exception(MessageMasked && msg_masked, int code, bool remote_);

    // delegating constructor to mask sensitive information from the message
    Exception(const std::string & msg, int code, bool remote_ = false): Exception(MessageMasked(msg), code, remote_) {}
    Exception(std::string && msg, int code, bool remote_ = false): Exception(MessageMasked(std::move(msg)), code, remote_) {}
    Exception(PreformattedMessage && msg, int code): Exception(std::move(msg.message), code)
    {
        message_format_string = msg.format_string;
    }

    template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, String>>>
    Exception(int code, T && message)
        : Exception(message, code)
    {
        message_format_string = tryGetStaticFormatString(message);
    }

    template<> Exception(int code, const String & message) : Exception(message, code) {}
    template<> Exception(int code, String & message) : Exception(message, code) {}
    template<> Exception(int code, String && message) : Exception(std::move(message), code) {}

    // Format message with fmt::format, like the logging functions.
    template <typename... Args>
    Exception(int code, FormatStringHelper<Args...> fmt, Args &&... args)
        : Exception(fmt::format(fmt.fmt_str, std::forward<Args>(args)...), code)
    {
        message_format_string = fmt.message_format_string;
    }

    struct CreateFromPocoTag {};
    struct CreateFromSTDTag {};

    Exception(CreateFromPocoTag, const Poco::Exception & exc);
    Exception(CreateFromSTDTag, const std::exception & exc);

    Exception * clone() const override { return new Exception(*this); }
    void rethrow() const override { throw *this; }
    const char * name() const noexcept override { return "DB::Exception"; }
    const char * what() const noexcept override { return message().data(); }

    /// Add something to the existing message.
    template <typename... Args>
    void addMessage(fmt::format_string<Args...> format, Args &&... args)
    {
        addMessage(fmt::format(format, std::forward<Args>(args)...));
    }

    void addMessage(const std::string& message)
    {
        addMessage(MessageMasked(message));
    }

    void addMessage(const MessageMasked & msg_masked)
    {
        extendedMessage(msg_masked.msg);
    }

    /// Used to distinguish local exceptions from the one that was received from remote node.
    void setRemoteException(bool remote_ = true) { remote = remote_; }
    bool isRemoteException() const { return remote; }

    std::string getStackTraceString() const;
    /// Used for system.errors
    FramePointers getStackFramePointers() const;

    std::string_view tryGetMessageFormatString() const { return message_format_string; }

private:
#ifndef STD_EXCEPTION_HAS_STACK_TRACE
    StackTrace trace;
#endif
    bool remote = false;

    const char * className() const noexcept override { return "DB::Exception"; }

protected:
    std::string_view message_format_string;
};


std::string getExceptionStackTraceString(const std::exception & e);
std::string getExceptionStackTraceString(std::exception_ptr e);


/// Contains an additional member `saved_errno`. See the throwFromErrno function.
class ErrnoException : public Exception
{
public:
    ErrnoException(const std::string & msg, int code, int saved_errno_, const std::optional<std::string> & path_ = {})
        : Exception(msg, code), saved_errno(saved_errno_), path(path_) {}

    ErrnoException * clone() const override { return new ErrnoException(*this); }
    void rethrow() const override { throw *this; }

    int getErrno() const { return saved_errno; }
    std::optional<std::string> getPath() const { return path; }

private:
    int saved_errno;
    std::optional<std::string> path;

    const char * name() const noexcept override { return "DB::ErrnoException"; }
    const char * className() const noexcept override { return "DB::ErrnoException"; }
};


/// Special class of exceptions, used mostly in ParallelParsingInputFormat for
/// more convenient calculation of problem line number.
class ParsingException : public Exception
{
public:
    ParsingException();
    ParsingException(const std::string & msg, int code);
    ParsingException(int code, const std::string & message);
    ParsingException(int code, std::string && message) : Exception(message, code) {}

    // Format message with fmt::format, like the logging functions.
    template <typename... Args>
    ParsingException(int code, FormatStringHelper<Args...> fmt, Args &&... args) : Exception(fmt::format(fmt.fmt_str, std::forward<Args>(args)...), code)
    {
        message_format_string = fmt.message_format_string;
    }

    std::string displayText() const override;

    ssize_t getLineNumber() const { return line_number; }
    void setLineNumber(int line_number_) { line_number = line_number_;}

    String getFileName() const { return file_name; }
    void setFileName(const String & file_name_) { file_name = file_name_; }

    Exception * clone() const override { return new ParsingException(*this); }
    void rethrow() const override { throw *this; }

private:
    ssize_t line_number{-1};
    String file_name;
    mutable std::string formatted_message;

    const char * name() const noexcept override { return "DB::ParsingException"; }
    const char * className() const noexcept override { return "DB::ParsingException"; }
};


using Exceptions = std::vector<std::exception_ptr>;


[[noreturn]] void throwFromErrno(const std::string & s, int code, int the_errno = errno);
/// Useful to produce some extra information about available space and inodes on device
[[noreturn]] void throwFromErrnoWithPath(const std::string & s, const std::string & path, int code,
                                         int the_errno = errno);


/** Try to write an exception to the log (and forget about it).
  * Can be used in destructors in the catch-all block.
  */
void tryLogCurrentException(const char * log_name, const std::string & start_of_message = "");
void tryLogCurrentException(Poco::Logger * logger, const std::string & start_of_message = "");


/** Prints current exception in canonical format.
  * with_stacktrace - prints stack trace for DB::Exception.
  * check_embedded_stacktrace - if DB::Exception has embedded stacktrace then
  *  only this stack trace will be printed.
  * with_extra_info - add information about the filesystem in case of "No space left on device" and similar.
  */
std::string getCurrentExceptionMessage(bool with_stacktrace, bool check_embedded_stacktrace = false,
                                       bool with_extra_info = true);
PreformattedMessage getCurrentExceptionMessageAndPattern(bool with_stacktrace, bool check_embedded_stacktrace = false,
                                       bool with_extra_info = true);

/// Returns error code from ErrorCodes
int getCurrentExceptionCode();
int getExceptionErrorCode(std::exception_ptr e);

/// Returns string containing extra diagnostic info for specific exceptions (like "no space left on device" and "memory limit exceeded")
std::string getExtraExceptionInfo(const std::exception & e);

/// An execution status of any piece of code, contains return code and optional error
struct ExecutionStatus
{
    int code = 0;
    std::string message;

    ExecutionStatus() = default;

    explicit ExecutionStatus(int return_code, const std::string & exception_message = "")
    : code(return_code), message(exception_message) {}

    static ExecutionStatus fromCurrentException(const std::string & start_of_message = "");

    static ExecutionStatus fromText(const std::string & data);

    std::string serializeText() const;

    void deserializeText(const std::string & data);

    bool tryDeserializeText(const std::string & data);
};


void tryLogException(std::exception_ptr e, const char * log_name, const std::string & start_of_message = "");
void tryLogException(std::exception_ptr e, Poco::Logger * logger, const std::string & start_of_message = "");

std::string getExceptionMessage(const Exception & e, bool with_stacktrace, bool check_embedded_stacktrace = false);
PreformattedMessage getExceptionMessageAndPattern(const Exception & e, bool with_stacktrace, bool check_embedded_stacktrace = false);
std::string getExceptionMessage(std::exception_ptr e, bool with_stacktrace);


template <typename T>
requires std::is_pointer_v<T>
T exception_cast(std::exception_ptr e)
{
    try
    {
        std::rethrow_exception(e);
    }
    catch (std::remove_pointer_t<T> & concrete)
    {
        return &concrete;
    }
    catch (...)
    {
        return nullptr;
    }
}

}
