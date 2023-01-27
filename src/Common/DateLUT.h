#pragma once

#include "DateLUTImpl.h"

#include <base/defines.h>

#include <boost/noncopyable.hpp>
//
//#include "Interpreters/Context_fwd.h"
//#include "Interpreters/Context.h"
#include "Common/CurrentThread.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>


/// This class provides lazy initialization and lookup of singleton DateLUTImpl objects for a given timezone.
class DateLUT : private boost::noncopyable
{
public:
    /// Return singleton DateLUTImpl instance for the default time zone.
    static ALWAYS_INLINE const DateLUTImpl & instance()  // -V1071
    {
        const auto & date_lut = getInstance();

        if (DB::CurrentThread::isInitialized())
        {
            const auto query_context = DB::CurrentThread::get().getQueryContext();

            if (query_context)
            {
                auto implicit_timezone = extractTimezoneFromContext(query_context);

                if (!implicit_timezone.empty())
                    return instance(implicit_timezone);
            }
        }

        return *date_lut.default_impl.load(std::memory_order_acquire);
    }

    /// Return singleton DateLUTImpl instance for a given time zone.
    static ALWAYS_INLINE const DateLUTImpl & instance(const std::string & time_zone)
    {
        if (time_zone.empty())
            return instance();

        const auto & date_lut = getInstance();
        return date_lut.getImplementation(time_zone);
    }
    static void setDefaultTimezone(const std::string & time_zone)
    {
        auto & date_lut = getInstance();
        const auto & impl = date_lut.getImplementation(time_zone);
        date_lut.default_impl.store(&impl, std::memory_order_release);
    }

protected:
    DateLUT();

private:
    static DateLUT & getInstance();

    static std::string extractTimezoneFromContext(const DB::ContextPtr query_context);

    const DateLUTImpl & getImplementation(const std::string & time_zone) const;

    using DateLUTImplPtr = std::unique_ptr<DateLUTImpl>;

    /// Time zone name -> implementation.
    mutable std::unordered_map<std::string, DateLUTImplPtr> impls;
    mutable std::mutex mutex;

    std::atomic<const DateLUTImpl *> default_impl;
};

inline UInt64 timeInMilliseconds(std::chrono::time_point<std::chrono::system_clock> timepoint)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(timepoint.time_since_epoch()).count();
}

inline UInt64 timeInMicroseconds(std::chrono::time_point<std::chrono::system_clock> timepoint)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(timepoint.time_since_epoch()).count();
}

inline UInt64 timeInSeconds(std::chrono::time_point<std::chrono::system_clock> timepoint)
{
    return std::chrono::duration_cast<std::chrono::seconds>(timepoint.time_since_epoch()).count();
}

inline UInt64 timeInNanoseconds(std::chrono::time_point<std::chrono::system_clock> timepoint)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(timepoint.time_since_epoch()).count();
}
