#include <DataStreams/SizeLimits.h>
#include <Common/formatReadable.h>
#include <Common/Exception.h>
#include <string>


namespace DB
{

bool SizeLimits::check(UInt64 rows, UInt64 bytes, const char * what, int exception_code) const
{
    if (overflow_mode == OverflowMode::THROW)
    {
        if (max_rows && rows > max_rows)
            throw Exception("Limit for " + std::string(what) + " exceeded, max rows: " + formatReadableQuantity(max_rows)
                + ", current rows: " + formatReadableQuantity(rows), exception_code);

        if (max_bytes && bytes > max_bytes)
            throw Exception("Limit for " + std::string(what) + " exceeded, max bytes: " + formatReadableSizeWithBinarySuffix(max_bytes)
                + ", current bytes: " + formatReadableSizeWithBinarySuffix(bytes), exception_code);

        return true;
    }

    return softCheck(rows, bytes);
}

bool SizeLimits::softCheck(UInt64 rows, UInt64 bytes) const
{
    if (max_rows && rows > max_rows)
        return false;
    if (max_bytes && bytes > max_bytes)
        return false;
    return true;
}

}
