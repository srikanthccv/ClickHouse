#pragma once

#include <Common/config.h>
#if USE_GPERF
// Definition of the class generated by gperf, present on gperf/tldLookup.gperf
class tldLookupHash
{
private:
    static inline unsigned int hash (const char *str, size_t len);
public:
    static const char *is_valid (const char *str, size_t len);
};

namespace DB
{
    using tldLookup = tldLookupHash;
}
#endif
