#include <Common/ClickHouseRevision.h>
#include <Common/config_version.h>

namespace ClickHouseRevision
{
    unsigned get() { return VERSION_REVISION; }
    unsigned getVersionInt() { return VERSION_INT; }
}
