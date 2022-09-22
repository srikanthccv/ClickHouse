#include <Storages/MergeTree/TemporaryParts.h>
#include <Common/Exception.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

bool TemporaryParts::contains(const std::string & basename) const
{
    std::lock_guard lock(mutex);
    return parts.contains(basename);
}

bool TemporaryParts::add(const std::string & basename)
{
    std::lock_guard lock(mutex);
    bool inserted = parts.emplace(basename).second;
    return inserted;
}

void TemporaryParts::remove(const std::string & basename)
{
    std::lock_guard lock(mutex);
    bool removed = parts.erase(basename);
    if (!removed)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Temporary part {} does not exist", basename);
}

}
