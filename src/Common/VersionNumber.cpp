#include <Common/VersionNumber.h>
#include <Common/Exception.h>
#include <cstdlib>
#include <iostream>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

VersionNumber::VersionNumber(const std::vector<long> & vec)
{
    if (vec.size() > SIZE)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Too much components ({})", vec.size());

    if (vec.size() > 0)
        std::get<0>(version) = vec[0];
    if (vec.size() > 1)
        std::get<1>(version) = vec[1];
    if (vec.size() > 2)
        std::get<2>(version) = vec[2];
}

std::string VersionNumber::toString() const
{
    return fmt::format("{}.{}.{}",
        std::get<0>(version), std::get<1>(version), std::get<2>(version));
}

VersionNumber VersionNumber::fromString(std::string version, bool strict)
{
    if (version.empty())
        return VersionNumber{};

    std::vector<long> comp;

    char * start = &version.front();
    char * end = start;
    const char * eos = &version.back() + 1;

    do
    {
        long value = strtol(start, &end, 10);
        comp.push_back(value);
        start = end + 1;
    }
    while (start < eos && (end < eos && *end == '.'));

    if (!strict && comp.size() > SIZE)
    {
        comp.resize(SIZE);
    }

    return VersionNumber(std::move(comp));
}


}
