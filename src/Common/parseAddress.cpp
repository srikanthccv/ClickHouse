#include <Common/parseAddress.h>
#include <Common/Exception.h>
#include <IO/ReadHelpers.h>
#include <common/find_symbols.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

std::pair<std::string, UInt16> parseAddress(const std::string & str, UInt16 default_port)
{
    if (str.empty())
        throw Exception("Empty address passed to function parseAddress", ErrorCodes::BAD_ARGUMENTS);

    const char * begin = str.data();
    const char * end = begin + str.size();
    const char * port = end; // NOLINT

    if (begin[0] == '[')
    {
        const char * closing_square_bracket = find_first_symbols<']'>(begin + 1, end);
        if (closing_square_bracket >= end)
            throw Exception("Illegal address passed to function parseAddress: "
                "the address begins with opening square bracket, but no closing square bracket found", ErrorCodes::BAD_ARGUMENTS);

        port = closing_square_bracket + 1;
    }
    else
        port = find_first_symbols<':'>(begin, end);

    if (port != end)
    {
        if (*port != ':')
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                "Illegal port prefix passed to function parseAddress: {}", port);

        UInt16 port_number;
        if (!tryParse<UInt16>(port_number, port + 1))
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                "Illegal port passed to function parseAddress: {}", port + 1);
        }
        return { std::string(begin, port), port_number };
    }
    else if (default_port)
    {
        return { str, default_port };
    }
    else
        throw Exception("The address passed to function parseAddress doesn't contain port number "
            "and no 'default_port' was passed", ErrorCodes::BAD_ARGUMENTS);
}

}
