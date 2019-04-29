#include <Functions/FunctionFactory.h>
#include <Functions/FunctionStringToString.h>
#include <Functions/FunctionsURL.h>
#include <common/find_symbols.h>

namespace DB
{

struct ExtractBasename
{
    static size_t getReserveLengthForElement() { return 25; }

    static void execute(Pos data, size_t size, Pos & res_data, size_t & res_size)
    {
        res_data = data;
        res_size = size;

        Pos pos = data;
        Pos end = pos + size;

        if ((pos = find_last_symbols_or_null<'/', '\\'>(pos, end)))
        {
            ++pos;
            res_data = pos;
            res_size = end - pos;
        }
    }
};

struct NameBasename { static constexpr auto name = "basename"; };
using FunctionBasename = FunctionStringToString<ExtractSubstringImpl<ExtractBasename>, NameBasename>;

void registerFunctionBasename(FunctionFactory & factory)
{
    factory.registerFunction<FunctionBasename>();
}

}
