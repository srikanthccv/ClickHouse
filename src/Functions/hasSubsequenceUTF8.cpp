#include <Functions/FunctionFactory.h>
#include <Functions/FunctionsStringSearch.h>
#include <Functions/HasSubsequenceImpl.h>


namespace DB
{
namespace
{

struct HasSubsequenceCaseSensitiveUTF8
{
    static void toLowerIfNeed(String & /*s*/) { }
};

struct NameHasSubsequenceUTF8
{
    static constexpr auto name = "hasSubsequenceUTF8";
};

using FunctionHasSubsequenceUTF8 = FunctionsStringSearch<HasSubsequenceImpl<NameHasSubsequenceUTF8, HasSubsequenceCaseSensitiveUTF8>>;
}

REGISTER_FUNCTION(hasSubsequenceUTF8)
{
    factory.registerFunction<FunctionHasSubsequenceUTF8>({}, FunctionFactory::CaseInsensitive);
}

}
