#include <Functions/FunctionBase58Conversion.h>
#include <Functions/FunctionFactory.h>

namespace DB
{
REGISTER_FUNCTION(Base58Encode)
{
    factory.registerFunction<FunctionBase58Conversion<Base58Encode>>();
}

REGISTER_FUNCTION(Base58Decode)
{
    factory.registerFunction<FunctionBase58Conversion<Base58Decode>>();
}
}
