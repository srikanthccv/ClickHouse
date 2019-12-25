#include <Functions/FunctionBase64Conversion.h>
#if USE_BASE64
#include <Functions/FunctionFactory.h>
#include <DataTypes/DataTypeString.h>
#include "registerFunctions.h"

namespace DB
{
void registerFunctionBase64Decode(FunctionFactory & factory)
{
    tb64ini(0);
    factory.registerFunction<FunctionBase64Conversion<Base64Decode>>();
}
}
#endif
