#include "AggregateFunctionHistogram.h"
#include "AggregateFunctionFactory.h"
#include "FactoryHelpers.h"
#include "Helpers.h"

#include <Common/FieldVisitors.h>

namespace DB {

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int BAD_ARGUMENTS;
}

namespace {

AggregateFunctionPtr createAggregateFunctionHistogram(const std::string & name, const DataTypes & arguments, const Array & params)
{
    if (params.size() != 1)
        throw Exception("Function " + name + " requires bins count", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    UInt32 bins_count = applyVisitor(FieldVisitorConvertToNumber<UInt32>(), params[0]);

    if (bins_count == 0)
        throw Exception("Bin count should be positive", ErrorCodes::BAD_ARGUMENTS);

    assertUnary(name, arguments);
    AggregateFunctionPtr res(createWithNumericType<AggregateFunctionHistogram>(*arguments[0], bins_count));

    if (!res)
        throw Exception("Illegal type " + arguments[0]->getName() + " of argument for aggregate function " + name, ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

    return res;
}

}

void registerAggregateFunctionHistogram(AggregateFunctionFactory & factory)
{
    factory.registerFunction("histogram", createAggregateFunctionHistogram);
}

}
