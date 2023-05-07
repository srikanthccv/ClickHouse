#include <AggregateFunctions/AggregateFunctionDeltaSumTimestamp.h>

#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/Helpers.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

namespace
{

AggregateFunctionPtr createAggregateFunctionDeltaSumTimestamp(
    const String & name,
    const DataTypes & arguments,
    const Array & params,
    const Settings *)
{
    assertNoParameters(name, params);

    if (arguments.size() != 2)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
            "Incorrect number of arguments for aggregate function {}", name);

    if (!isInteger(arguments[0]) && !isFloat(arguments[0]) && !isDate(arguments[0]) && !isDateTime(arguments[0]))
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument for aggregate function {}, "
                        "must be Int, Float, Date, DateTime", arguments[0]->getName(), name);

    if (!isInteger(arguments[1]) && !isFloat(arguments[1]) && !isDate(arguments[1]) && !isDateTime(arguments[1]))
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument for aggregate function {}, "
                        "must be Int, Float, Date, DateTime", arguments[1]->getName(), name);

    return AggregateFunctionPtr(createWithTwoNumericOrDateTypes<AggregationFunctionDeltaSumTimestamp>(
        *arguments[0], *arguments[1], arguments, params));
}
}

void registerAggregateFunctionDeltaSumTimestamp(AggregateFunctionFactory & factory)
{
    AggregateFunctionProperties properties = { .returns_default_when_only_null = true, .is_order_dependent = true };

    factory.registerFunction("deltaSumTimestamp", { createAggregateFunctionDeltaSumTimestamp, properties });
}

}
