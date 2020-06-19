#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionRankCorr.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include "registerAggregateFunctions.h"

#include <DataTypes/DataTypeAggregateFunction.h>


namespace ErrorCodes
{
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int NOT_IMPLEMENTED;
}

namespace DB
{

namespace
{

AggregateFunctionPtr createAggregateFunctionRankCorr(const std::string & name,
                                                       const DataTypes & argument_types,
                                                       const Array & parameters)
{
    assertBinary(name, argument_types);
    assertNoParameters(name, parameters);

    AggregateFunctionPtr res;

    if (isDecimal(argument_types[0]) || isDecimal(argument_types[1]))
    {
        throw Exception("Aggregate function " + name + " only supports numerical types.", ErrorCodes::NOT_IMPLEMENTED);
    }

    else
    {
        res.reset(createWithTwoNumericTypes<AggregateFunctionRankCorr>(*argument_types[0], *argument_types[1],
                                                                         argument_types));
    }


    if (!res)
    {
        throw Exception("Aggregate function " + name + " only supports numerical types.", ErrorCodes::NOT_IMPLEMENTED);
    }

    return res;
}

}


void registerAggregateFunctionRankCorr(AggregateFunctionFactory & factory)
{
    factory.registerFunction("RankCorr", createAggregateFunctionRankCorr, AggregateFunctionFactory::CaseInsensitive);
}

}
