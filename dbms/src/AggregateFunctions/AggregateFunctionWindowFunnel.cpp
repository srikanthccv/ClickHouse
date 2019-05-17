#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionWindowFunnel.h>
#include <AggregateFunctions/Helpers.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>


namespace DB
{

namespace
{

template <template <typename> class Data>
AggregateFunctionPtr createAggregateFunctionWindowFunnel(const std::string & name, const DataTypes & arguments, const Array & params)
{
    if (params.size() != 1)
        throw Exception{"Aggregate function " + name + " requires exactly one parameter.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

    if (arguments.size() < 2)
        throw Exception("Aggregate function " + name + " requires one timestamp argument and at least one event condition.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    if (arguments.size() > max_events + 1)
        throw Exception("Too many event arguments for aggregate function " + name, ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    AggregateFunctionPtr res(createWithUnsignedIntegerType<AggregateFunctionWindowFunnel, Data>(*arguments[0], arguments, params));
    WhichDataType which(arguments.front().get());
    if (res)
        return res;
    else if (which.isDate())
        return std::make_shared<AggregateFunctionWindowFunnel<DataTypeDate::FieldType, Data<DataTypeDate::FieldType>>>(arguments, params);
    else if (which.isDateTime())
        return std::make_shared<AggregateFunctionWindowFunnel<DataTypeDateTime::FieldType, Data<DataTypeDateTime::FieldType>>>(arguments, params);
    
    throw Exception{"Illegal type " + arguments.front().get()->getName() + " of first argument of aggregate function " 
        + name + ", must be Number, Date, DateTime", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};
}

}

void registerAggregateFunctionWindowFunnel(AggregateFunctionFactory & factory)
{
    factory.registerFunction("windowFunnel", createAggregateFunctionWindowFunnel<AggregateFunctionWindowFunnelData>, AggregateFunctionFactory::CaseInsensitive);
}

}
