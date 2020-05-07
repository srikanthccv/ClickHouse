#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionWelchTTest.h>
#include <AggregateFunctions/FactoryHelpers.h>


namespace DB
{

namespace
{

template <typename X, typename Y, typename Ret>
AggregateFunctionPtr createAggregateFunctionWelchTTest(const DataTypes & argument_types, const Array & parameters)
{
    // default value
    Float64 significance_level = 0.1;
    if (!params.empty())
    {
        significance_level = applyVisitor(FieldVisitorConvertToNumber<Float64>(), params[0]);
    }


    return std::make_shared<AggregateFunctionWelchTTest<X, Y, Ret>>(significance_level, argument_types, parameters);

}

}

void registerAggregateFunctionWelchTTest(AggregateFunctionFactory & factory)
{

    factory.registerFunction("WelchTTest", createAggregateFunctionWelchTTest);
}

}
