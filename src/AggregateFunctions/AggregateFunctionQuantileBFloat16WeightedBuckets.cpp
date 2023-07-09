#include <AggregateFunctions/AggregateFunctionQuantile.h>
#include <AggregateFunctions/QuantileBFloat16HistogramBuckets.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/Helpers.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <Core/Field.h>


namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

namespace
{

template <typename Value, bool float_return> using FuncQuantileBFloat16WeightedBuckets = AggregateFunctionQuantile<Value, QuantileBFloat16HistogramBuckets<Value>, NameQuantileBFloat16WeightedBuckets, true, std::conditional_t<float_return, Float64, void>, false>;
template <typename Value, bool float_return> using FuncQuantilesBFloat16WeightedBuckets = AggregateFunctionQuantile<Value, QuantileBFloat16HistogramBuckets<Value>, NameQuantilesBFloat16WeightedBuckets, true, std::conditional_t<float_return, Float64, void>, true>;

template <template <typename, bool> class Function>
AggregateFunctionPtr createAggregateFunctionQuantile(
    const std::string & name, const DataTypes & argument_types, const Array & params, const Settings *)
{
    /// Second argument type check doesn't depend on the type of the first one.
    Function<void, true>::assertSecondArg(argument_types);

    const DataTypePtr & argument_type = argument_types[0];
    WhichDataType which(argument_type);

#define DISPATCH(TYPE) \
    if (which.idx == TypeIndex::TYPE) return std::make_shared<Function<TYPE, true>>(argument_types, params);
    FOR_BASIC_NUMERIC_TYPES(DISPATCH)
#undef DISPATCH
    if (which.idx == TypeIndex::Date) return std::make_shared<Function<DataTypeDate::FieldType, false>>(argument_types, params);
    if (which.idx == TypeIndex::DateTime) return std::make_shared<Function<DataTypeDateTime::FieldType, false>>(argument_types, params);

    throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument for aggregate function {}",
                    argument_type->getName(), name);
}

}

void registerAggregateFunctionsQuantileBFloat16WeightedBuckets(AggregateFunctionFactory & factory)
{
    factory.registerFunction(NameQuantileBFloat16WeightedBuckets::name, createAggregateFunctionQuantile<FuncQuantileBFloat16WeightedBuckets>);
    factory.registerFunction(NameQuantilesBFloat16WeightedBuckets::name, createAggregateFunctionQuantile<FuncQuantilesBFloat16WeightedBuckets>);

    /// 'median' is an alias for 'quantile'
    factory.registerAlias("medianBFloat16WeightedBuckets", NameQuantileBFloat16WeightedBuckets::name);
}

}
