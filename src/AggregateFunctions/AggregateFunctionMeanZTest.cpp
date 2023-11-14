#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/Moments.h>

#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnTuple.h>
#include <Common/assert_cast.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <cmath>


namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}


namespace DB
{
struct Settings;

namespace
{

/// Returns tuple of (z-statistic, p-value, confidence-interval-low, confidence-interval-high)
template <typename Data>
class AggregateFunctionMeanZTest :
    public IAggregateFunctionDataHelper<Data, AggregateFunctionMeanZTest<Data>>
{
private:
    Float64 pop_var_x;
    Float64 pop_var_y;
    Float64 confidence_level;

public:
    AggregateFunctionMeanZTest(const DataTypes & arguments, const Array & params)
        : IAggregateFunctionDataHelper<Data, AggregateFunctionMeanZTest<Data>>({arguments}, params, createResultType())
    {
        pop_var_x = params.at(0).safeGet<Float64>();
        pop_var_y = params.at(1).safeGet<Float64>();
        confidence_level = params.at(2).safeGet<Float64>();

        if (!std::isfinite(pop_var_x) || !std::isfinite(pop_var_y) || !std::isfinite(confidence_level))
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires finite parameter values.", Data::name);
        }

        if (pop_var_x < 0.0 || pop_var_y < 0.0)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                            "Population variance parameters must be larger than or equal to zero "
                            "in aggregate function {}.", Data::name);
        }

        if (confidence_level <= 0.0 || confidence_level >= 1.0)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Confidence level parameter must be between 0 and 1 in aggregate function {}.", Data::name);
        }
    }

    String getName() const override
    {
        return Data::name;
    }

    static DataTypePtr createResultType()
    {
        DataTypes types
        {
            std::make_shared<DataTypeNumber<Float64>>(),
            std::make_shared<DataTypeNumber<Float64>>(),
            std::make_shared<DataTypeNumber<Float64>>(),
            std::make_shared<DataTypeNumber<Float64>>(),
        };

        Strings names
        {
            "z_statistic",
            "p_value",
            "confidence_interval_low",
            "confidence_interval_high"
        };

        return std::make_shared<DataTypeTuple>(
            std::move(types),
            std::move(names)
        );
    }

    bool allocatesMemoryInArena() const override { return false; }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        Float64 value = columns[0]->getFloat64(row_num);
        UInt8 is_second = columns[1]->getUInt(row_num);

        if (is_second)
            this->data(place).addY(value);
        else
            this->data(place).addX(value);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> /* version */) const override
    {
        this->data(place).write(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> /* version */, Arena *) const override
    {
        this->data(place).read(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto [z_stat, p_value] = this->data(place).getResult(pop_var_x, pop_var_y);
        auto [ci_low, ci_high] = this->data(place).getConfidenceIntervals(pop_var_x, pop_var_y, confidence_level);

        /// Because p-value is a probability.
        p_value = std::min(1.0, std::max(0.0, p_value));

        auto & column_tuple = assert_cast<ColumnTuple &>(to);
        auto & column_stat = assert_cast<ColumnVector<Float64> &>(column_tuple.getColumn(0));
        auto & column_value = assert_cast<ColumnVector<Float64> &>(column_tuple.getColumn(1));
        auto & column_ci_low = assert_cast<ColumnVector<Float64> &>(column_tuple.getColumn(2));
        auto & column_ci_high = assert_cast<ColumnVector<Float64> &>(column_tuple.getColumn(3));

        column_stat.getData().push_back(z_stat);
        column_value.getData().push_back(p_value);
        column_ci_low.getData().push_back(ci_low);
        column_ci_high.getData().push_back(ci_high);
    }
};


struct MeanZTestData : public ZTestMoments<Float64>
{
    static constexpr auto name = "meanZTest";

    std::pair<Float64, Float64> getResult(Float64 pop_var_x, Float64 pop_var_y) const
    {
        Float64 mean_x = getMeanX();
        Float64 mean_y = getMeanY();

        /// z = \frac{\bar{X_{1}} - \bar{X_{2}}}{\sqrt{\frac{\sigma_{1}^{2}}{n_{1}} + \frac{\sigma_{2}^{2}}{n_{2}}}}
        Float64 zstat = (mean_x - mean_y) / getStandardError(pop_var_x, pop_var_y);

        if (unlikely(!std::isfinite(zstat)))
            return {std::numeric_limits<Float64>::quiet_NaN(), std::numeric_limits<Float64>::quiet_NaN()};

        Float64 pvalue = 2.0 * boost::math::cdf(boost::math::normal(0.0, 1.0), -1.0 * std::abs(zstat));

        return {zstat, pvalue};
    }
};

AggregateFunctionPtr createAggregateFunctionMeanZTest(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    assertBinary(name, argument_types);

    if (parameters.size() != 3)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function {} requires three parameter.", name);

    if (!isNumber(argument_types[0]) || !isNumber(argument_types[1]))
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} only supports numerical types", name);

    return std::make_shared<AggregateFunctionMeanZTest<MeanZTestData>>(argument_types, parameters);
}

}

void registerAggregateFunctionMeanZTest(AggregateFunctionFactory & factory)
{
    factory.registerFunction("meanZTest", createAggregateFunctionMeanZTest);
}

}
