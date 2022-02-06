#include <Columns/ColumnTuple.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <Functions/IFunction.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <cfloat>


namespace DB
{

    template <typename Impl>
    class FunctionMinSampleSize : public IFunction
    {
    public:
        static constexpr auto name = Impl::name;

        static FunctionPtr create(ContextPtr)
        {
            return std::make_shared<FunctionMinSampleSize<Impl>>();
        }

        String getName() const override
        {
            return name;
        }

        size_t getNumberOfArguments() const override { return Impl::num_args; }

        bool useDefaultImplementationForNulls() const override { return false; }
        bool useDefaultImplementationForConstants() const override { return true; }
        bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return false; }

        static DataTypePtr getReturnType()
        {
            DataTypes types
            {
                std::make_shared<DataTypeNumber<UInt64>>(),
                std::make_shared<DataTypeNumber<Float64>>(),
                std::make_shared<DataTypeNumber<Float64>>(),
            };

            Strings names
            {
                "minimum_sample_size",
                "detect_range_lower",
                "detect_range_upper",
            };

            return std::make_shared<DataTypeTuple>(
                std::move(types),
                std::move(names)
            );
        }

        DataTypePtr getReturnTypeImpl(const DataTypes & /*arguments*/) const override
        {
            return getReturnType();
        }

        ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
        {
            MutableColumnPtr to{getReturnType()->createColumn()};
            to->reserve(input_rows_count);

            for (size_t row_num = 0; row_num < input_rows_count; ++row_num)
            {
                to->insert(Impl::execute(arguments, row_num));
            }

            return to;
        }

    };


    static bool isBetweenZeroAndOne(Float64 v)
    {
        return v >= 0.0 && v <= 1.0 && fabs(v - 0.0) >= DBL_EPSILON && fabs(v - 1.0) >= DBL_EPSILON;
    }


    struct ContinousImpl
    {
        static constexpr auto name = "minSampleSizeContinous";
        static constexpr size_t num_args = 5;

        static Tuple execute(const ColumnsWithTypeAndName & arguments, size_t row_num)
        {
            /// Mean of control-metric
            Float64 baseline = arguments[0].column->getFloat64(row_num);
            /// Standard deviation of conrol-metric
            Float64 sigma = arguments[1].column->getFloat64(row_num);
            /// Minimal Detectable Effect
            Float64 mde = arguments[2].column->getFloat64(row_num);
            /// Sufficient statistical power to detect a treatment effect
            Float64 power = arguments[3].column->getFloat64(row_num);
            /// Significance level
            Float64 alpha = arguments[4].column->getFloat64(row_num);

            if (!std::isfinite(baseline) || !std::isfinite(sigma) || !isBetweenZeroAndOne(mde) || !isBetweenZeroAndOne(power) || !isBetweenZeroAndOne(alpha))
            {
                return {0, std::numeric_limits<Float64>::quiet_NaN(), std::numeric_limits<Float64>::quiet_NaN()};
            }

            Float64 delta = baseline * mde;

            using namespace boost::math;
            normal_distribution<> nd(0.0, 1.0);
            Float64 min_sample_size = 2 * (std::pow(sigma, 2)) * std::pow(quantile(nd, 1.0 - alpha / 2) + quantile(nd, power), 2) / std::pow(delta, 2);

            return {static_cast<UInt64>(min_sample_size), baseline - delta, baseline + delta};
        }
    };


    struct ConversionImpl
    {
        static constexpr auto name = "minSampleSizeConversion";
        static constexpr size_t num_args = 4;

        static Tuple execute(const ColumnsWithTypeAndName & arguments, size_t row_num)
        {
            /// Mean of control-metric
            Float64 p1 = arguments[0].column->getFloat64(row_num);
            /// Minimal Detectable Effect
            Float64 mde = arguments[1].column->getFloat64(row_num);
            /// Sufficient statistical power to detect a treatment effect
            Float64 power = arguments[2].column->getFloat64(row_num);
            /// Significance level
            Float64 alpha = arguments[3].column->getFloat64(row_num);

            if (!std::isfinite(p1) || !isBetweenZeroAndOne(mde) || !isBetweenZeroAndOne(power) || !isBetweenZeroAndOne(alpha))
            {
                return {0, std::numeric_limits<Float64>::quiet_NaN(), std::numeric_limits<Float64>::quiet_NaN()};
            }

            Float64 q1 = 1.0 - p1;
            Float64 p2 = p1 + mde;
            Float64 q2 = 1.0 - p2;
            Float64 p_bar = (p1 + p2) / 2.0;

            using namespace boost::math;
            normal_distribution<> nd(0.0, 1.0);
            Float64 min_sample_size = std::pow(
                quantile(nd, 1.0 - alpha / 2.0) * std::sqrt(2.0 * p_bar * (1 - p_bar))
                + quantile(nd, power) * std::sqrt(p1 * q1 + p2 * q2), 2
            ) / std::pow(mde, 2);

            return {static_cast<UInt64>(min_sample_size), p1 - mde, p1 + mde};
        }
    };


    void registerFunctionMinSampleSize(FunctionFactory & factory)
    {
        factory.registerFunction<FunctionMinSampleSize<ContinousImpl>>();
        factory.registerFunction<FunctionMinSampleSize<ConversionImpl>>();
    }

}

