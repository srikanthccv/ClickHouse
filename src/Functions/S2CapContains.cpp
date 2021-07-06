#include "config_functions.h"

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <Functions/FunctionFactory.h>
#include <Common/typeid_cast.h>
#include <common/range.h>

#include "s2_fwd.h"

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace
{

/// TODO: Comment this
class FunctionS2CapContains : public IFunction
{
public:
    static constexpr auto name = "S2CapContains";

    static FunctionPtr create(ContextPtr)
    {
        return std::make_shared<FunctionS2CapContains>();
    }

    std::string getName() const override
    {
        return name;
    }

    size_t getNumberOfArguments() const override { return 3; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        size_t number_of_arguments = arguments.size();

        if (number_of_arguments != 3) {
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "Number of arguments for function {} doesn't match: passed {}, should be 3",
                getName(),
                toString(number_of_arguments));
        }

        const auto * arg = arguments[0].get();

        if (!WhichDataType(arg).isUInt64()) {
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument {} of function {}. Must be UInt64",
                arg->getName(), 1, getName());
        }

        arg = arguments[1].get();

        if (!WhichDataType(arg).isFloat64()) {
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument {} of function {}. Must be Float64",
                arg->getName(), 2, getName());
        }

        arg = arguments[2].get();

        if (!WhichDataType(arg).isUInt64()) {
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument {} of function {}. Must be UInt64",
                arg->getName(), 3, getName());
        }

        return std::make_shared<DataTypeUInt8>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * col_center = arguments[0].column.get();
        const auto * col_degrees = arguments[1].column.get();
        const auto * col_point = arguments[2].column.get();

        auto dst = ColumnVector<UInt8>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        for (const auto row : collections::range(0, input_rows_count))
        {
            const UInt64 center = col_center->getUInt(row);
            const Float64 degrees = col_degrees->getFloat64(row);
            const UInt64 point = col_point->getInt(row);

            S1Angle angle = S1Angle::Degrees(degrees);
            S2Cap cap(S2CellId(center).ToPoint(), angle);

            dst_data[row] = cap.Contains(S2CellId(point).ToPoint());
        }

        return dst;
    }

};

}

void registerFunctionS2CapContains(FunctionFactory & factory)
{
    factory.registerFunction<FunctionS2CapContains>();
}


}
