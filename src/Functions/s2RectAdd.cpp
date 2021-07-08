#if !defined(ARCADIA_BUILD)
#    include "config_functions.h"
#endif

#if USE_S2_GEOMETRY

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <Functions/FunctionFactory.h>
#include <Common/typeid_cast.h>
#include <common/range.h>

#include "s2_fwd.h"

class S2CellId;

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

namespace
{


class FunctionS2RectAdd : public IFunction
{
public:
    static constexpr auto name = "s2RectAdd";

    static FunctionPtr create(ContextPtr)
    {
        return std::make_shared<FunctionS2RectAdd>();
    }

    std::string getName() const override
    {
        return name;
    }

    size_t getNumberOfArguments() const override { return 4; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        for (size_t index = 0; index < getNumberOfArguments(); ++index)
        {
            const auto * arg = arguments[index].get();
            if (!WhichDataType(arg).isUInt64())
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of argument {} of function {}. Must be UInt64",
                    arg->getName(), index, getName());
        }

        DataTypePtr element = std::make_shared<DataTypeUInt64>();

        return std::make_shared<DataTypeTuple>(DataTypes{element, element});
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * col_lo = arguments[0].column.get();
        const auto * col_hi = arguments[1].column.get();
        const auto * col_point = arguments[2].column.get();

        auto col_res_first = ColumnUInt64::create();
        auto col_res_second = ColumnUInt64::create();

        auto & vec_res_first = col_res_first->getData();
        vec_res_first.reserve(input_rows_count);

        auto & vec_res_second = col_res_second->getData();
        vec_res_second.reserve(input_rows_count);

        for (const auto row : collections::range(0, input_rows_count))
        {
            const UInt64 lo = col_lo->getUInt(row);
            const UInt64 hi = col_hi->getUInt(row);
            const UInt64 point = col_point->getUInt(row);

            S2LatLngRect rect(S2CellId(lo).ToLatLng(), S2CellId(hi).ToLatLng());

            rect.AddPoint(S2CellId(point).ToPoint());

            vec_res_first.emplace_back(S2CellId(rect.lo()).id());
            vec_res_second.emplace_back(S2CellId(rect.hi()).id());
        }

        return ColumnTuple::create(Columns{std::move(col_res_first), std::move(col_res_second)});
    }

};

}

void registerFunctionS2RectAdd(FunctionFactory & factory)
{
    factory.registerFunction<FunctionS2RectAdd>();
}


}

#endif
