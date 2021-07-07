#if !defined(ARCADIA_BUILD)
#    include "config_functions.h"
#endif

#if USE_S2_GEOMETRY

#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>
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
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace
{

/// TODO: Comment this
class FunctionDegreesToS2 : public IFunction
{
public:
    static constexpr auto name = "degreesToS2";

    static FunctionPtr create(ContextPtr)
    {
        return std::make_shared<FunctionDegreesToS2>();
    }

    std::string getName() const override
    {
        return name;
    }

    size_t getNumberOfArguments() const override { return 2; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        for (size_t i = 0; i < getNumberOfArguments(); ++i)
        {
            const auto * arg = arguments[i].get();
            if (!WhichDataType(arg).isFloat64()) {
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of argument {} of function {}. Must be Float64",
                    arg->getName(), i, getName()
                    );
            }
        }

        return std::make_shared<DataTypeUInt64>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * col_lon = arguments[0].column.get();
        const auto * col_lat = arguments[1].column.get();

        auto dst = ColumnVector<UInt64>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        for (const auto row : collections::range(0, input_rows_count))
        {
            const Float64 lon = col_lon->getFloat64(row);
            const Float64 lat = col_lat->getFloat64(row);

            S2LatLng lat_lng = S2LatLng::FromDegrees(lat, lon);
            S2CellId id(lat_lng);

            dst_data[row] = id.id();
        }

        return dst;
    }

};

}

void registerFunctionDegreesToS2(FunctionFactory & factory)
{
    factory.registerFunction<FunctionDegreesToS2>();
}


}

#endif
