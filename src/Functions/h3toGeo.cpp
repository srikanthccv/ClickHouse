#if !defined(ARCADIA_BUILD)
#    include "config_functions.h"
#endif

#if USE_H3

#include <array>
#include <math.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>
#include <Common/typeid_cast.h>
#include <ext/range.h>

#include <h3api.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ILLEGAL_COLUMN;

}

namespace
{

/// Implements the function h3ToGeo which takes a single argument (h3Index)
/// and returns the longitude and latitude that correspond to the provided h3 index
class FunctionH3ToGeo : public IFunction
{
public:
    static constexpr auto name = "h3ToGeo";

    static FunctionPtr create(ContextConstPtr) { return std::make_shared<FunctionH3ToGeo>(); }

    std::string getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }
    bool useDefaultImplementationForConstants() const override { return true; }

        DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
        {
            const auto * arg = arguments[0].get();
            if (!WhichDataType(arg).isUInt64())
                throw Exception(
                    "Illegal type " + arg->getName() + " of argument " + std::to_string(1) + " of function " + getName() + ". Must be UInt64",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            return std::make_shared<DataTypeTuple>(
                DataTypes{std::make_shared<DataTypeFloat64>(), std::make_shared<DataTypeFloat64>()},
                Strings{"longitude", "latitude"});
        }

        ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
        {
            const auto * col_index = arguments[0].column.get();

            auto dst = ColumnVector<Float64>::create();

            auto & dst_data = dst->getData();
            dst_data.resize(input_rows_count);
            ColumnPtr res_column;


            auto latitude = ColumnFloat64::create(input_rows_count);
            auto longitude = ColumnFloat64::create(input_rows_count);

            ColumnFloat64::Container & lon_data = longitude->getData();
            ColumnFloat64::Container & lat_data = latitude->getData();

            for (const auto row : ext::range(0, input_rows_count))
            {
                const UInt64 h3Index = col_index->getUInt(row);
                GeoCoord coord;
                h3ToGeo(h3Index,&coord);
                lon_data[row] = radsToDegs(coord.lon);
                lat_data[row] = radsToDegs(coord.lat);

            }
            MutableColumns result;
            result.emplace_back(std::move(longitude));
            result.emplace_back(std::move(latitude));
            return ColumnTuple::create(std::move(result));
        }
};

}

void registerFunctionH3ToGeo(FunctionFactory & factory)
{
    factory.registerFunction<FunctionH3ToGeo>();
}

}

#endif
