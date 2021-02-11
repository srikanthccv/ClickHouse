#include <Functions/FunctionFactory.h>
#include <Functions/geometryConverters.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <common/logger_useful.h>

#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeCustomGeo.h>

#include <memory>
#include <utility>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

template <typename Point>
class FunctionPolygonsEquals : public IFunction
{
public:
    static const char * name;

    explicit FunctionPolygonsEquals() = default;

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionPolygonsEquals>();
    }

    String getName() const override
    {
        return name;
    }

    bool isVariadic() const override
    {
        return false;
    }

    size_t getNumberOfArguments() const override
    {
        return 2;
    }

    DataTypePtr getReturnTypeImpl(const DataTypes &) const override
    {
        return std::make_shared<DataTypeUInt8>();
    }

    void checkInputType(const ColumnsWithTypeAndName & arguments) const
    {
        /// Array(Array(Array(Tuple(Float64, Float64))))
        auto desired = std::make_shared<const DataTypeArray>(
            std::make_shared<const DataTypeArray>(
                std::make_shared<const DataTypeArray>(
                    std::make_shared<const DataTypeTuple>(
                        DataTypes{std::make_shared<const DataTypeFloat64>(), std::make_shared<const DataTypeFloat64>()}
                    )
                )
            )
        );
        if (!desired->equals(*arguments[0].type))
            throw Exception(fmt::format("The type of the first argument of function {} must be Array(Array(Array(Tuple(Float64, Float64))))", name), ErrorCodes::BAD_ARGUMENTS);

        if (!desired->equals(*arguments[1].type))
            throw Exception(fmt::format("The type of the second argument of function {} must be Array(Array(Array(Tuple(Float64, Float64))))", name), ErrorCodes::BAD_ARGUMENTS);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t input_rows_count) const override
    {
        checkInputType(arguments);
        auto first_parser = makeGeometryFromColumnParser<Point>(arguments[0]);
        auto first_container = createContainer(first_parser);

        auto second_parser = makeGeometryFromColumnParser<Point>(arguments[1]);
        auto second_container = createContainer(second_parser);

        auto res_column = ColumnUInt8::create();

        for (size_t i = 0; i < input_rows_count; i++)
        {
            get(first_parser, first_container, i);
            get(second_parser, second_container, i);

            auto first = boost::get<MultiPolygon<Point>>(first_container);
            auto second = boost::get<MultiPolygon<Point>>(second_container);

            boost::geometry::correct(first);
            boost::geometry::correct(second);

            bool equals = boost::geometry::equals(first, second);

            res_column->insertValue(equals);
        }

        return res_column;
    }

    bool useDefaultImplementationForConstants() const override
    {
        return true;
    }
};


template <>
const char * FunctionPolygonsEquals<CartesianPoint>::name = "polygonsEqualsCartesian";


void registerFunctionPolygonsEquals(FunctionFactory & factory)
{
    factory.registerFunction<FunctionPolygonsEquals<CartesianPoint>>();
}

}
