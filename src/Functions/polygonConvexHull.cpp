#include <Functions/FunctionFactory.h>
#include <Functions/geometryConverters.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

template <typename Point>
class FunctionPolygonConvexHull : public IFunction
{
public:
    static const char * name;

    explicit FunctionPolygonConvexHull() = default;

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionPolygonConvexHull>();
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
        return 1;
    }

    DataTypePtr getReturnTypeImpl(const DataTypes &) const override
    {
        return DataTypeCustomPolygonSerialization::nestedDataType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t input_rows_count) const override
    {
        PolygonSerializer<Point> serializer;

        callOnGeometryDataType<Point>(arguments[0].type, [&] (const auto & type)
        {
            using TypeConverter = std::decay_t<decltype(type)>;
            using Converter = typename TypeConverter::Type;

            if constexpr (std::is_same_v<Converter, ColumnToPointsConverter<Point>>)
                throw Exception(fmt::format("The argument of function {} must not be a Point", getName()), ErrorCodes::BAD_ARGUMENTS);
            else
            {
                auto geometries = Converter::convert(arguments[0].column->convertToFullColumnIfConst());

                for (size_t i = 0; i < input_rows_count; i++)
                {
                    Polygon<Point> convex_hull{};
                    boost::geometry::convex_hull(geometries[i], convex_hull);
                    serializer.add(convex_hull);
                }
            }
        }
        );

        return serializer.finalize();
    }

    bool useDefaultImplementationForConstants() const override
    {
        return true;
    }
};


template <>
const char * FunctionPolygonConvexHull<CartesianPoint>::name = "polygonConvexHullCartesian";


void registerFunctionPolygonConvexHull(FunctionFactory & factory)
{
    factory.registerFunction<FunctionPolygonConvexHull<CartesianPoint>>();
}

}
