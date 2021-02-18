// #include <Functions/FunctionFactory.h>
// #include <Functions/geometryConverters.h>

// #include <boost/geometry.hpp>
// #include <boost/geometry/geometries/point_xy.hpp>
// #include <boost/geometry/geometries/polygon.hpp>

// #include <common/logger_useful.h>

// #include <Columns/ColumnArray.h>
// #include <Columns/ColumnTuple.h>
// #include <Columns/ColumnConst.h>
// #include <DataTypes/DataTypeArray.h>
// #include <DataTypes/DataTypeTuple.h>
// #include <DataTypes/DataTypeCustomGeo.h>

// #include <memory>
// #include <string>

// namespace DB
// {

// template <typename Point>
// class FunctionPolygonsUnion : public IFunction
// {
// public:
//     static inline const char * name;

//     explicit FunctionPolygonsUnion() = default;

//     static FunctionPtr create(const Context &)
//     {
//         return std::make_shared<FunctionPolygonsUnion>();
//     }

//     String getName() const override
//     {
//         return name;
//     }

//     bool isVariadic() const override
//     {
//         return false;
//     }

//     size_t getNumberOfArguments() const override
//     {
//         return 2;
//     }

//     DataTypePtr getReturnTypeImpl(const DataTypes &) const override
//     {
//         return DataTypeCustomMultiPolygonSerialization::nestedDataType();
//     }

//     ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t input_rows_count) const override
//     {
//         checkColumnTypeOrThrow<Point, MultiPolygon>(arguments[0]);
//         auto first_parser = MultiPolygonFromColumnParser<Point>(std::move(arguments[0].column->convertToFullColumnIfConst()));
//         MultiPolygon<Point> first_container;

//         checkColumnTypeOrThrow<Point, MultiPolygon>(arguments[1]);
//         auto second_parser = MultiPolygonFromColumnParser<Point>(std::move(arguments[1].column->convertToFullColumnIfConst()));
//         MultiPolygon<Point> second_container;

//         MultiPolygonSerializer<Point> serializer;
//         MultiPolygon<Point> polygons_union{};

//         /// We are not interested in some pitfalls in third-party libraries
//         /// NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
//         for (size_t i = 0; i < input_rows_count; i++)
//         {
//             first_parser.get(first_container, i);
//             second_parser.get(second_container, i);

//             /// Orient the polygons correctly.
//             boost::geometry::correct(first_container);
//             boost::geometry::correct(second_container);

//             /// Main work here.
//             boost::geometry::union_(first_container, second_container, polygons_union);

//             serializer.add(polygons_union);
//         }

//         return serializer.finalize();
//     }

//     bool useDefaultImplementationForConstants() const override
//     {
//         return true;
//     }
// };

// template <>
// const char * FunctionPolygonsUnion<CartesianPoint>::name = "polygonsUnionCartesian";

// template <>
// const char * FunctionPolygonsUnion<GeographicPoint>::name = "polygonsUnionGeographic";


// void registerFunctionPolygonsUnion(FunctionFactory & factory)
// {
//     factory.registerFunction<FunctionPolygonsUnion<CartesianPoint>>();
//     factory.registerFunction<FunctionPolygonsUnion<GeographicPoint>>();
// }

// }
