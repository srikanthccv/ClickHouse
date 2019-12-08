#include <DataTypes/DataTypesNumber.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnConst.h>
#include <Common/typeid_cast.h>
#include <Common/assert_cast.h>
#include <Functions/IFunction.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/FunctionFactory.h>
#include <ext/range.h>
#include <math.h>
#include <array>


namespace DB
{

namespace ErrorCodes
{
    extern const int ARGUMENT_OUT_OF_BOUND;
    extern const int ILLEGAL_COLUMN;
    extern const int LOGICAL_ERROR;
}

/** https://en.wikipedia.org/wiki/Great-circle_distance
 *
 *  The function calculates distance in meters between two points on Earth specified by longitude and latitude in degrees.
 *  The function uses great circle distance formula https://en.wikipedia.org/wiki/Great-circle_distance .
 *  Throws exception when one or several input values are not within reasonable bounds.
 *  Latitude must be in [-90, 90], longitude must be [-180, 180].
 *  Original code of this implementation of this function is here https://github.com/sphinxsearch/sphinx/blob/409f2c2b5b2ff70b04e38f92b6b1a890326bad65/src/sphinxexpr.cpp#L3825.
 *  Andrey Aksenov, the author of original code, permitted to use this code in ClickHouse under the Apache 2.0 license.
 *  Presentation about this code from Highload++ Siberia 2019 is here https://github.com/ClickHouse/ClickHouse/files/3324740/1_._._GEODIST_._.pdf
 *  The main idea of this implementation is optimisations based on Taylor series, trigonometric identity and calculated constants once for cosine, arcsine(sqrt) and look up table.
 */

namespace
{

constexpr double PI = 3.14159265358979323846;
constexpr float DEG_IN_RAD = static_cast<float>(PI / 180.0);
constexpr float DEG_IN_RAD_HALF = static_cast<float>(PI / 360.0);

constexpr size_t COS_LUT_SIZE = 1024; // maxerr 0.00063%
constexpr size_t ASIN_SQRT_LUT_SIZE = 512;
constexpr size_t METRIC_LUT_SIZE = 1024;

float cos_lut[COS_LUT_SIZE + 1];       /// cos(x) table
float asin_sqrt_lut[ASIN_SQRT_LUT_SIZE + 1]; /// asin(sqrt(x)) table
float metric_lut[METRIC_LUT_SIZE + 1][2];    /// geodistAdaptive() flat ellipsoid method k1, k2 coeffs table

inline double sqr(double v)
{
    return v * v;
}

inline float sqrf(float v)
{
    return v * v;
}

void geodistInit()
{
    for (size_t i = 0; i <= COS_LUT_SIZE; ++i)
        cos_lut[i] = static_cast<float>(cos(2 * PI * i / COS_LUT_SIZE)); // [0, 2 * pi] -> [0, COS_LUT_SIZE]

    for (size_t i = 0; i <= ASIN_SQRT_LUT_SIZE; ++i)
        asin_sqrt_lut[i] = static_cast<float>(asin(
            sqrt(static_cast<double>(i) / ASIN_SQRT_LUT_SIZE))); // [0, 1] -> [0, ASIN_SQRT_LUT_SIZE]

    for (size_t i = 0; i <= METRIC_LUT_SIZE; ++i)
    {
        double x = i * (PI / METRIC_LUT_SIZE) - PI * 0.5; // [-pi / 2, pi / 2] -> [0, METRIC_LUT_SIZE]

        metric_lut[i][0] = static_cast<float>(sqr(111132.09 - 566.05 * cos(2 * x) + 1.20 * cos(4 * x)));
        metric_lut[i][1] = static_cast<float>(sqr(111415.13 * cos(x) - 94.55 * cos(3 * x) + 0.12 * cos(5 * x)));
    }
}

inline float geodistDegDiff(float f)
{
    f = fabsf(f);
    while (f > 360)
        f -= 360;
    if (f > 180)
        f = 360 - f;
    return f;
}

inline float geodistFastCos(float x)
{
    float y = fabsf(x) * (COS_LUT_SIZE / PI / 2);
    int i = static_cast<int>(y);
    y -= i;
    i &= (COS_LUT_SIZE - 1);
    return cos_lut[i] + (cos_lut[i + 1] - cos_lut[i]) * y;
}

inline float geodistFastSin(float x)
{
    float y = fabsf(x) * (COS_LUT_SIZE / PI / 2);
    int i = static_cast<int>(y);
    y -= i;
    i = (i - COS_LUT_SIZE / 4) & (COS_LUT_SIZE - 1); // cos(x - pi / 2) = sin(x), costable / 4 = pi / 2
    return cos_lut[i] + (cos_lut[i + 1] - cos_lut[i]) * y;
}

/// fast implementation of asin(sqrt(x))
/// max error in floats 0.00369%, in doubles 0.00072%
inline float geodistFastAsinSqrt(float x)
{
    if (x < 0.122f)
    {
        // distance under 4546km, Taylor error under 0.00072%
        float y = sqrtf(x);
        return y + x * y * 0.166666666666666f + x * x * y * 0.075f + x * x * x * y * 0.044642857142857f;
    }
    if (x < 0.948f)
    {
        // distance under 17083km, 512-entry LUT error under 0.00072%
        x *= ASIN_SQRT_LUT_SIZE;
        int i = static_cast<int>(x);
        return asin_sqrt_lut[i] + (asin_sqrt_lut[i + 1] - asin_sqrt_lut[i]) * (x - i);
    }
    return asinf(sqrtf(x)); // distance over 17083km, just compute honestly
}

}


class FunctionGreatCircleDistance : public IFunction
{
public:
    static constexpr auto name = "greatCircleDistance";
    static FunctionPtr create(const Context &) { return std::make_shared<FunctionGreatCircleDistance>(); }

private:
    String getName() const override { return name; }
    size_t getNumberOfArguments() const override { return 4; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        for (const auto arg_idx : ext::range(0, arguments.size()))
        {
            const auto arg = arguments[arg_idx].get();
            if (!isNumber(WhichDataType(arg)))
                throw Exception(
                    "Illegal type " + arg->getName() + " of argument " + std::to_string(arg_idx + 1) + " of function " + getName() + ". Must be numeric",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        }

        return std::make_shared<DataTypeFloat32>();
    }

    Float32 greatCircleDistance(Float32 lon1deg, Float32 lat1deg, Float32 lon2deg, Float32 lat2deg)
    {
        float lat_diff = geodistDegDiff(lat1deg - lat2deg);
        float lon_diff = geodistDegDiff(lon1deg - lon2deg);

        if (lon_diff < 13)
        {
            // points are close enough; use flat ellipsoid model
            // interpolate metric coefficients using latitudes midpoint

            float latitude_midpoint = (lat1deg + lat2deg + 180) * METRIC_LUT_SIZE / 360; // [-90, 90] degrees -> [0, KTABLE] indexes
            size_t latitude_midpoint_index = static_cast<size_t>(latitude_midpoint) & (METRIC_LUT_SIZE - 1);

            /// This is linear interpolation between two table items at index "latitude_midpoint_index" and "latitude_midpoint_index + 1".

            float k_lat = metric_lut[latitude_midpoint_index][0]
                + (metric_lut[latitude_midpoint_index + 1][0] - metric_lut[latitude_midpoint_index][0]) * (latitude_midpoint - latitude_midpoint_index);

            float k_lon = metric_lut[latitude_midpoint_index][1]
                + (metric_lut[latitude_midpoint_index + 1][1] - metric_lut[latitude_midpoint_index][1]) * (latitude_midpoint - latitude_midpoint_index);

            /// Metric on a tangent plane: it differs from Euclidean metric only by scale of coordinates.

            return sqrtf(k_lat * lat_diff * lat_diff + k_lon * lon_diff * lon_diff);
        }
        else
        {
            // points too far away; use haversine

            /// Earth mean diameter in meters, https://en.wikipedia.org/wiki/Earth
            static constexpr float diameter = 2 * 6371000;

            float a = sqrf(geodistFastSin(lat_diff * DEG_IN_RAD_HALF))
                + geodistFastCos(lat1deg * DEG_IN_RAD) * geodistFastCos(lat2deg * DEG_IN_RAD) * sqrf(geodistFastSin(lon_diff * DEG_IN_RAD_HALF));

            return diameter * geodistFastAsinSqrt(a);
        }
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t input_rows_count) override
    {
        auto dst = ColumnVector<Float32>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        const IColumn & col_lon1 = *block.getByPosition(arguments[0]).column;
        const IColumn & col_lat1 = *block.getByPosition(arguments[1]).column;
        const IColumn & col_lon2 = *block.getByPosition(arguments[2]).column;
        const IColumn & col_lat2 = *block.getByPosition(arguments[3]).column;

        for (size_t row_num = 0; row_num < input_rows_count; ++row_num)
            dst_data[row_num] = greatCircleDistance(
                col_lon1.getFloat32(row_num), col_lat1.getFloat32(row_num),
                col_lon2.getFloat32(row_num), col_lat2.getFloat32(row_num));

        block.getByPosition(result).column = std::move(dst);
    }
};


void registerFunctionGreatCircleDistance(FunctionFactory & factory)
{
    geodistInit();
    factory.registerFunction<FunctionGreatCircleDistance>();
}

}

