#if !defined(ARCADIA_BUILD)
#    include "config_functions.h"
#endif

#if USE_H3

#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>
#include <Common/typeid_cast.h>
#include <common/range.h>

#include <h3api.h>


namespace DB
{
namespace ErrorCodes
{
extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

namespace
{

class FunctionH3GetFaces : public IFunction
{
public:
    static constexpr auto name = "h3GetFaces";

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionH3GetFaces>(); }

    std::string getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }
    bool useDefaultImplementationForConstants() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return false; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        const auto * arg = arguments[0].get();
        if (!WhichDataType(arg).isUInt64())
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument {} of function {}. Must be UInt64",
                arg->getName(), 1, getName());

        return std::make_shared<DataTypeUInt8>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * col_hindex = arguments[0].column.get();

        auto dst = ColumnVector<UInt8>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        for (const auto row : collections::range(0, input_rows_count))
        {
            const UInt64 hindex = col_hindex->getUInt(row);

            int max_faces = maxFaceCount(hindex);
            std::unique_ptr<int> faces(new int(max_faces));

            // function name h3GetFaces (v3.x) changed to getIcosahedronFaces (v4.0.0).
            getIcosahedronFaces(hindex, faces.get());

            int total_faces = 0;
            for(int i = 0 ; i < max_faces ; i++) {
                // valid icosahedron faces are represented by integers 0-19
                if (faces.get()[i] >= 0 && faces.get()[i] <= 19) total_faces++;
            }
            dst_data[row] = total_faces;
        }

        return dst;
    }
};

}

void registerFunctionH3GetFaces(FunctionFactory & factory)
{
    factory.registerFunction<FunctionH3GetFaces>();
}

}

#endif
