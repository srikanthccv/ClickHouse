#include <DataTypes/DataTypeString.h>
#include <Functions/FunctionFactory.h>
#include <Functions/geometryConverters.h>
#include <Columns/ColumnString.h>

#include <string>
#include <memory>

namespace DB
{

class FunctionWkt : public IFunction
{
public:
    static inline const char * name = "wkt";

    explicit FunctionWkt() {}

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionWkt>();
    }

    String getName() const override
    {
        return name;
    }

    size_t getNumberOfArguments() const override
    {
        return 1;
    }

    DataTypePtr getReturnTypeImpl(const DataTypes &) const override
    {
        return std::make_shared<DataTypeString>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t input_rows_count) const override
    {
        auto parser = makeGeometryFromColumnParser<CartesianPoint>(arguments[0]);
        auto res_column = ColumnString::create();

        auto container = createContainer(parser);

        for (size_t i = 0; i < input_rows_count; i++)
        {
            /// FIXME
            std::stringstream str; // STYLE_CHECK_ALLOW_STD_STRING_STREAM
            get(parser, container, i);
            str << boost::geometry::wkt(container);
            std::string serialized = str.str();
            res_column->insertData(serialized.c_str(), serialized.size());
        }

        return res_column;
    }

    bool useDefaultImplementationForConstants() const override
    {
        return true;
    }
};

void registerFunctionWkt(FunctionFactory & factory)
{
    factory.registerFunction<FunctionWkt>();
}

}
