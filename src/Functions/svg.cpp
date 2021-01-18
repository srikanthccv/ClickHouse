#include <DataTypes/DataTypeString.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/geometryConverters.h>
#include <Columns/ColumnString.h>

#include <string>
#include <memory>

namespace DB
{

namespace ErrorCodes {
    extern const int TOO_MANY_ARGUMENTS_FOR_FUNCTION;
    extern const int TOO_FEW_ARGUMENTS_FOR_FUNCTION;
}

class FunctionSvg : public IFunction
{
public:
    static inline const char * name = "svg";

    explicit FunctionSvg() {}

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionSvg>();
    }

    String getName() const override
    {
        return name;
    }

    bool isVariadic() const override
    {
        return true;
    }

    size_t getNumberOfArguments() const override
    {
        return 2;
    }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() > 2)
        {
            throw Exception("Too many arguments", ErrorCodes::TOO_MANY_ARGUMENTS_FOR_FUNCTION);
        }
        if (arguments.size() == 0) {
            throw Exception("Too few arguments", ErrorCodes::TOO_FEW_ARGUMENTS_FOR_FUNCTION);
        }
        if (arguments.size() == 2 && checkAndGetDataType<DataTypeString>(arguments[1].get()) == nullptr)
        {
            throw Exception("Second argument should be String",
                        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        }

        return std::make_shared<DataTypeString>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t input_rows_count) const override
    {
        const auto * const_col = checkAndGetColumn<ColumnConst>(arguments[0].column.get());

        auto parser = const_col ?
            makeCartesianGeometryFromColumnParser(ColumnWithTypeAndName(const_col->getDataColumnPtr(), arguments[0].type, arguments[0].name)) :
            makeCartesianGeometryFromColumnParser(arguments[0]);

        bool geo_column_is_const = static_cast<bool>(const_col);

        auto res_column = ColumnString::create();
        auto container = createContainer(parser);

        bool has_style = arguments.size() > 1;
        ColumnPtr style;
        if (has_style) {
            style = arguments[1].column;
        }

        for (size_t i = 0; i < input_rows_count; i++)
        {
            std::stringstream str;
            if (!geo_column_is_const || i == 0)
                get(parser, container, i);

            str << boost::geometry::svg(container, has_style ? style->getDataAt(i).toString() : "");
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

void registerFunctionSvg(FunctionFactory & factory)
{
    factory.registerFunction<FunctionSvg>();
}

}
