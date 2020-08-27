#include <Functions/IFunctionImpl.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypesDecimal.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnConst.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ILLEGAL_COLUMN;
}

/// Returns 1 if and Decimal value has more digits then it's Precision allow, 0 otherwise.
/// Precision could be set as second argument or omitted. If ommited function uses Decimal presicion of the first argument.
class FunctionCountDigits : public IFunction
{
public:
    static constexpr auto name = "countDigits";

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionCountDigits>();
    }

    String getName() const override { return name; }
    bool useDefaultImplementationForNulls() const override { return false; }
    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        WhichDataType which_first(arguments[0]->getTypeId());

        if (!which_first.isInt() && !which_first.isUInt() && !which_first.isDecimal())
            throw Exception("Illegal type " + arguments[0]->getName() + " of argument of function " + getName(),
                            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        return std::make_shared<DataTypeUInt8>(); /// Up to 255 decimal digits.
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result_pos, size_t input_rows_count) const override
    {
        const auto & src_column = block.getByPosition(arguments[0]);
        if (!src_column.column)
            throw Exception("Illegal column while execute function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        auto result_column = ColumnUInt8::create();

        auto call = [&](const auto & types) -> bool
        {
            using Types = std::decay_t<decltype(types)>;
            using Type = typename Types::RightType;
            using ColVecType = std::conditional_t<IsDecimalNumber<Type>, ColumnDecimal<Type>, ColumnVector<Type>>;

            if (const ColumnConst * const_column = checkAndGetColumnConst<ColVecType>(src_column.column.get()))
            {
                Type const_value = checkAndGetColumn<ColVecType>(const_column->getDataColumnPtr().get())->getData()[0];
                UInt32 num_digits = 0;
                if constexpr (IsDecimalNumber<Type>)
                    num_digits = digits(const_value.value);
                else
                    num_digits = digits(const_value);
                result_column->getData().resize_fill(input_rows_count, num_digits);
                return true;
            }
            else if (const ColVecType * col_vec = checkAndGetColumn<ColVecType>(src_column.column.get()))
            {
                execute<Type>(*col_vec, *result_column, input_rows_count);
                return true;
            }

            throw Exception("Illegal column while execute function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        };

        TypeIndex dec_type_idx = src_column.type->getTypeId();
        if (!callOnBasicType<void, true, false, true, false>(dec_type_idx, call))
            throw Exception("Wrong call for " + getName() + " with " + src_column.type->getName(),
                            ErrorCodes::ILLEGAL_COLUMN);

        block.getByPosition(result_pos).column = std::move(result_column);
    }

private:
    template <typename T, typename ColVecType>
    static void execute(const ColVecType & col, ColumnUInt8 & result_column, size_t rows_count)
    {
        using NativeT = typename NativeType<T>::Type;

        const auto & src_data = col.getData();
        auto & dst_data = result_column.getData();
        dst_data.resize(rows_count);

        for (size_t i = 0; i < rows_count; ++i)
        {
            if constexpr (IsDecimalNumber<T>)
                dst_data[i] = digits<NativeT>(src_data[i].value);
            else
                dst_data[i] = digits<NativeT>(src_data[i]);
        }
    }

    template <typename T>
    static UInt32 digits(T value)
    {
        static_assert(!IsDecimalNumber<T>);
        using DivT = std::conditional_t<is_signed_v<T>, Int32, UInt32>;

        UInt32 res = 0;
        T tmp;

        if constexpr (sizeof(T) > sizeof(Int32))
        {
            static constexpr const DivT e9 = 1000000000;

            tmp = value / e9;
            while (tmp != 0)
            {
                value = tmp;
                tmp /= e9;
                res += 9;
            }
        }

        static constexpr const DivT e3 = 1000;

        tmp = value / e3;
        while (tmp != 0)
        {
            value = tmp;
            tmp /= e3;
            res += 3;
        }

        while (value != 0)
        {
            value /= 10;
            ++res;
        }
        return res;
    }
};


void registerFunctionCountDigits(FunctionFactory & factory)
{
    factory.registerFunction<FunctionCountDigits>();
}

}
