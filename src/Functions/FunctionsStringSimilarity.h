#pragma once

#include <Columns/ColumnConst.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunction.h>
#include <Interpreters/Context_fwd.h>

namespace DB
{
/** Calculate similarity metrics:
  *
  * ngramDistance(haystack, needle) - calculate n-gram distance between haystack and needle.
  * Returns float number from 0 to 1 - the closer to zero, the more strings are similar to each other.
  * Also support CaseInsensitive and UTF8 formats.
  * ngramDistanceCaseInsensitive(haystack, needle)
  * ngramDistanceUTF8(haystack, needle)
  * ngramDistanceCaseInsensitiveUTF8(haystack, needle)
  */

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ILLEGAL_COLUMN;
    extern const int TOO_LARGE_STRING_SIZE;
}

template <typename T>
concept has_max_string_size = requires { T::max_string_size; };

template <typename Impl, typename Name>
class FunctionsStringSimilarity : public IFunction
{
public:
    static constexpr auto name = Name::name;

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionsStringSimilarity>(); }

    String getName() const override { return name; }

    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isString(arguments[0]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument of function {}",
                arguments[0]->getName(), getName());

        if (!isString(arguments[1]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument of function {}",
                arguments[1]->getName(), getName());

        return std::make_shared<DataTypeNumber<typename Impl::ResultType>>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        using ResultType = typename Impl::ResultType;

        const ColumnPtr & column_haystack = arguments[0].column;
        const ColumnPtr & column_needle = arguments[1].column;

        const ColumnConst * col_haystack_const = typeid_cast<const ColumnConst *>(&*column_haystack);
        const ColumnConst * col_needle_const = typeid_cast<const ColumnConst *>(&*column_needle);

        if (col_haystack_const && col_needle_const)
        {
            ResultType res{};
            const String & needle = col_needle_const->getValue<String>();
            if constexpr (has_max_string_size<Impl>)
            {
                if (needle.size() > Impl::max_string_size)
                {
                    throw Exception(
                        ErrorCodes::TOO_LARGE_STRING_SIZE,
                        "String size of needle is too big for function {}. "
                        "Should be at most {}",
                        getName(),
                        Impl::max_string_size);
                }
            }
            Impl::constantConstant(col_haystack_const->getValue<String>(), needle, res);
            return result_type->createColumnConst(col_haystack_const->size(), toField(res));
        }

        auto col_res = ColumnVector<ResultType>::create();

        typename ColumnVector<ResultType>::Container & vec_res = col_res->getData();
        vec_res.resize(input_rows_count);

        const ColumnString * col_haystack_vector = checkAndGetColumn<ColumnString>(&*column_haystack);
        const ColumnString * col_needle_vector = checkAndGetColumn<ColumnString>(&*column_needle);

        if (col_haystack_vector && col_needle_const)
        {
            const String & needle = col_needle_const->getValue<String>();
            if constexpr (has_max_string_size<Impl>)
            {
                if (needle.size() > Impl::max_string_size)
                {
                    throw Exception(
                        ErrorCodes::TOO_LARGE_STRING_SIZE,
                        "String size of needle is too big for function {}. "
                        "Should be at most {}",
                        getName(),
                        Impl::max_string_size);
                }
            }
            Impl::vectorConstant(col_haystack_vector->getChars(), col_haystack_vector->getOffsets(), needle, vec_res, input_rows_count);
        }
        else if (col_haystack_vector && col_needle_vector)
        {
            Impl::vectorVector(
                col_haystack_vector->getChars(),
                col_haystack_vector->getOffsets(),
                col_needle_vector->getChars(),
                col_needle_vector->getOffsets(),
                vec_res,
                input_rows_count);
        }
        else if (col_haystack_const && col_needle_vector)
        {
            const String & haystack = col_haystack_const->getValue<String>();
            if constexpr (has_max_string_size<Impl>)
            {
                if (haystack.size() > Impl::max_string_size)
                {
                    throw Exception(
                        ErrorCodes::TOO_LARGE_STRING_SIZE,
                        "String size of haystack is too big for function {}. "
                        "Should be at most {}",
                        getName(),
                        Impl::max_string_size);
                }
            }
            Impl::constantVector(haystack, col_needle_vector->getChars(), col_needle_vector->getOffsets(), vec_res, input_rows_count);
        }
        else
        {
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Illegal columns {} and {} of arguments of function {}",
                arguments[0].column->getName(), arguments[1].column->getName(), getName());
        }

        return col_res;
    }
};

}
