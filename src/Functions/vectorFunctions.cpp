#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeInterval.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeNothing.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/ITupleFunction.h>
#include <Functions/castTypeToEither.h>
#include "Functions/IFunction.h"

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ILLEGAL_COLUMN;
    extern const int ARGUMENT_OUT_OF_BOUND;
}

struct PlusName { static constexpr auto name = "plus"; };
struct MinusName { static constexpr auto name = "minus"; };
struct MultiplyName { static constexpr auto name = "multiply"; };
struct DivideName { static constexpr auto name = "divide"; };

struct L1Label { static constexpr auto name = "1"; };
struct L2Label { static constexpr auto name = "2"; };
struct L2SquaredLabel { static constexpr auto name = "2Squared"; };
struct LinfLabel { static constexpr auto name = "inf"; };
struct LpLabel { static constexpr auto name = "p"; };

/// str starts from the lowercase letter; not constexpr due to the compiler version
/*constexpr*/ std::string makeFirstLetterUppercase(const std::string& str)
{
    std::string res(str);
    res[0] += 'A' - 'a';
    return res;
}

template <class FuncName>
class FunctionTupleOperator : public ITupleFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "tuple" + makeFirstLetterUppercase(FuncName::name);

    explicit FunctionTupleOperator(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleOperator>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());

        if (!left_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        if (!right_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 1 of function {} should be tuple, got {}",
                            getName(), arguments[1].type->getName());

        const auto & left_types = left_tuple->getElements();
        const auto & right_types = right_tuple->getElements();

        Columns left_elements;
        Columns right_elements;
        if (arguments[0].column)
            left_elements = getTupleElements(*arguments[0].column);
        if (arguments[1].column)
            right_elements = getTupleElements(*arguments[1].column);

        if (left_types.size() != right_types.size())
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                            "Expected tuples of the same size as arguments of function {}. Got {} and {}",
                            getName(), arguments[0].type->getName(), arguments[1].type->getName());

        size_t tuple_size = left_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto func = FunctionFactory::instance().get(FuncName::name, context);
        DataTypes types(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName left{left_elements.empty() ? nullptr : left_elements[i], left_types[i], {}};
                ColumnWithTypeAndName right{right_elements.empty() ? nullptr : right_elements[i], right_types[i], {}};
                auto elem_func = func->build(ColumnsWithTypeAndName{left, right});
                types[i] = elem_func->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return std::make_shared<DataTypeTuple>(types);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());
        const auto & left_types = left_tuple->getElements();
        const auto & right_types = right_tuple->getElements();
        auto left_elements = getTupleElements(*arguments[0].column);
        auto right_elements = getTupleElements(*arguments[1].column);

        size_t tuple_size = left_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto func = FunctionFactory::instance().get(FuncName::name, context);
        Columns columns(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName left{left_elements[i], left_types[i], {}};
            ColumnWithTypeAndName right{right_elements[i], right_types[i], {}};
            auto elem_func = func->build(ColumnsWithTypeAndName{left, right});
            columns[i] = elem_func->execute({left, right}, elem_func->getResultType(), input_rows_count)
                                  ->convertToFullColumnIfConst();
        }

        return ColumnTuple::create(columns);
    }
};

using FunctionTuplePlus = FunctionTupleOperator<PlusName>;

using FunctionTupleMinus = FunctionTupleOperator<MinusName>;

using FunctionTupleMultiply = FunctionTupleOperator<MultiplyName>;

using FunctionTupleDivide = FunctionTupleOperator<DivideName>;

class FunctionTupleNegate : public ITupleFunction
{
public:
    static constexpr auto name = "tupleNegate";

    explicit FunctionTupleNegate(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleNegate>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto negate = FunctionFactory::instance().get("negate", context);
        DataTypes types(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_negate = negate->build(ColumnsWithTypeAndName{cur});
                types[i] = elem_negate->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return std::make_shared<DataTypeTuple>(types);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto negate = FunctionFactory::instance().get("negate", context);
        Columns columns(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_negate = negate->build(ColumnsWithTypeAndName{cur});
            columns[i] = elem_negate->execute({cur}, elem_negate->getResultType(), input_rows_count)
                                    ->convertToFullColumnIfConst();
        }

        return ColumnTuple::create(columns);
    }
};

template <class FuncName>
class FunctionTupleOperatorByNumber : public ITupleFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "tuple" + makeFirstLetterUppercase(FuncName::name) + "ByNumber";

    explicit FunctionTupleOperatorByNumber(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleOperatorByNumber>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        const auto & p_column = arguments[1];
        auto func = FunctionFactory::instance().get(FuncName::name, context);
        DataTypes types(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_func = func->build(ColumnsWithTypeAndName{cur, p_column});
                types[i] = elem_func->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return std::make_shared<DataTypeTuple>(types);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        const auto & p_column = arguments[1];
        auto func = FunctionFactory::instance().get(FuncName::name, context);
        Columns columns(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_func = func->build(ColumnsWithTypeAndName{cur, p_column});
            columns[i] = elem_func->execute({cur, p_column}, elem_func->getResultType(), input_rows_count)
                                  ->convertToFullColumnIfConst();
        }

        return ColumnTuple::create(columns);
    }
};

using FunctionTupleMultiplyByNumber = FunctionTupleOperatorByNumber<MultiplyName>;

using FunctionTupleDivideByNumber = FunctionTupleOperatorByNumber<DivideName>;

class FunctionDotProduct : public ITupleFunction
{
public:
    static constexpr auto name = "dotProduct";

    explicit FunctionDotProduct(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionDotProduct>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());

        if (!left_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        if (!right_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 1 of function {} should be tuple, got {}",
                            getName(), arguments[1].type->getName());

        const auto & left_types = left_tuple->getElements();
        const auto & right_types = right_tuple->getElements();

        Columns left_elements;
        Columns right_elements;
        if (arguments[0].column)
            left_elements = getTupleElements(*arguments[0].column);
        if (arguments[1].column)
            right_elements = getTupleElements(*arguments[1].column);

        if (left_types.size() != right_types.size())
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                            "Expected tuples of the same size as arguments of function {}. Got {} and {}",
                            getName(), arguments[0].type->getName(), arguments[1].type->getName());

        size_t tuple_size = left_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        DataTypePtr res_type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName left{left_elements.empty() ? nullptr : left_elements[i], left_types[i], {}};
                ColumnWithTypeAndName right{right_elements.empty() ? nullptr : right_elements[i], right_types[i], {}};
                auto elem_multiply = multiply->build(ColumnsWithTypeAndName{left, right});

                if (i == 0)
                {
                    res_type = elem_multiply->getResultType();
                    continue;
                }

                ColumnWithTypeAndName left_type{res_type, {}};
                ColumnWithTypeAndName right_type{elem_multiply->getResultType(), {}};
                auto plus_elem = plus->build({left_type, right_type});
                res_type = plus_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return res_type;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());
        const auto & left_types = left_tuple->getElements();
        const auto & right_types = right_tuple->getElements();
        auto left_elements = getTupleElements(*arguments[0].column);
        auto right_elements = getTupleElements(*arguments[1].column);

        size_t tuple_size = left_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName left{left_elements[i], left_types[i], {}};
            ColumnWithTypeAndName right{right_elements[i], right_types[i], {}};
            auto elem_multiply = multiply->build(ColumnsWithTypeAndName{left, right});

            ColumnWithTypeAndName column;
            column.type = elem_multiply->getResultType();
            column.column = elem_multiply->execute({left, right}, column.type, input_rows_count);

            if (i == 0)
            {
                res = std::move(column);
            }
            else
            {
                auto plus_elem = plus->build({res, column});
                auto res_type = plus_elem->getResultType();
                res.column = plus_elem->execute({res, column}, res_type, input_rows_count);
                res.type = res_type;
            }
        }

        return res.column;
    }
};

template <typename Impl>
class FunctionDateOrDateTimeOperationTupleOfIntervals : public ITupleFunction
{
public:
    static constexpr auto name = Impl::name;

    explicit FunctionDateOrDateTimeOperationTupleOfIntervals(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_)
    {
        return std::make_shared<FunctionDateOrDateTimeOperationTupleOfIntervals>(context_);
    }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (!isDate(arguments[0].type) && !isDate32(arguments[0].type) && !isDateTime(arguments[0].type) && !isDateTime64(arguments[0].type))
                throw Exception{ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of first argument of function {}. Should be a date or a date with time",
                    arguments[0].type->getName(), getName()};

        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());

        if (!cur_tuple)
            throw Exception{ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of second argument of function {}. Should be a tuple",
                    arguments[0].type->getName(), getName()};

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[1].column)
            cur_elements = getTupleElements(*arguments[1].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return arguments[0].type;

        auto plus = FunctionFactory::instance().get(Impl::func_name, context);
        DataTypePtr res_type = arguments[0].type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName left{res_type, {}};
                ColumnWithTypeAndName right{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto plus_elem = plus->build({left, right});
                res_type = plus_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return res_type;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[1].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return arguments[0].column;

        auto plus = FunctionFactory::instance().get(Impl::func_name, context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName column{cur_elements[i], cur_types[i], {}};
            auto elem_plus = plus->build(ColumnsWithTypeAndName{i == 0 ? arguments[0] : res, column});
            auto res_type = elem_plus->getResultType();
            res.column = elem_plus->execute({i == 0 ? arguments[0] : res, column}, res_type, input_rows_count);
            res.type = res_type;
        }

        return res.column;
    }
};

struct AddTupleOfIntervalsImpl
{
    static constexpr auto name = "addTupleOfIntervals";
    static constexpr auto func_name = "plus";
};

struct SubtractTupleOfIntervalsImpl
{
    static constexpr auto name = "subtractTupleOfIntervals";
    static constexpr auto func_name = "minus";
};

using FunctionAddTupleOfIntervals = FunctionDateOrDateTimeOperationTupleOfIntervals<AddTupleOfIntervalsImpl>;

using FunctionSubtractTupleOfIntervals = FunctionDateOrDateTimeOperationTupleOfIntervals<SubtractTupleOfIntervalsImpl>;

template <bool is_minus>
struct FunctionTupleOperationInterval : public ITupleFunction
{
public:
    static constexpr auto name = is_minus ? "subtractInterval" : "addInterval";

    explicit FunctionTupleOperationInterval(ContextPtr context_) : ITupleFunction(context_) {}

    static FunctionPtr create(ContextPtr context_)
    {
        return std::make_shared<FunctionTupleOperationInterval>(context_);
    }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isTuple(arguments[0]) && !isInterval(arguments[0]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of first argument of function {}, must be Tuple or Interval",
                arguments[0]->getName(), getName());

        if (!isInterval(arguments[1]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of second argument of function {}, must be Interval",
                arguments[0]->getName(), getName());

        DataTypes types;

        const auto * tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].get());

        if (tuple)
        {
            const auto & cur_types = tuple->getElements();

            for (auto & type : cur_types)
                if (!isInterval(type))
                    throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                        "Illegal type {} of Tuple element of first argument of function {}, must be Interval",
                        types.back()->getName(), getName());

            types = cur_types;
        }
        else
        {
            types = {arguments[0]};
        }

        const auto * interval_last = checkAndGetDataType<DataTypeInterval>(types.back().get());
        const auto * interval_new = checkAndGetDataType<DataTypeInterval>(arguments[1].get());

        if (!interval_last->equals(*interval_new))
            types.push_back(arguments[1]);

        return std::make_shared<DataTypeTuple>(types);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        if (!isInterval(arguments[1].type))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of second argument of function {}, must be Interval",
                arguments[0].type->getName(), getName());

        Columns tuple_columns;

        const auto * first_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * first_interval = checkAndGetDataType<DataTypeInterval>(arguments[0].type.get());
        const auto * second_interval = checkAndGetDataType<DataTypeInterval>(arguments[1].type.get());

        bool can_be_merged;

        if (first_interval)
        {
            can_be_merged = first_interval->equals(*second_interval);

            if (can_be_merged)
                tuple_columns.resize(1);
            else
                tuple_columns.resize(2);

            tuple_columns[0] = arguments[0].column->convertToFullColumnIfConst();
        }
        else if (first_tuple)
        {
            const auto & cur_types = first_tuple->getElements();

            for (auto & type : cur_types)
                if (!isInterval(type))
                    throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                        "Illegal type {} of Tuple element of first argument of function {}, must be Interval",
                        type->getName(), getName());

            auto cur_elements = getTupleElements(*arguments[0].column);
            size_t tuple_size = cur_elements.size();

            if (tuple_size == 0)
            {
                can_be_merged = false;
            }
            else
            {
                const auto * tuple_last_interval = checkAndGetDataType<DataTypeInterval>(cur_types.back().get());
                can_be_merged = tuple_last_interval->equals(*second_interval);
            }

            if (can_be_merged)
                tuple_columns.resize(tuple_size);
            else
                tuple_columns.resize(tuple_size + 1);

            for (size_t i = 0; i < tuple_size; ++i)
                tuple_columns[i] = cur_elements[i];
        }
        else
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of first argument of function {}, must be Tuple or Interval",
                arguments[0].type->getName(), getName());


        ColumnPtr & last_column = tuple_columns.back();

        if (can_be_merged)
        {
            ColumnWithTypeAndName left{last_column, arguments[1].type, {}};

            if constexpr (is_minus)
            {
                auto minus = FunctionFactory::instance().get("minus", context);
                auto elem_minus = minus->build({left, arguments[1]});
                last_column = elem_minus->execute({left, arguments[1]}, arguments[1].type, input_rows_count)
                                        ->convertToFullColumnIfConst();
            }
            else
            {
                auto plus = FunctionFactory::instance().get("plus", context);
                auto elem_plus = plus->build({left, arguments[1]});
                last_column = elem_plus->execute({left, arguments[1]}, arguments[1].type, input_rows_count)
                                        ->convertToFullColumnIfConst();
            }
        }
        else
        {
            if constexpr (is_minus)
            {
                auto negate = FunctionFactory::instance().get("negate", context);
                auto elem_negate = negate->build({arguments[1]});
                last_column = elem_negate->execute({arguments[1]}, arguments[1].type, input_rows_count);
            }
            else
            {
                last_column = arguments[1].column;
            }
        }

        return ColumnTuple::create(tuple_columns);
    }
};

using FunctionTupleAddInterval = FunctionTupleOperationInterval<false>;

using FunctionTupleSubtractInterval = FunctionTupleOperationInterval<true>;


/// this is for convenient usage in LNormalize
template <class FuncLabel>
class FunctionLNorm : public ITupleFunction {};

template <>
class FunctionLNorm<L1Label> : public ITupleFunction
{
public:
    static constexpr auto name = "L1Norm";

    explicit FunctionLNorm(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto abs = FunctionFactory::instance().get("abs", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        DataTypePtr res_type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});

                if (i == 0)
                {
                    res_type = elem_abs->getResultType();
                    continue;
                }

                ColumnWithTypeAndName left_type{res_type, {}};
                ColumnWithTypeAndName right_type{elem_abs->getResultType(), {}};
                auto plus_elem = plus->build({left_type, right_type});
                res_type = plus_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return res_type;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto abs = FunctionFactory::instance().get("abs", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});

            ColumnWithTypeAndName column;
            column.type = elem_abs->getResultType();
            column.column = elem_abs->execute({cur}, column.type, input_rows_count);

            if (i == 0)
            {
                res = std::move(column);
            }
            else
            {
                auto plus_elem = plus->build({res, column});
                auto res_type = plus_elem->getResultType();
                res.column = plus_elem->execute({res, column}, res_type, input_rows_count);
                res.type = res_type;
            }
        }

        return res.column;
    }
};
using FunctionL1Norm = FunctionLNorm<L1Label>;

template <>
class FunctionLNorm<L2SquaredLabel> : public ITupleFunction
{
public:
    static constexpr auto name = "L2SquaredNorm";

    explicit FunctionLNorm(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        DataTypePtr res_type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_multiply = multiply->build(ColumnsWithTypeAndName{cur, cur});

                if (i == 0)
                {
                    res_type = elem_multiply->getResultType();
                    continue;
                }

                ColumnWithTypeAndName left_type{res_type, {}};
                ColumnWithTypeAndName right_type{elem_multiply->getResultType(), {}};
                auto plus_elem = plus->build({left_type, right_type});
                res_type = plus_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return res_type;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_multiply = multiply->build(ColumnsWithTypeAndName{cur, cur});

            ColumnWithTypeAndName column;
            column.type = elem_multiply->getResultType();
            column.column = elem_multiply->execute({cur, cur}, column.type, input_rows_count);

            if (i == 0)
            {
                res = std::move(column);
            }
            else
            {
                auto plus_elem = plus->build({res, column});
                auto res_type = plus_elem->getResultType();
                res.column = plus_elem->execute({res, column}, res_type, input_rows_count);
                res.type = res_type;
            }
        }

        return res.column;
    }
};
using FunctionL2SquaredNorm = FunctionLNorm<L2SquaredLabel>;

template <>
class FunctionLNorm<L2Label> : public FunctionL2SquaredNorm
{
private:
    using Base =  FunctionL2SquaredNorm;
public:
    static constexpr auto name = "L2Norm";

    explicit FunctionLNorm(ContextPtr context_) : Base(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();
        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto sqrt = FunctionFactory::instance().get("sqrt", context);
        return sqrt->build({ColumnWithTypeAndName{Base::getReturnTypeImpl(arguments), {}}})->getResultType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        ColumnWithTypeAndName squared_res;
        squared_res.type = Base::getReturnTypeImpl(arguments);
        squared_res.column = Base::executeImpl(arguments, squared_res.type, input_rows_count);

        auto sqrt = FunctionFactory::instance().get("sqrt", context);
        auto sqrt_elem = sqrt->build({squared_res});
        return sqrt_elem->execute({squared_res}, sqrt_elem->getResultType(), input_rows_count);
    }
};
using FunctionL2Norm = FunctionLNorm<L2Label>;

template <>
class FunctionLNorm<LinfLabel> : public ITupleFunction
{
public:
    static constexpr auto name = "LinfNorm";

    explicit FunctionLNorm(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        auto abs = FunctionFactory::instance().get("abs", context);
        auto max = FunctionFactory::instance().get("max2", context);
        DataTypePtr res_type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});

                if (i == 0)
                {
                    res_type = elem_abs->getResultType();
                    continue;
                }

                ColumnWithTypeAndName left_type{res_type, {}};
                ColumnWithTypeAndName right_type{elem_abs->getResultType(), {}};
                auto max_elem = max->build({left_type, right_type});
                res_type = max_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        return res_type;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        auto abs = FunctionFactory::instance().get("abs", context);
        auto max = FunctionFactory::instance().get("max2", context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});

            ColumnWithTypeAndName column;
            column.type = elem_abs->getResultType();
            column.column = elem_abs->execute({cur}, column.type, input_rows_count);

            if (i == 0)
            {
                res = std::move(column);
            }
            else
            {
                auto max_elem = max->build({res, column});
                auto res_type = max_elem->getResultType();
                res.column = max_elem->execute({res, column}, res_type, input_rows_count);
                res.type = res_type;
            }
        }

        return res.column;
    }
};
using FunctionLinfNorm = FunctionLNorm<LinfLabel>;

template <>
class FunctionLNorm<LpLabel> : public ITupleFunction
{
public:
    static constexpr auto name = "LpNorm";

    explicit FunctionLNorm(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {1}; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuple, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        const auto & p_column = arguments[1];
        auto abs = FunctionFactory::instance().get("abs", context);
        auto pow = FunctionFactory::instance().get("pow", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        DataTypePtr res_type;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            try
            {
                ColumnWithTypeAndName cur{cur_elements.empty() ? nullptr : cur_elements[i], cur_types[i], {}};
                auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});
                cur.type = elem_abs->getResultType();
                cur.column = cur.type->createColumn();

                auto elem_pow = pow->build(ColumnsWithTypeAndName{cur, p_column});

                if (i == 0)
                {
                    res_type = elem_pow->getResultType();
                    continue;
                }

                ColumnWithTypeAndName left_type{res_type, {}};
                ColumnWithTypeAndName right_type{elem_pow->getResultType(), {}};
                auto plus_elem = plus->build({left_type, right_type});
                res_type = plus_elem->getResultType();
            }
            catch (DB::Exception & e)
            {
                e.addMessage("While executing function {} for tuple element {}", getName(), i);
                throw;
            }
        }

        ColumnWithTypeAndName inv_p_column{std::make_shared<DataTypeFloat64>(), {}};
        return pow->build({ColumnWithTypeAndName{res_type, {}}, inv_p_column})->getResultType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto & cur_types = cur_tuple->getElements();
        auto cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_elements.size();
        if (tuple_size == 0)
            return DataTypeUInt8().createColumnConstWithDefaultValue(input_rows_count);

        const auto & p_column = arguments[1];

        if (!isColumnConst(*p_column.column) && p_column.column->size() != 1)
            throw Exception{"Second argument for function " + getName() + " must be either constant Float64 or constant UInt", ErrorCodes::ILLEGAL_COLUMN};

        double p;
        if (isFloat(p_column.column->getDataType()))
            p = p_column.column->getFloat64(0);
        else if (isUnsignedInteger(p_column.column->getDataType()))
            p = p_column.column->getUInt(0);
        else
            throw Exception{"Second argument for function " + getName() + " must be either constant Float64 or constant UInt", ErrorCodes::ILLEGAL_COLUMN};

        if (p < 1 || p >= HUGE_VAL)
            throw Exception{"Second argument for function " + getName() + " must be not less than one and not be an infinity", ErrorCodes::ARGUMENT_OUT_OF_BOUND};

        auto abs = FunctionFactory::instance().get("abs", context);
        auto pow = FunctionFactory::instance().get("pow", context);
        auto plus = FunctionFactory::instance().get("plus", context);
        ColumnWithTypeAndName res;
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});
            cur.column = elem_abs->execute({cur}, elem_abs->getResultType(), input_rows_count);
            cur.type = elem_abs->getResultType();

            auto elem_pow = pow->build(ColumnsWithTypeAndName{cur, p_column});

            ColumnWithTypeAndName column;
            column.type = elem_pow->getResultType();
            column.column = elem_pow->execute({cur, p_column}, column.type, input_rows_count);

            if (i == 0)
            {
                res = std::move(column);
            }
            else
            {
                auto plus_elem = plus->build({res, column});
                auto res_type = plus_elem->getResultType();
                res.column = plus_elem->execute({res, column}, res_type, input_rows_count);
                res.type = res_type;
            }
        }

        ColumnWithTypeAndName inv_p_column{DataTypeFloat64().createColumnConst(input_rows_count, 1 / p),
                                           std::make_shared<DataTypeFloat64>(), {}};
        auto pow_elem = pow->build({res, inv_p_column});
        return pow_elem->execute({res, inv_p_column}, pow_elem->getResultType(), input_rows_count);
    }
};
using FunctionLpNorm = FunctionLNorm<LpLabel>;

template <class FuncLabel>
class FunctionLDistance : public ITupleFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = std::string("L") + FuncLabel::name + "Distance";

    explicit FunctionLDistance(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLDistance>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override
    {
        if constexpr (FuncLabel::name[0] == 'p')
            return 3;
        else
            return 2;
    }

    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override
    {
        if constexpr (FuncLabel::name[0] == 'p')
            return {2};
        else
            return {};
    }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        FunctionTupleMinus tuple_minus(context);
        auto type = tuple_minus.getReturnTypeImpl(arguments);

        ColumnWithTypeAndName minus_res{type, {}};

        auto func = FunctionFactory::instance().get(std::string("L") + FuncLabel::name + "Norm", context);
        if constexpr (FuncLabel::name[0] == 'p')
            return func->build({minus_res, arguments[2]})->getResultType();
        else
            return func->build({minus_res})->getResultType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        FunctionTupleMinus tuple_minus(context);
        auto type = tuple_minus.getReturnTypeImpl(arguments);
        auto column = tuple_minus.executeImpl(arguments, DataTypePtr(), input_rows_count);

        ColumnWithTypeAndName minus_res{column, type, {}};

        auto func = FunctionFactory::instance().get(std::string("L") + FuncLabel::name + "Norm", context);
        if constexpr (FuncLabel::name[0] == 'p')
        {
            auto func_elem = func->build({minus_res, arguments[2]});
            return func_elem->execute({minus_res, arguments[2]}, func_elem->getResultType(), input_rows_count);
        }
        else
        {
            auto func_elem = func->build({minus_res});
            return func_elem->execute({minus_res}, func_elem->getResultType(), input_rows_count);
        }
    }
};

using FunctionL1Distance = FunctionLDistance<L1Label>;

using FunctionL2Distance = FunctionLDistance<L2Label>;

using FunctionL2SquaredDistance = FunctionLDistance<L2SquaredLabel>;

using FunctionLinfDistance = FunctionLDistance<LinfLabel>;

using FunctionLpDistance = FunctionLDistance<LpLabel>;

template <class FuncLabel>
class FunctionLNormalize : public ITupleFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = std::string("L") + FuncLabel::name + "Normalize";

    explicit FunctionLNormalize(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNormalize>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override
    {
        if constexpr (FuncLabel::name[0] == 'p')
            return 2;
        else
            return 1;
    }

    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override
    {
        if constexpr (FuncLabel::name[0] == 'p')
            return {1};
        else
            return {};
    }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        FunctionLNorm<FuncLabel> norm(context);
        auto type = norm.getReturnTypeImpl(arguments);

        ColumnWithTypeAndName norm_res{type, {}};

        FunctionTupleDivideByNumber divide(context);
        return divide.getReturnTypeImpl({arguments[0], norm_res});
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        FunctionLNorm<FuncLabel> norm(context);
        auto type = norm.getReturnTypeImpl(arguments);
        auto column = norm.executeImpl(arguments, DataTypePtr(), input_rows_count);

        ColumnWithTypeAndName norm_res{column, type, {}};

        FunctionTupleDivideByNumber divide(context);
        return divide.executeImpl({arguments[0], norm_res}, DataTypePtr(), input_rows_count);
    }
};

using FunctionL1Normalize = FunctionLNormalize<L1Label>;

using FunctionL2Normalize = FunctionLNormalize<L2Label>;

using FunctionLinfNormalize = FunctionLNormalize<LinfLabel>;

using FunctionLpNormalize = FunctionLNormalize<LpLabel>;

class FunctionCosineDistance : public ITupleFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "cosineDistance";

    explicit FunctionCosineDistance(ContextPtr context_) : ITupleFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionCosineDistance>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        FunctionDotProduct dot(context);
        ColumnWithTypeAndName dot_result{dot.getReturnTypeImpl(arguments), {}};

        FunctionL2Norm norm(context);
        ColumnWithTypeAndName first_norm{norm.getReturnTypeImpl({arguments[0]}), {}};
        ColumnWithTypeAndName second_norm{norm.getReturnTypeImpl({arguments[1]}), {}};

        auto minus = FunctionFactory::instance().get("minus", context);
        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto divide = FunctionFactory::instance().get("divide", context);

        ColumnWithTypeAndName one{std::make_shared<DataTypeUInt8>(), {}};

        ColumnWithTypeAndName multiply_result{multiply->build({first_norm, second_norm})->getResultType(), {}};
        ColumnWithTypeAndName divide_result{divide->build({dot_result, multiply_result})->getResultType(), {}};
        return minus->build({one, divide_result})->getResultType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        if (getReturnTypeImpl(arguments)->isNullable())
        {
            return DataTypeNullable(std::make_shared<DataTypeNothing>())
                   .createColumnConstWithDefaultValue(input_rows_count);
        }

        FunctionDotProduct dot(context);
        ColumnWithTypeAndName dot_result{dot.executeImpl(arguments, DataTypePtr(), input_rows_count),
                                         dot.getReturnTypeImpl(arguments), {}};

        FunctionL2Norm norm(context);
        ColumnWithTypeAndName first_norm{norm.executeImpl({arguments[0]}, DataTypePtr(), input_rows_count),
                                         norm.getReturnTypeImpl({arguments[0]}), {}};
        ColumnWithTypeAndName second_norm{norm.executeImpl({arguments[1]}, DataTypePtr(), input_rows_count),
                                          norm.getReturnTypeImpl({arguments[1]}), {}};

        auto minus = FunctionFactory::instance().get("minus", context);
        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto divide = FunctionFactory::instance().get("divide", context);

        ColumnWithTypeAndName one{DataTypeUInt8().createColumnConst(input_rows_count, 1),
                                  std::make_shared<DataTypeUInt8>(), {}};

        auto multiply_elem = multiply->build({first_norm, second_norm});
        ColumnWithTypeAndName multiply_result;
        multiply_result.type = multiply_elem->getResultType();
        multiply_result.column = multiply_elem->execute({first_norm, second_norm},
                                                        multiply_result.type, input_rows_count);

        auto divide_elem = divide->build({dot_result, multiply_result});
        ColumnWithTypeAndName divide_result;
        divide_result.type = divide_elem->getResultType();
        divide_result.column = divide_elem->execute({dot_result, multiply_result},
                                                    divide_result.type, input_rows_count);

        auto minus_elem = minus->build({one, divide_result});
        return minus_elem->execute({one, divide_result}, minus_elem->getResultType(), {});
    }
};


/// An adaptor to call Norm/Distance function for tuple or array depending on the 1st argument type
template <class Traits>
class TupleOrArrayFunction : public IFunction
{
public:
    static constexpr auto name = Traits::name;

    explicit TupleOrArrayFunction(ContextPtr context_)
        : IFunction()
        , tuple_function(Traits::CreateTupleFunction(context_))
        , array_function(Traits::CreateArrayFunction(context_)) {}

    static FunctionPtr create(ContextPtr context_) { return std::make_shared<TupleOrArrayFunction>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return tuple_function->getNumberOfArguments(); }

    bool useDefaultImplementationForConstants() const override { return true; }

    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return false; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        bool is_array = checkDataTypes<DataTypeArray>(arguments[0].type.get());
        return (is_array ? array_function : tuple_function)->getReturnTypeImpl(arguments);
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        bool is_array = checkDataTypes<DataTypeArray>(arguments[0].type.get());
        return (is_array ? array_function : tuple_function)->executeImpl(arguments, result_type, input_rows_count);
    }

private:
    FunctionPtr tuple_function;
    FunctionPtr array_function;
};

extern FunctionPtr createFunctionArrayL1Norm(ContextPtr context_);
extern FunctionPtr createFunctionArrayL2Norm(ContextPtr context_);
extern FunctionPtr createFunctionArrayL2SquaredNorm(ContextPtr context_);
extern FunctionPtr createFunctionArrayLpNorm(ContextPtr context_);
extern FunctionPtr createFunctionArrayLinfNorm(ContextPtr context_);

extern FunctionPtr createFunctionArrayL1Distance(ContextPtr context_);
extern FunctionPtr createFunctionArrayL2Distance(ContextPtr context_);
extern FunctionPtr createFunctionArrayL2SquaredDistance(ContextPtr context_);
extern FunctionPtr createFunctionArrayLpDistance(ContextPtr context_);
extern FunctionPtr createFunctionArrayLinfDistance(ContextPtr context_);
extern FunctionPtr createFunctionArrayCosineDistance(ContextPtr context_);

struct L1NormTraits
{
    static constexpr auto name = "L1Norm";

    static constexpr auto CreateTupleFunction = FunctionL1Norm::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL1Norm;
};

struct L2NormTraits
{
    static constexpr auto name = "L2Norm";

    static constexpr auto CreateTupleFunction = FunctionL2Norm::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL2Norm;
};

struct L2SquaredNormTraits
{
    static constexpr auto name = "L2SquaredNorm";

    static constexpr auto CreateTupleFunction = FunctionL2SquaredNorm::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL2SquaredNorm;
};

struct LpNormTraits
{
    static constexpr auto name = "LpNorm";

    static constexpr auto CreateTupleFunction = FunctionLpNorm::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayLpNorm;
};

struct LinfNormTraits
{
    static constexpr auto name = "LinfNorm";

    static constexpr auto CreateTupleFunction = FunctionLinfNorm::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayLinfNorm;
};

struct L1DistanceTraits
{
    static constexpr auto name = "L1Distance";

    static constexpr auto CreateTupleFunction = FunctionL1Distance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL1Distance;
};

struct L2DistanceTraits
{
    static constexpr auto name = "L2Distance";

    static constexpr auto CreateTupleFunction = FunctionL2Distance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL2Distance;
};

struct L2SquaredDistanceTraits
{
    static constexpr auto name = "L2SquaredDistance";

    static constexpr auto CreateTupleFunction = FunctionL2SquaredDistance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayL2SquaredDistance;
};

struct LpDistanceTraits
{
    static constexpr auto name = "LpDistance";

    static constexpr auto CreateTupleFunction = FunctionLpDistance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayLpDistance;
};

struct LinfDistanceTraits
{
    static constexpr auto name = "LinfDistance";

    static constexpr auto CreateTupleFunction = FunctionLinfDistance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayLinfDistance;
};

struct CosineDistanceTraits
{
    static constexpr auto name = "cosineDistance";

    static constexpr auto CreateTupleFunction = FunctionCosineDistance::create;
    static constexpr auto CreateArrayFunction = createFunctionArrayCosineDistance;
};

using TupleOrArrayFunctionL1Norm = TupleOrArrayFunction<L1NormTraits>;
using TupleOrArrayFunctionL2Norm = TupleOrArrayFunction<L2NormTraits>;
using TupleOrArrayFunctionL2SquaredNorm = TupleOrArrayFunction<L2SquaredNormTraits>;
using TupleOrArrayFunctionLpNorm = TupleOrArrayFunction<LpNormTraits>;
using TupleOrArrayFunctionLinfNorm = TupleOrArrayFunction<LinfNormTraits>;

using TupleOrArrayFunctionL1Distance = TupleOrArrayFunction<L1DistanceTraits>;
using TupleOrArrayFunctionL2Distance = TupleOrArrayFunction<L2DistanceTraits>;
using TupleOrArrayFunctionL2SquaredDistance = TupleOrArrayFunction<L2SquaredDistanceTraits>;
using TupleOrArrayFunctionLpDistance = TupleOrArrayFunction<LpDistanceTraits>;
using TupleOrArrayFunctionLinfDistance = TupleOrArrayFunction<LinfDistanceTraits>;
using TupleOrArrayFunctionCosineDistance = TupleOrArrayFunction<CosineDistanceTraits>;

REGISTER_FUNCTION(VectorFunctions)
{
    factory.registerFunction<FunctionTuplePlus>();
    factory.registerAlias("vectorSum", FunctionTuplePlus::name, FunctionFactory::CaseInsensitive);
    factory.registerFunction<FunctionTupleMinus>();
    factory.registerAlias("vectorDifference", FunctionTupleMinus::name, FunctionFactory::CaseInsensitive);
    factory.registerFunction<FunctionTupleMultiply>();
    factory.registerFunction<FunctionTupleDivide>();
    factory.registerFunction<FunctionTupleNegate>();

    factory.registerFunction<FunctionAddTupleOfIntervals>(
        {
            R"(
Consecutively adds a tuple of intervals to a Date or a DateTime.
[example:tuple]
)",
            Documentation::Examples{
                {"tuple", "WITH toDate('2018-01-01') AS date SELECT addTupleOfIntervals(date, (INTERVAL 1 DAY, INTERVAL 1 YEAR))"},
                },
            Documentation::Categories{"Tuple", "Interval", "Date", "DateTime"}
        });

    factory.registerFunction<FunctionSubtractTupleOfIntervals>(
        {
            R"(
Consecutively subtracts a tuple of intervals from a Date or a DateTime.
[example:tuple]
)",
            Documentation::Examples{
                {"tuple", "WITH toDate('2018-01-01') AS date SELECT subtractTupleOfIntervals(date, (INTERVAL 1 DAY, INTERVAL 1 YEAR))"},
                },
            Documentation::Categories{"Tuple", "Interval", "Date", "DateTime"}
        });

    factory.registerFunction<FunctionTupleAddInterval>(
        {
            R"(
Adds an interval to another interval or tuple of intervals. The returned value is tuple of intervals.
[example:tuple]
[example:interval1]

If the types of the first interval (or the interval in the tuple) and the second interval are the same they will be merged into one interval.
[example:interval2]
)",
            Documentation::Examples{
                {"tuple", "SELECT addInterval((INTERVAL 1 DAY, INTERVAL 1 YEAR), INTERVAL 1 MONTH)"},
                {"interval1", "SELECT addInterval(INTERVAL 1 DAY, INTERVAL 1 MONTH)"},
                {"interval2", "SELECT addInterval(INTERVAL 1 DAY, INTERVAL 1 DAY)"},
                },
            Documentation::Categories{"Tuple", "Interval"}
        });
    factory.registerFunction<FunctionTupleSubtractInterval>(
        {
            R"(
Adds an negated interval to another interval or tuple of intervals. The returned value is tuple of intervals.
[example:tuple]
[example:interval1]

If the types of the first interval (or the interval in the tuple) and the second interval are the same they will be merged into one interval.
[example:interval2]
)",
            Documentation::Examples{
                {"tuple", "SELECT subtractInterval((INTERVAL 1 DAY, INTERVAL 1 YEAR), INTERVAL 1 MONTH)"},
                {"interval1", "SELECT subtractInterval(INTERVAL 1 DAY, INTERVAL 1 MONTH)"},
                {"interval2", "SELECT subtractInterval(INTERVAL 2 DAY, INTERVAL 1 DAY)"},
                },
            Documentation::Categories{"Tuple", "Interval"}
        });

    factory.registerFunction<FunctionTupleMultiplyByNumber>();
    factory.registerFunction<FunctionTupleDivideByNumber>();

    factory.registerFunction<FunctionDotProduct>();
    factory.registerAlias("scalarProduct", FunctionDotProduct::name, FunctionFactory::CaseInsensitive);

    factory.registerFunction<TupleOrArrayFunctionL1Norm>();
    factory.registerFunction<TupleOrArrayFunctionL2Norm>();
    factory.registerFunction<TupleOrArrayFunctionL2SquaredNorm>();
    factory.registerFunction<TupleOrArrayFunctionLinfNorm>();
    factory.registerFunction<TupleOrArrayFunctionLpNorm>();

    factory.registerAlias("normL1", TupleOrArrayFunctionL1Norm::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normL2", TupleOrArrayFunctionL2Norm::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normL2Squared", TupleOrArrayFunctionL2SquaredNorm::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normLinf", TupleOrArrayFunctionLinfNorm::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normLp", FunctionLpNorm::name, FunctionFactory::CaseInsensitive);

    factory.registerFunction<TupleOrArrayFunctionL1Distance>();
    factory.registerFunction<TupleOrArrayFunctionL2Distance>();
    factory.registerFunction<TupleOrArrayFunctionL2SquaredDistance>();
    factory.registerFunction<TupleOrArrayFunctionLinfDistance>();
    factory.registerFunction<TupleOrArrayFunctionLpDistance>();

    factory.registerAlias("distanceL1", FunctionL1Distance::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("distanceL2", FunctionL2Distance::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("distanceL2Squared", FunctionL2SquaredDistance::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("distanceLinf", FunctionLinfDistance::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("distanceLp", FunctionLpDistance::name, FunctionFactory::CaseInsensitive);

    factory.registerFunction<FunctionL1Normalize>();
    factory.registerFunction<FunctionL2Normalize>();
    factory.registerFunction<FunctionLinfNormalize>();
    factory.registerFunction<FunctionLpNormalize>();

    factory.registerAlias("normalizeL1", FunctionL1Normalize::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normalizeL2", FunctionL2Normalize::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normalizeLinf", FunctionLinfNormalize::name, FunctionFactory::CaseInsensitive);
    factory.registerAlias("normalizeLp", FunctionLpNormalize::name, FunctionFactory::CaseInsensitive);

    factory.registerFunction<TupleOrArrayFunctionCosineDistance>();
}
}
