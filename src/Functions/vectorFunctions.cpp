#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/TupleIFunction.h>
#include <Functions/castTypeToEither.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
}

static const char PLUS_NAME[] = "plus";
static const char MINUS_NAME[] = "minus";
static const char MULTIPLY_NAME[] = "multiply";
static const char DIVIDE_NAME[] = "divide";

static constexpr char L1_LABEL[] = "1";
static constexpr char L2_LABEL[] = "2";
static constexpr char Linf_LABEL[] = "inf";
static constexpr char Lp_LABEL[] = "p";

/// str starts from the lowercase letter; not constexpr due to the compiler version
/*constexpr*/ std::string makeFirstLetterUppercase(std::string && str)
{
    std::string res(str);
    res[0] += 'A' - 'a';
    return res;
}

template <const char * func_name>
class FunctionTupleOperator : public TupleIFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "tuple" + makeFirstLetterUppercase(func_name);

    explicit FunctionTupleOperator(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleOperator>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());

        if (!left_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
                            getName(), arguments[0].type->getName());

        if (!right_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 1 of function {} should be tuples, got {}",
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

        auto func = FunctionFactory::instance().get(func_name, context);
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

        auto func = FunctionFactory::instance().get(func_name, context);
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

using FunctionTuplePlus = FunctionTupleOperator<PLUS_NAME>;

using FunctionTupleMinus = FunctionTupleOperator<MINUS_NAME>;

using FunctionTupleMultiply = FunctionTupleOperator<MULTIPLY_NAME>;

using FunctionTupleDivide = FunctionTupleOperator<DIVIDE_NAME>;

class FunctionTupleNegate : public TupleIFunction
{
public:
    static constexpr auto name = "tupleNegate";

    explicit FunctionTupleNegate(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleNegate>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
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
            columns[i] = elem_negate->execute({cur}, elem_negate->getResultType(), input_rows_count);
        }

        return ColumnTuple::create(columns);
    }
};

template <const char * func_name>
class FunctionTupleOperatorByNumber : public TupleIFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "tuple" + makeFirstLetterUppercase(func_name) + "ByNumber";

    explicit FunctionTupleOperatorByNumber(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionTupleOperatorByNumber>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
                            getName(), arguments[0].type->getName());

        const auto & cur_types = cur_tuple->getElements();

        Columns cur_elements;
        if (arguments[0].column)
            cur_elements = getTupleElements(*arguments[0].column);

        size_t tuple_size = cur_types.size();
        if (tuple_size == 0)
            return std::make_shared<DataTypeUInt8>();

        const auto & p_column = arguments[1];
        auto func = FunctionFactory::instance().get(func_name, context);
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
        auto func = FunctionFactory::instance().get(func_name, context);
        Columns columns(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i)
        {
            ColumnWithTypeAndName cur{cur_elements[i], cur_types[i], {}};
            auto elem_func = func->build(ColumnsWithTypeAndName{cur, p_column});
            columns[i] = elem_func->execute({cur, p_column}, elem_func->getResultType(), input_rows_count);
        }

        return ColumnTuple::create(columns);
    }
};

using FunctionTupleMultiplyByNumber = FunctionTupleOperatorByNumber<MULTIPLY_NAME>;

using FunctionTupleDivideByNumber = FunctionTupleOperatorByNumber<DIVIDE_NAME>;

class FunctionDotProduct : public TupleIFunction
{
public:
    static constexpr auto name = "dotProduct";

    explicit FunctionDotProduct(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionDotProduct>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * left_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const auto * right_tuple = checkAndGetDataType<DataTypeTuple>(arguments[1].type.get());

        if (!left_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
                            getName(), arguments[0].type->getName());

        if (!right_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 1 of function {} should be tuples, got {}",
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

/// this is for convenient usage in LNormalize
template <const char * func_label>
class FunctionLNorm : public TupleIFunction {};

template <>
class FunctionLNorm<L1_LABEL> : public TupleIFunction
{
public:
    static constexpr auto name = "L1Norm";

    explicit FunctionLNorm(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
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
using FunctionL1Norm = FunctionLNorm<L1_LABEL>;

template <>
class FunctionLNorm<L2_LABEL> : public TupleIFunction
{
public:
    static constexpr auto name = "L2Norm";

    explicit FunctionLNorm(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
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

        auto sqrt = FunctionFactory::instance().get("sqrt", context);
        return sqrt->build({ColumnWithTypeAndName{res_type, {}}})->getResultType();
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

        auto sqrt = FunctionFactory::instance().get("sqrt", context);
        auto sqrt_elem = sqrt->build({res});
        return sqrt_elem->execute({res}, sqrt_elem->getResultType(), input_rows_count);
    }
};
using FunctionL2Norm = FunctionLNorm<L2_LABEL>;

template <>
class FunctionLNorm<Linf_LABEL> : public TupleIFunction
{
public:
    static constexpr auto name = "LinfNorm";

    explicit FunctionLNorm(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
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
using FunctionLinfNorm = FunctionLNorm<Linf_LABEL>;

template <>
class FunctionLNorm<Lp_LABEL> : public TupleIFunction
{
public:
    static constexpr auto name = "LpNorm";

    explicit FunctionLNorm(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNorm>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 2; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const auto * cur_tuple = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());

        if (!cur_tuple)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Argument 0 of function {} should be tuples, got {}",
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

        auto divide = FunctionFactory::instance().get("divide", context);
        ColumnWithTypeAndName one{std::make_shared<DataTypeUInt8>(), {}};
        auto div_elem = divide->build({one, p_column});
        ColumnWithTypeAndName inv_p_column{div_elem->getResultType(), {}};
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

        auto divide = FunctionFactory::instance().get("divide", context);
        ColumnWithTypeAndName one{DataTypeUInt8().createColumnConst(input_rows_count, 1)->convertToFullColumnIfConst(),
                                  std::make_shared<DataTypeUInt8>(), {}};
        auto div_elem = divide->build({one, p_column});
        ColumnWithTypeAndName inv_p_column;
        inv_p_column.type = div_elem->getResultType();
        inv_p_column.column = div_elem->execute({one, p_column}, inv_p_column.type, input_rows_count);
        auto pow_elem = pow->build({res, inv_p_column});
        return pow_elem->execute({res, inv_p_column}, pow_elem->getResultType(), input_rows_count);
    }
};
using FunctionLpNorm = FunctionLNorm<Lp_LABEL>;

template <const char * func_label>
class FunctionLDistance : public TupleIFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "L" + std::string(func_label) + "Distance";

    explicit FunctionLDistance(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLDistance>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override
    {
        if constexpr (func_label[0] == 'p')
            return 3;
        else
            return 2;
    }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        FunctionTupleMinus tuple_minus(context);
        auto type = tuple_minus.getReturnTypeImpl(arguments);

        ColumnWithTypeAndName minus_res{type, {}};

        auto func = FunctionFactory::instance().get("L" + std::string(func_label) + "Norm", context);
        if constexpr (func_label[0] == 'p')
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

        auto func = FunctionFactory::instance().get("L" + std::string(func_label) + "Norm", context);
        if constexpr (func_label[0] == 'p')
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

using FunctionL1Distance = FunctionLDistance<L1_LABEL>;

using FunctionL2Distance = FunctionLDistance<L2_LABEL>;

using FunctionLinfDistance = FunctionLDistance<Linf_LABEL>;

using FunctionLpDistance = FunctionLDistance<Lp_LABEL>;

template <const char * func_label>
class FunctionLNormalize : public TupleIFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "L" + std::string(func_label) + "Normalize";

    explicit FunctionLNormalize(ContextPtr context_) : TupleIFunction(context_) {}
    static FunctionPtr create(ContextPtr context_) { return std::make_shared<FunctionLNormalize>(context_); }

    String getName() const override { return name; }

    size_t getNumberOfArguments() const override
    {
        if constexpr (func_label[0] == 'p')
            return 2;
        else
            return 1;
    }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        FunctionLNorm<func_label> norm(context);
        auto type = norm.getReturnTypeImpl(arguments);

        ColumnWithTypeAndName norm_res{type, {}};

        FunctionTupleDivideByNumber divide(context);
        return divide.getReturnTypeImpl({arguments[0], norm_res});
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        FunctionLNorm<func_label> norm(context);
        auto type = norm.getReturnTypeImpl(arguments);
        auto column = norm.executeImpl(arguments, DataTypePtr(), input_rows_count);

        ColumnWithTypeAndName norm_res{column, type, {}};

        FunctionTupleDivideByNumber divide(context);
        return divide.executeImpl({arguments[0], norm_res}, DataTypePtr(), input_rows_count);
    }
};

using FunctionL1Normalize = FunctionLNormalize<L1_LABEL>;

using FunctionL2Normalize = FunctionLNormalize<L2_LABEL>;

using FunctionLinfNormalize = FunctionLNormalize<Linf_LABEL>;

using FunctionLpNormalize = FunctionLNormalize<Lp_LABEL>;

/*class FunctionCosineDistance : public TupleIFunction
{
public:
    /// constexpr cannot be used due to std::string has not constexpr constructor in this compiler version
    static inline auto name = "cosineDistance";

    explicit FunctionCosineDistance(ContextPtr context_) : TupleIFunction(context_) {}
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

        auto multiply = FunctionFactory::instance().get("multiply", context);
        auto divide = FunctionFactory::instance().get("divide", context);

        auto elem_multiply

            auto elem_abs = abs->build(ColumnsWithTypeAndName{cur});
        cur.column = elem_abs->execute({cur}, elem_abs->getResultType(), 1);
        cur.type = elem_abs->getResultType();

        return func->build({minus_res, arguments[2]})->getResultType();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        FunctionTupleMinus tuple_minus(context);
        auto type = tuple_minus.getReturnTypeImpl(arguments);
        auto column = tuple_minus.executeImpl(arguments, DataTypePtr(), input_rows_count);

        ColumnWithTypeAndName minus_res{column, type, {}};

        auto func = FunctionFactory::instance().get("L" + std::string(func_label) + "Norm", context);
        if constexpr (func_label[0] == 'p')
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
};*/

void registerVectorFunctions(FunctionFactory & factory)
{
    factory.registerFunction<FunctionTuplePlus>();
    factory.registerAlias("vectorSum", FunctionTuplePlus::name, FunctionFactory::CaseInsensitive);
    factory.registerFunction<FunctionTupleMinus>();
    factory.registerAlias("vectorDifference", FunctionTupleMinus::name, FunctionFactory::CaseInsensitive);
    factory.registerFunction<FunctionTupleMultiply>();
    factory.registerFunction<FunctionTupleDivide>();
    factory.registerFunction<FunctionTupleNegate>();

    factory.registerFunction<FunctionTupleMultiplyByNumber>();
    factory.registerFunction<FunctionTupleDivideByNumber>();

    factory.registerFunction<FunctionDotProduct>();
    factory.registerAlias("scalarProduct", FunctionDotProduct::name, FunctionFactory::CaseInsensitive);

    factory.registerFunction<FunctionL1Norm>();
    factory.registerFunction<FunctionL2Norm>();
    factory.registerFunction<FunctionLinfNorm>();
    factory.registerFunction<FunctionLpNorm>();

    factory.registerFunction<FunctionL1Distance>();
    factory.registerFunction<FunctionL2Distance>();
    factory.registerFunction<FunctionLinfDistance>();
    factory.registerFunction<FunctionLpDistance>();

    factory.registerFunction<FunctionL1Normalize>();
    factory.registerFunction<FunctionL2Normalize>();
    factory.registerFunction<FunctionLinfNormalize>();
    factory.registerFunction<FunctionLpNormalize>();
//
//    factory.registerFunction<FunctionCosineDistance>();
}
}
