#pragma once

#include <DataTypes/DataTypesNumber.h>
#include <Columns/ColumnVector.h>
#include <Functions/IFunctionImpl.h>
#include <IO/WriteHelpers.h>

#include <Functions/TargetSpecific.h>
#include <Functions/PerformanceAdaptors.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

/** Pseudo-random number generation functions.
  * The function can be called without arguments or with one argument.
  * The argument is ignored and only serves to ensure that several calls to one function are considered different and do not stick together.
  *
  * Example:
  * SELECT rand(), rand() - will output two identical columns.
  * SELECT rand(1), rand(2) - will output two different columns.
  *
  * Non-cryptographic generators:
  *
  * rand   - linear congruential generator 0 .. 2^32 - 1.
  * rand64 - combines several rand values to get values from the range 0 .. 2^64 - 1.
  *
  * randConstant - service function, produces a constant column with a random value.
  *
  * The time is used as the seed.
  * Note: it is reinitialized for each block.
  * This means that the timer must be of sufficient resolution to give different values to each block.
  */

DECLARE_MULTITARGET_CODE(

struct RandImpl
{
    static void execute(char * output, size_t size);
    static String getImplementationTag() { return ToString(BuildArch); }
};

struct RandImpl2
{
    static void execute(char * output, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_v2"; }
};

struct RandImpl3
{
    static void execute(char * output, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_v3"; }
};

struct RandImpl4
{
    static void execute(char * output, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_v4"; }
};

struct RandImpl5
{
    static void execute(char * output, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_v5"; }
};

template <int VectorSize>
struct RandVecImpl
{
    static void execute(char * outpu, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_vec_" + toString(VectorSize); }
};

template <int VectorSize>
struct RandVecImpl2
{
    static void execute(char * outpu, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_vec2_" + toString(VectorSize); }
};

struct RandImpl6
{
    static void execute(char * outpu, size_t size);
    static String getImplementationTag() { return ToString(BuildArch) + "_v6"; }
};

) // DECLARE_MULTITARGET_CODE

template <typename RandImpl, typename ToType, typename Name>
class FunctionRandomImpl : public IFunction
{
public:
    static constexpr auto name = Name::name;

    String getName() const override
    {
        return name;
    }

    static String getImplementationTag()
    {
        return RandImpl::getImplementationTag();
    }

    bool isDeterministic() const override { return false; }
    bool isDeterministicInScopeOfQuery() const override { return false; }
    bool useDefaultImplementationForNulls() const override { return false; }

    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() > 1)
            throw Exception("Number of arguments for function " + getName() + " doesn't match: passed "
                + toString(arguments.size()) + ", should be 0 or 1.",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        return std::make_shared<DataTypeNumber<ToType>>();
    }

    void executeImpl(Block & block, const ColumnNumbers &, size_t result, size_t input_rows_count) override
    {
        auto col_to = ColumnVector<ToType>::create();
        typename ColumnVector<ToType>::Container & vec_to = col_to->getData();

        size_t size = input_rows_count;
        vec_to.resize(size);
        RandImpl::execute(reinterpret_cast<char *>(vec_to.data()), vec_to.size() * sizeof(ToType));

        block.getByPosition(result).column = std::move(col_to);
    }
};

template <typename ToType, typename Name>
class FunctionRandom : public FunctionRandomImpl<TargetSpecific::Default::RandImpl, ToType, Name>
{
public:
    FunctionRandom(const Context & context) : selector(context)
    {
        selector.registerImplementation<TargetArch::Default,
            FunctionRandomImpl<TargetSpecific::Default::RandImpl, ToType, Name>>();
        selector.registerImplementation<TargetArch::Default,
            FunctionRandomImpl<TargetSpecific::Default::RandImpl2, ToType, Name>>();

        if constexpr (UseMultitargetCode)
        {
            selector.registerImplementation<TargetArch::SSE42,
                FunctionRandomImpl<TargetSpecific::SSE42::RandImpl, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX,
                FunctionRandomImpl<TargetSpecific::AVX::RandImpl, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX512F,
                FunctionRandomImpl<TargetSpecific::AVX512F::RandImpl, ToType, Name>>();

            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl2, ToType, Name>>();

            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandImpl3, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl3, ToType, Name>>();

            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandImpl4, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl4, ToType, Name>>();

            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandImpl5, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl5, ToType, Name>>();

            // vec impl
            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl<4>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl<4>, ToType, Name>>();
            
            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl<8>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl<8>, ToType, Name>>();

            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl<16>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl<16>, ToType, Name>>();

            // vec impl 2
            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl2<4>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl2<4>, ToType, Name>>();
            
            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl2<8>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl2<8>, ToType, Name>>();

            selector.registerImplementation<TargetArch::Default,
                FunctionRandomImpl<TargetSpecific::Default::RandVecImpl2<16>, ToType, Name>>();
            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandVecImpl2<16>, ToType, Name>>();

            selector.registerImplementation<TargetArch::AVX2,
                FunctionRandomImpl<TargetSpecific::AVX2::RandImpl6, ToType, Name>>();
        }
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t input_rows_count) override
    {
        selector.selectAndExecute(block, arguments, result, input_rows_count);
    }

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<FunctionRandom<ToType, Name>>(context);
    }

private:
    ImplementationSelector<IFunction> selector;
};

}
