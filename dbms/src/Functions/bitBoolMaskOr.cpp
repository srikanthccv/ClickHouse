#include <Functions/FunctionFactory.h>
#include <Functions/FunctionBinaryArithmetic.h>
#include <DataTypes/NumberTraits.h>

namespace DB
{

    template <typename A, typename B>
    struct BitBoolMaskOrImpl
    {
        using ResultType = UInt8;

        template <typename Result = ResultType>
        static inline Result apply(A left, B right)
        {
            return static_cast<ResultType>(
                    ((static_cast<ResultType>(left) & 1) | (static_cast<ResultType>(right) & 1))
                    | ((((static_cast<ResultType>(left) >> 1) & (static_cast<ResultType>(right) >> 1)) & 1) << 1));
        }

#if USE_EMBEDDED_COMPILER
        static constexpr bool compilable = false;

    static inline llvm::Value * compile(llvm::IRBuilder<> & b, llvm::Value * left, llvm::Value * right, bool)
    {
        if (!left->getType()->isIntegerTy() && !right->getType()->isIntegerTy())
            throw Exception("__bitBoolMaskOr expected an integral type", ErrorCodes::LOGICAL_ERROR);
    }
#endif
    };

    struct NameBitBoolMaskOr { static constexpr auto name = "__bitBoolMaskOr"; };
    using FunctionBitBoolMaskOr = FunctionBinaryArithmetic<BitBoolMaskOrImpl, NameBitBoolMaskOr>;

    void registerFunctionBitBoolMaskOr(FunctionFactory & factory)
    {
        factory.registerFunction<FunctionBitBoolMaskOr>();
    }

}
