#include <Functions/FunctionFactory.h>
#include <Functions/FunctionBinaryArithmetic.h>
#include <Common/hex.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
    extern const int LOGICAL_ERROR;
}

namespace
{

template <typename A, typename B>
struct BitShiftRightImpl
{
    using ResultType = typename NumberTraits::ResultOfBit<A, B>::Type;
    static const constexpr bool allow_fixed_string = false;
    static const constexpr bool allow_string_integer = true;

    template <typename Result = ResultType>
    static inline NO_SANITIZE_UNDEFINED Result apply(A a [[maybe_unused]], B b [[maybe_unused]])
    {
        if constexpr (is_big_int_v<B>)
            throw Exception("BitShiftRight is not implemented for big integers as second argument", ErrorCodes::NOT_IMPLEMENTED);
        else if constexpr (is_big_int_v<A>)
            return static_cast<Result>(a) >> static_cast<UInt32>(b);
        else
            return static_cast<Result>(a) >> static_cast<Result>(b);
    }

    static inline NO_SANITIZE_UNDEFINED void apply(const UInt8 * pos [[maybe_unused]], const UInt8 * end [[maybe_unused]], const B & b [[maybe_unused]], ColumnString::Chars & out_vec, ColumnString::Offsets & out_offsets)
    {
        if constexpr (is_big_int_v<B>)
            throw Exception("BitShiftRight is not implemented for big integers as second argument", ErrorCodes::NOT_IMPLEMENTED);
        else
        {
            UInt8 word_size = 8;
            if (b >= static_cast<B>((end - pos) * word_size))
            {
                // insert default value
                out_vec.push_back(0);
                out_offsets.push_back(out_offsets.back() + 1);
                return;
            }

            size_t shift_right_bytes = b / word_size;
            size_t shift_right_bits = b % word_size;

            const UInt8 * begin = pos;
            const UInt8 * shift_right_end = end - shift_right_bytes;

            const size_t old_size = out_vec.size();
            size_t length = shift_right_end - begin;
            const size_t new_size = old_size + length + 1;
            out_vec.resize(new_size);

            UInt8 * op_pointer = const_cast<UInt8 *>(shift_right_end);
            out_vec[old_size + length] = 0;
            UInt8 * out = out_vec.data() + old_size + length;
            while (op_pointer > begin)
            {
                op_pointer--;
                out--;
                UInt8 temp_value = *op_pointer >> shift_right_bits;
                if (op_pointer - 1 >= begin)
                {
                    *out = UInt8(UInt8(*(op_pointer - 1) << (8 - shift_right_bits)) | temp_value);
                }
                else
                    *out = temp_value;
            }
            out_offsets.push_back(new_size);
        }
    }

#if USE_EMBEDDED_COMPILER
    static constexpr bool compilable = true;

    static inline llvm::Value * compile(llvm::IRBuilder<> & b, llvm::Value * left, llvm::Value * right, bool is_signed)
    {
        if (!left->getType()->isIntegerTy())
            throw Exception("BitShiftRightImpl expected an integral type", ErrorCodes::LOGICAL_ERROR);
        return is_signed ? b.CreateAShr(left, right) : b.CreateLShr(left, right);
    }
#endif
};


struct NameBitShiftRight { static constexpr auto name = "bitShiftRight"; };
using FunctionBitShiftRight = BinaryArithmeticOverloadResolver<BitShiftRightImpl, NameBitShiftRight, true, false>;

}

void registerFunctionBitShiftRight(FunctionFactory & factory)
{
    factory.registerFunction<FunctionBitShiftRight>();
}

}
