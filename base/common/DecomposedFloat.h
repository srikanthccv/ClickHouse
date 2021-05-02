#pragma once

#include <cstdint>
#include <cstring>
#include <common/extended_types.h>


/// Allows to check the internals of IEEE-754 floating point number.

template <typename T> struct FloatTraits;

template <>
struct FloatTraits<float>
{
    using UInt = uint32_t;
    static constexpr size_t bits = 32;
    static constexpr size_t exponent_bits = 8;
    static constexpr size_t mantissa_bits = bits - exponent_bits - 1;
};

template <>
struct FloatTraits<double>
{
    using UInt = uint64_t;
    static constexpr size_t bits = 64;
    static constexpr size_t exponent_bits = 11;
    static constexpr size_t mantissa_bits = bits - exponent_bits - 1;
};


/// x = sign * (2 ^ normalized_exponent) * (1 + mantissa * 2 ^ -mantissa_bits)
/// x = sign * (2 ^ normalized_exponent + mantissa * 2 ^ (normalized_exponent - mantissa_bits))
template <typename T>
struct DecomposedFloat
{
    using Traits = FloatTraits<T>;

    DecomposedFloat(T x)
    {
        memcpy(&x_uint, &x, sizeof(x));
    }

    typename Traits::UInt x_uint;

    bool is_negative() const
    {
        return x_uint >> (Traits::bits - 1);
    }

    /// Returns 0 for both +0. and -0.
    int sign() const
    {
        return (exponent() == 0 && mantissa() == 0)
            ? 0
            : (is_negative()
                ? -1
                : 1);
    }

    uint16_t exponent() const
    {
        return (x_uint >> (Traits::mantissa_bits)) & (((1ull << (Traits::exponent_bits + 1)) - 1) >> 1);
    }

    int16_t normalized_exponent() const
    {
        return int16_t(exponent()) - ((1ull << (Traits::exponent_bits - 1)) - 1);
    }

    uint64_t mantissa() const
    {
        return x_uint & ((1ull << Traits::mantissa_bits) - 1);
    }

    int64_t mantissa_with_sign() const
    {
        return is_negative() ? -mantissa() : mantissa();
    }

    /// NOTE Probably floating point instructions can be better.
    bool is_integer_in_representable_range() const
    {
        return x_uint == 0
            || (normalized_exponent() >= 0  /// The number is not less than one
                /// The number is inside the range where every integer has exact representation in float
                && normalized_exponent() <= static_cast<int16_t>(Traits::mantissa_bits)
                /// After multiplying by 2^exp, the fractional part becomes zero, means the number is integer
                && ((mantissa() & ((1ULL << (Traits::mantissa_bits - normalized_exponent())) - 1)) == 0));
    }


    template <typename Int>
    int compare(Int rhs)
    {
        if (rhs == 0)
            return sign();

        /// Different signs
        if (is_negative() && rhs > 0)
            return -1;
        if (!is_negative() && rhs < 0)
            return 1;

        /// Fractional number with magnitude less than one
        if (normalized_exponent() < 0)
        {
            if (!is_negative())
                return rhs > 0 ? -1 : 1;
            else
                return rhs >= 0 ? -1 : 1;
        }

        /// Too large number: abs(float) > abs(rhs)
        if (normalized_exponent() >= static_cast<int16_t>(8 * sizeof(Int) - is_signed_v<Int>))
        {
            /// The case of most negative integer - it can be equal to float
            if constexpr (is_signed_v<Int>)
            {
                if (rhs == std::numeric_limits<Int>::lowest()
                    && normalized_exponent() == static_cast<int16_t>(8 * sizeof(Int) - is_signed_v<Int>)
                    && is_negative()
                    && mantissa() == 0)
                {
                    return 0;
                }
            }

            return is_negative() ? -1 : 1;
        }

        using UInt = make_unsigned_t<Int>;
        UInt uint_rhs = rhs < 0 ? -rhs : rhs;

        /// Smaller octave: abs(rhs) < abs(float)
        if (uint_rhs < (static_cast<UInt>(1) << normalized_exponent()))
            return is_negative() ? -1 : 1;

        /// Larger octave: abs(rhs) > abs(float)
        if (normalized_exponent() + 1 < static_cast<int16_t>(8 * sizeof(Int) - is_signed_v<Int>)
            && uint_rhs >= (static_cast<UInt>(1) << (normalized_exponent() + 1)))
            return is_negative() ? 1 : -1;

        /// The same octave
        /// uint_rhs == 2 ^ normalized_exponent + mantissa * 2 ^ (normalized_exponent - mantissa_bits)

        bool large_and_always_integer = normalized_exponent() >= static_cast<int16_t>(Traits::mantissa_bits);

        typename Traits::UInt a = large_and_always_integer
            ? mantissa() << (normalized_exponent() - Traits::mantissa_bits)
            : mantissa() >> (Traits::mantissa_bits - normalized_exponent());

        typename Traits::UInt b = uint_rhs - (static_cast<UInt>(1) << normalized_exponent());

        if (a < b)
            return is_negative() ? 1 : -1;
        if (a > b)
            return is_negative() ? -1 : 1;

        /// Float has no fractional part means that the numbers are equal.
        if (large_and_always_integer || (mantissa() & ((1ULL << (Traits::mantissa_bits - normalized_exponent())) - 1)) == 0)
            return 0;
        else
            /// Float has fractional part means its abs value is larger.
            return is_negative() ? -1 : 1;
    }


    template <typename Int>
    bool equals(Int rhs)
    {
        return compare(rhs) == 0;
    }

    template <typename Int>
    bool notEquals(Int rhs)
    {
        return compare(rhs) != 0;
    }

    template <typename Int>
    bool less(Int rhs)
    {
        return compare(rhs) < 0;
    }

    template <typename Int>
    bool greater(Int rhs)
    {
        return compare(rhs) > 0;
    }

    template <typename Int>
    bool lessOrEquals(Int rhs)
    {
        return compare(rhs) <= 0;
    }

    template <typename Int>
    bool greaterOrEquals(Int rhs)
    {
        return compare(rhs) >= 0;
    }
};


using DecomposedFloat64 = DecomposedFloat<double>;
using DecomposedFloat32 = DecomposedFloat<float>;
