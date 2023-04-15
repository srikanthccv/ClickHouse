#pragma once

#include <base/types.h>

namespace wide
{

template <size_t Bits, typename Signed>
class integer;

}

using Int128 = wide::integer<128, signed>;
using UInt128 = wide::integer<128, unsigned>;
using Int256 = wide::integer<256, signed>;
using UInt256 = wide::integer<256, unsigned>;

namespace DB
{

template <class> struct Decimal;

using Decimal32 = Decimal<Int32>;
using Decimal64 = Decimal<Int64>;
using Decimal128 = Decimal<Int128>;
using Decimal256 = Decimal<Int256>;

class DateTime64;

template <class T>
concept is_decimal =
    std::is_same_v<T, Decimal32>
    || std::is_same_v<T, Decimal64>
    || std::is_same_v<T, Decimal128>
    || std::is_same_v<T, Decimal256>
    || std::is_same_v<T, DateTime64>;

template <class T>
concept is_over_big_int =
    std::is_same_v<T, Int128>
    || std::is_same_v<T, UInt128>
    || std::is_same_v<T, Int256>
    || std::is_same_v<T, UInt256>
    || std::is_same_v<T, Decimal128>
    || std::is_same_v<T, Decimal256>;
}
