#pragma once
#include <stdint.h>
#include <limits>
#include <type_traits>

// Assume little endian

inline uint16_t & LO_16(uint32_t & x) { return reinterpret_cast<uint16_t *>(&x)[0]; }
inline uint16_t & HI_16(uint32_t & x) { return reinterpret_cast<uint16_t *>(&x)[1]; }

inline uint32_t & LO_32(uint64_t & x) { return reinterpret_cast<uint32_t *>(&x)[0]; }
inline uint32_t & HI_32(uint64_t & x) { return reinterpret_cast<uint32_t *>(&x)[1]; }


#if defined(__GNUC__)
        inline unsigned GetValueBitCountImpl(unsigned int value) noexcept {
            // Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned int>::digits - __builtin_clz(value);
        }

        inline unsigned GetValueBitCountImpl(unsigned long value) noexcept {
            // Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned long>::digits - __builtin_clzl(value);
        }

        inline unsigned GetValueBitCountImpl(unsigned long long value) noexcept {
            // Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned long long>::digits - __builtin_clzll(value);
        }
#else
        /// Stupid realization for non-GCC. Can use BSR from x86 instructions set.
        template <typename T>
        inline unsigned GetValueBitCountImpl(T value) noexcept {
            // Y_ASSERT(value);     // because __builtin_clz* have undefined result for zero.
            unsigned result = 1; // result == 0 - impossible value, see Y_ASSERT().
            value >>= 1;
            while (value) {
                value >>= 1;
                ++result;
            }

            return result;
        }
#endif


/**
 * Returns the number of leading 0-bits in `value`, starting at the most significant bit position.
 */
template <typename T>
static inline unsigned GetValueBitCount(T value) noexcept {
    // Y_ASSERT(value > 0);
    using TCvt = std::make_unsigned_t<std::decay_t<T>>;
    return GetValueBitCountImpl(static_cast<TCvt>(value));
}
