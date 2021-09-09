#pragma once

#include "GatherUtils.h"
#include "Slices.h"
#include "sliceEqualElements.h"

#if defined(__SSE4_2__)
    #include <emmintrin.h>
    #include <smmintrin.h>
    #include <nmmintrin.h>
#endif
#if defined(__AVX2__)
    #include <immintrin.h>
#endif
namespace DB::GatherUtils
{

namespace
{

inline ALWAYS_INLINE bool hasNull(const UInt8 * null_map, size_t null_map_size)
{
    if (null_map != nullptr)
    {
        for (size_t i = 0; i < null_map_size; ++i)
        {
            if (null_map[i])
                return true;
        }
    }
    return false;
}

}

/// Methods to check if first array has elements from second array, overloaded for various combinations of types.
template <
    ArraySearchType search_type,
    typename FirstSliceType,
    typename SecondSliceType,
          bool (*isEqual)(const FirstSliceType &, const SecondSliceType &, size_t, size_t)>
bool sliceHasImplAnyAll(const FirstSliceType & first, const SecondSliceType & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map != nullptr;

    const bool has_second_null = hasNull(second_null_map, second.size);
    if (has_second_null)
    {
        const bool has_first_null = hasNull(first_null_map, first.size);

        if (has_first_null && search_type == ArraySearchType::Any)
            return true;

        if (!has_first_null && search_type == ArraySearchType::All)
            return false;
    }

    for (size_t i = 0; i < second.size; ++i)
    {
        if (has_second_null_map && second_null_map[i])
            continue;

        bool has = false;

        for (size_t j = 0; j < first.size && !has; ++j)
        {
            if (has_first_null_map && first_null_map[j])
                continue;

            if (isEqual(first, second, j, i))
            {
                has = true;
                break;
            }
        }

        if (has && search_type == ArraySearchType::Any)
            return true;

        if (!has && search_type == ArraySearchType::All)
            return false;
    }
    return search_type == ArraySearchType::All;
}


#if defined(__AVX2__) || defined(__SSE4_2__)

namespace
{

template<class T>
inline ALWAYS_INLINE bool hasAllIntegralLoopRemainder(
    size_t j, const NumericArraySlice<T> & first, const NumericArraySlice<T> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map != nullptr;

    for (; j < second.size; ++j)
    {
        // skip null elements since both have at least one - assuming it was checked earlier that at least one element in 'first' is null
        if (has_second_null_map && second_null_map[j])
            continue;

        bool found = false;

        for (size_t i = 0; i < first.size; ++i)
        {
            if (has_first_null_map && first_null_map[i])
                continue;

            if (first.data[i] == second.data[j])
            {
                found = true;
                break;
            }
        }

        if (!found)
            return false;
    }
    return true;
}

}

#endif

#if defined(__AVX2__)
// AVX2 Int specialization of sliceHasImplAnyAll
template <>
inline ALWAYS_INLINE bool sliceHasImplAnyAll<ArraySearchType::All, NumericArraySlice<int>, NumericArraySlice<int>, sliceEqualElements<int,int> >(
    const NumericArraySlice<int> & first, const NumericArraySlice<int> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    if (second.size == 0)
        return true;

    if (!hasNull(first_null_map, first.size) && hasNull(second_null_map, second.size))
        return false;

    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map != nullptr;

    size_t j = 0;
    short has_mask = 1;
    const int full = -1, none = 0;
    const __m256i ones = _mm256_set1_epi32(full);
    const __m256i zeros = _mm256_setzero_si256();
    if (second.size > 7 && first.size > 7)
    {
        for (; j < second.size - 7 && has_mask; j += 8)
        {
            has_mask = 0;
            const __m256i f_data = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(second.data + j));
            // bits of the bitmask are set to one if considered as null in the corresponding null map, 0 otherwise;
            __m256i bitmask = has_second_null_map ?
                _mm256_set_epi32(
                    (second_null_map[j + 7]) ? full : none,
                    (second_null_map[j + 6]) ? full : none,
                    (second_null_map[j + 5]) ? full : none,
                    (second_null_map[j + 4]) ? full : none,
                    (second_null_map[j + 3]) ? full : none,
                    (second_null_map[j + 2]) ? full : none,
                    (second_null_map[j + 1]) ? full : none,
                    (second_null_map[j]) ? full : none)
                : zeros;

            size_t i = 0;
            for (; i < first.size - 7 && !has_mask; has_mask = _mm256_testc_si256(bitmask, ones), i += 8)
            {
                const __m256i s_data = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(first.data + i));
                // Create a mask to avoid to compare null elements
                // set_m128i takes two arguments: (high segment, low segment) that are two __m128i convert from 8bits to 32bits to match with next operations
                const __m256i first_nm_mask = has_first_null_map?
                    _mm256_set_m128i(
                        _mm_cvtepi8_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i + 4))),
                        _mm_cvtepi8_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i))))
                    : zeros;
                bitmask =
                    _mm256_or_si256(
                        _mm256_or_si256(
                            _mm256_or_si256(
                                _mm256_or_si256(
                                    _mm256_andnot_si256(
                                        first_nm_mask,
                                        _mm256_cmpeq_epi32(f_data, s_data)),
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(6,5,4,3,2,1,0,7)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(6,5,4,3,2,1,0,7))))),
                                _mm256_or_si256(
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(5,4,3,2,1,0,7,6)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(5,4,3,2,1,0,7,6)))),
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(4,3,2,1,0,7,6,5)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(4,3,2,1,0,7,6,5)))))
                            ),
                            _mm256_or_si256(
                                _mm256_or_si256(
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(3,2,1,0,7,6,5,4)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(3,2,1,0,7,6,5,4)))),
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(2,1,0,7,6,5,4,3)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(2,1,0,7,6,5,4,3))))),
                                _mm256_or_si256(
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(1,0,7,6,5,4,3,2)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(1,0,7,6,5,4,3,2)))),
                                    _mm256_andnot_si256(
                                        _mm256_permutevar8x32_epi32(first_nm_mask, _mm256_set_epi32(0,7,6,5,4,3,2,1)),
                                        _mm256_cmpeq_epi32(f_data, _mm256_permutevar8x32_epi32(s_data, _mm256_set_epi32(0,7,6,5,4,3,2,1))))))),
                        bitmask);
            }

            if (i < first.size)
            {
                //  Loop(i)-jam
                for (; i < first.size && !has_mask; ++i)
                {
                    if (has_first_null_map && first_null_map[i])
                        continue;
                    __m256i v_i = _mm256_set1_epi32(first.data[i]);
                    bitmask = _mm256_or_si256(bitmask, _mm256_cmpeq_epi32(f_data, v_i));
                    has_mask = _mm256_testc_si256(bitmask, ones);
                }
            }
        }
    }

    if (!has_mask && second.size > 7)
        return false;

    return hasAllIntegralLoopRemainder(j, first, second, first_null_map, second_null_map);
}

#elif defined(__SSE4_2__)

// SSE4.2 Int specialization
template <>
inline ALWAYS_INLINE bool sliceHasImplAnyAll<ArraySearchType::All, NumericArraySlice<int>, NumericArraySlice<int>, sliceEqualElements<int,int> >(
    const NumericArraySlice<int> & first, const NumericArraySlice<int> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    if (second.size == 0)
        return true;

    if (!hasNull(first_null_map, first.size) && hasNull(second_null_map, second.size))
        return false;

    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map  != nullptr;

    size_t j = 0;
    short has_mask = 1;
    const int full = -1, none = 0;
    const __m128i zeros = _mm_setzero_si128();
    if (second.size > 3 && first.size > 3)
    {
        for (; j < second.size - 3 && has_mask; j += 4)
        {
            has_mask = 0;
            const __m128i f_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(second.data + j));
            __m128i bitmask = has_second_null_map ?
                _mm_set_epi32(
                    (second_null_map[j + 3]) ? full : none,
                    (second_null_map[j + 2]) ? full : none,
                    (second_null_map[j + 1]) ? full : none,
                    (second_null_map[j]) ? full : none)
                : zeros;

            unsigned i = 0;
            for (; i < first.size - 3 && !has_mask; has_mask = _mm_test_all_ones(bitmask), i += 4)
            {
                const __m128i s_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(first.data + i));
                const __m128i first_nm_mask = has_first_null_map ?
                    _mm_cvtepi8_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i)))
                    : zeros;

                bitmask =
                    _mm_or_si128(
                        _mm_or_si128(
                            _mm_or_si128(
                                _mm_andnot_si128(
                                        first_nm_mask,
                                        _mm_cmpeq_epi32(f_data, s_data)),
                                _mm_andnot_si128(
                                    _mm_shuffle_epi32(first_nm_mask, _MM_SHUFFLE(2,1,0,3)),
                                    _mm_cmpeq_epi32(f_data, _mm_shuffle_epi32(s_data, _MM_SHUFFLE(2,1,0,3))))),
                            _mm_or_si128(
                                _mm_andnot_si128(
                                    _mm_shuffle_epi32(first_nm_mask, _MM_SHUFFLE(1,0,3,2)),
                                    _mm_cmpeq_epi32(f_data, _mm_shuffle_epi32(s_data, _MM_SHUFFLE(1,0,3,2)))),
                                _mm_andnot_si128(
                                    _mm_shuffle_epi32(first_nm_mask, _MM_SHUFFLE(0,3,2,1)),
                                    _mm_cmpeq_epi32(f_data, _mm_shuffle_epi32(s_data, _MM_SHUFFLE(0,3,2,1)))))
                        ),
                        bitmask);
            }

            if (i < first.size)
            {
                for (; i < first.size && !has_mask; ++i)
                {
                    if (has_first_null_map && first_null_map[i])
                        continue;
                    __m128i r_i = _mm_set1_epi32(first.data[i]);
                    bitmask = _mm_or_si128(bitmask, _mm_cmpeq_epi32(f_data, r_i));
                    has_mask = _mm_test_all_ones(bitmask);
                }
            }
        }
    }

    if (!has_mask && second.size > 3)
        return false;

    return hasAllIntegralLoopRemainder(j, first, second, first_null_map, second_null_map);
}

// SSE4.2 Int64 specialization
template <>
inline ALWAYS_INLINE bool sliceHasImplAnyAll<ArraySearchType::All, NumericArraySlice<Int64>, NumericArraySlice<Int64>, sliceEqualElements<Int64,Int64> >(
    const NumericArraySlice<Int64> & first, const NumericArraySlice<Int64> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    if (second.size == 0)
        return true;

    if (!hasNull(first_null_map, first.size) && hasNull(second_null_map, second.size))
        return false;

    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map  != nullptr;

    size_t j = 0;
    short has_mask = 1;
    const Int64 full = -1, none = 0;
    const __m128i zeros = _mm_setzero_si128();
    if (second.size > 1 && first.size > 1)
    {
        for (; j < second.size - 1 && has_mask; j += 2)
        {
            has_mask = 0;
            const __m128i f_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(second.data + j));
            __m128i bitmask = has_second_null_map ?
                _mm_set_epi64x(
                    (second_null_map[j + 1]) ? full : none,
                    (second_null_map[j]) ? full : none)
                : zeros;
            unsigned i = 0;
            for (; i < first.size - 1 && !has_mask; has_mask = _mm_test_all_ones(bitmask), i += 2)
            {
                const __m128i s_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(first.data + i));
                const __m128i first_nm_mask = has_first_null_map ?
                    _mm_cvtepi8_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i)))
                    : zeros;
                bitmask =
                    _mm_or_si128(
                            _mm_or_si128(
                                _mm_andnot_si128(
                                    first_nm_mask,
                                    _mm_cmpeq_epi64(f_data, s_data)),
                                _mm_andnot_si128(
                                    _mm_shuffle_epi32(first_nm_mask, _MM_SHUFFLE(1,0,3,2)),
                                    _mm_cmpeq_epi64(f_data, _mm_shuffle_epi32(s_data, _MM_SHUFFLE(1,0,3,2))))),
                        bitmask);
            }

            if (i < first.size)
            {
                for (; i < first.size && !has_mask; ++i)
                {
                    if (has_first_null_map && first_null_map[i])
                        continue;
                    __m128i v_i = _mm_set1_epi64x(first.data[i]);
                    bitmask = _mm_or_si128(bitmask, _mm_cmpeq_epi64(f_data, v_i));
                    has_mask = _mm_test_all_ones(bitmask);
                }
            }
        }
    }

    if (!has_mask && second.size > 1)
        return false;

    return hasAllIntegralLoopRemainder(j, first, second, first_null_map, second_null_map);
}

// SSE4.2 Int16_t specialization
template <>
inline ALWAYS_INLINE bool sliceHasImplAnyAll<ArraySearchType::All, NumericArraySlice<int16_t>, NumericArraySlice<int16_t>, sliceEqualElements<int16_t,int16_t> >(
    const NumericArraySlice<int16_t> & first, const NumericArraySlice<int16_t> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    if (second.size == 0)
        return true;

    if (!hasNull(first_null_map, first.size) && hasNull(second_null_map, second.size))
        return false;

    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map  != nullptr;

    size_t j = 0;
    short has_mask = 1;
    const int16_t full = -1, none = 0;
    const __m128i zeros = _mm_setzero_si128();
    if (second.size > 6 && first.size > 6)
    {
        for (; j < second.size - 7 && has_mask; j += 8)
        {
            has_mask = 0;
            const __m128i f_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(second.data + j));
            __m128i bitmask = has_second_null_map ?
                _mm_set_epi16(
                    (second_null_map[j + 7]) ? full : none, (second_null_map[j + 6]) ? full : none,
                    (second_null_map[j + 5]) ? full : none, (second_null_map[j + 4]) ? full : none,
                    (second_null_map[j + 3]) ? full : none, (second_null_map[j + 2]) ? full : none,
                    (second_null_map[j + 1]) ? full : none, (second_null_map[j]) ? full: none)
                : zeros;
            unsigned i = 0;
            for (; i < first.size-7 && !has_mask; has_mask = _mm_test_all_ones(bitmask), i += 8)
            {
                const __m128i s_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(first.data + i));
                const __m128i first_nm_mask = has_first_null_map ?
                    _mm_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i)))
                    : zeros;
                bitmask =
                    _mm_or_si128(
                            _mm_or_si128(
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            first_nm_mask,
                                            _mm_cmpeq_epi16(f_data, s_data)),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,15,14,13,12)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,15,14,13,12)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(9,8,7,6,5,4,3,2,1,0,15,14,13,12,11,10)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(9,8,7,6,5,4,3,2,1,0,15,14,13,12,11,10)))))
                                ),
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(5,4,3,2,1,0,15,14,13,12,11,10,9,8,7,6)),
                                        _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(5,4,3,2,1,0,15,14,13,12,11,10,9,8,7,6))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(3,2,1,0,15,14,13,12,11,10,9,8,7,6,5,4)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(3,2,1,0,15,14,13,12,11,10,9,8,7,6,5,4)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2)),
                                            _mm_cmpeq_epi16(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2))))))
                        ),
                        bitmask);
            }

            if (i < first.size)
            {
                for (; i < first.size && !has_mask; ++i)
                {
                    if (has_first_null_map && first_null_map[i])
                        continue;
                    __m128i v_i = _mm_set1_epi16(first.data[i]);
                    bitmask = _mm_or_si128(bitmask, _mm_cmpeq_epi16(f_data, v_i));
                    has_mask = _mm_test_all_ones(bitmask);
                }
            }
        }
    }

    if (!has_mask && second.size > 6)
        return false;

    return hasAllIntegralLoopRemainder(j, first, second, first_null_map, second_null_map);
}

// Int8 version is faster with SSE than with AVX2
#if defined(__SSE4_2__)
// SSE2 Int8_t specialization
template <>
inline ALWAYS_INLINE bool sliceHasImplAnyAll<ArraySearchType::All, NumericArraySlice<int8_t>, NumericArraySlice<int8_t>, sliceEqualElements<int8_t,int8_t> >(
    const NumericArraySlice<int8_t> & first, const NumericArraySlice<int8_t> & second, const UInt8 * first_null_map, const UInt8 * second_null_map)
{
    if (second.size == 0)
        return true;

    if (!hasNull(first_null_map, first.size) && hasNull(second_null_map, second.size))
        return false;

    const bool has_first_null_map = first_null_map != nullptr;
    const bool has_second_null_map = second_null_map  != nullptr;

    size_t j = 0;
    short has_mask = 1;
    const int8_t full = -1, none = 0;
    const __m128i zeros = _mm_setzero_si128();
    if (second.size > 15 && first.size > 15)
    {
        for (; j < second.size - 15 && has_mask; j += 16)
        {
            has_mask = 0;
            const __m128i f_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(second.data + j));
            __m128i bitmask = has_second_null_map ?
                _mm_set_epi8(
                    (second_null_map[j + 15]) ? full : none, (second_null_map[j + 14]) ? full : none,
                    (second_null_map[j + 13]) ? full : none, (second_null_map[j + 12]) ? full : none,
                    (second_null_map[j + 11]) ? full : none, (second_null_map[j + 10]) ? full : none,
                    (second_null_map[j + 9]) ? full : none, (second_null_map[j + 8]) ? full : none,
                    (second_null_map[j + 7]) ? full : none, (second_null_map[j + 6]) ? full : none,
                    (second_null_map[j + 5]) ? full : none, (second_null_map[j + 4]) ? full : none,
                    (second_null_map[j + 3]) ? full : none, (second_null_map[j + 2]) ? full : none,
                    (second_null_map[j + 1]) ? full : none, (second_null_map[j]) ? full : none)
                : zeros;
            unsigned i = 0;
            for (; i < first.size - 15 && !has_mask; has_mask = _mm_test_all_ones(bitmask), i += 16)
            {
                const __m128i s_data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(first.data + i));
                const __m128i first_nm_mask = has_first_null_map ?
                    _mm_lddqu_si128(reinterpret_cast<const __m128i *>(first_null_map + i))
                    : zeros;
                bitmask =
                    _mm_or_si128(
                        _mm_or_si128(
                            _mm_or_si128(
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            first_nm_mask,
                                            _mm_cmpeq_epi8(f_data, s_data)),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,15)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,15))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(12,11,10,9,8,7,6,5,4,3,2,1,0,15,14,13)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(12,11,10,9,8,7,6,5,4,3,2,1,0,15,14,13)))))
                                ),
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,15,14,13,12)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,15,14,13,12)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(10,9,8,7,6,5,4,3,2,1,0,15,14,13,12,11)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(10,9,8,7,6,5,4,3,2,1,0,15,14,13,12,11))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(9,8,7,6,5,4,3,2,1,0,15,14,13,12,11,10)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(9,8,7,6,5,4,3,2,1,0,15,14,13,12,11,10)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(8,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(8,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9))))))),
                            _mm_or_si128(
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,7)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,7))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(5,4,3,2,1,0,15,14,13,12,11,10,9,8,7,6)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(5,4,3,2,1,0,15,14,13,12,11,10,9,8,7,6)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(4,3,2,1,0,15,14,13,12,11,10,9,8,7,6,5)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(4,3,2,1,0,15,14,13,12,11,10,9,8,7,6,5)))))),
                                _mm_or_si128(
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(3,2,1,0,15,14,13,12,11,10,9,8,7,6,5,4)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(3,2,1,0,15,14,13,12,11,10,9,8,7,6,5,4)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(2,1,0,15,14,13,12,11,10,9,8,7,6,5,4,3)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(2,1,0,15,14,13,12,11,10,9,8,7,6,5,4,3))))),
                                    _mm_or_si128(
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2)))),
                                        _mm_andnot_si128(
                                            _mm_shuffle_epi8(first_nm_mask, _mm_set_epi8(0,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)),
                                            _mm_cmpeq_epi8(f_data, _mm_shuffle_epi8(s_data, _mm_set_epi8(0,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)))))))),
                        bitmask);
            }

            if (i < first.size)
            {
                for (; i < first.size && !has_mask; ++i)
                {
                    if (has_first_null_map && first_null_map[i])
                        continue;
                    __m128i v_i = _mm_set1_epi8(first.data[i]);
                    bitmask = _mm_or_si128(bitmask, _mm_cmpeq_epi8(f_data, v_i));
                    has_mask = _mm_test_all_ones(bitmask);
                }
            }
        }
    }

    if (!has_mask && second.size > 15)
        return false;

    return hasAllIntegralLoopRemainder(j, first, second, first_null_map, second_null_map);
}

#endif

}
