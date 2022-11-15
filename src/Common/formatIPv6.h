#pragma once

#include <base/types.h>
#include <cstring>
#include <algorithm>
#include <utility>
#include <base/range.h>
#include <base/unaligned.h>
#include <Common/hex.h>
#include <Common/StringUtils/StringUtils.h>

constexpr size_t IPV4_BINARY_LENGTH = 4;
constexpr size_t IPV6_BINARY_LENGTH = 16;
constexpr size_t IPV4_MAX_TEXT_LENGTH = 15;     /// Does not count tail zero byte.
constexpr size_t IPV6_MAX_TEXT_LENGTH = 45;     /// Does not count tail zero byte.

namespace DB
{


/** Rewritten inet_ntop6 from http://svn.apache.org/repos/asf/apr/apr/trunk/network_io/unix/inet_pton.c
  *  performs significantly faster than the reference implementation due to the absence of sprintf calls,
  *  bounds checking, unnecessary string copying and length calculation.
  */
void formatIPv6(const unsigned char * src, char *& dst, uint8_t zeroed_tail_bytes_count = 0);

/** Unsafe (no bounds-checking for src nor dst), optimized version of parsing IPv4 string.
 *
 * Parses the input string `src` and stores binary host-endian value into buffer pointed by `dst`,
 * which should be long enough.
 * That is "127.0.0.1" becomes 0x7f000001.
 *
 * In case of failure returns nullptr and doesn't modify buffer pointed by `dst`.
 *
 * @param src     - beginning of the input string.
 * @param src_end - optional, end of the input string, if nullptr string will be parsed until not valid symbol is met.
 * @param dst     - where to put output bytes, expected to be non-null and at IPV4_BINARY_LENGTH-long.
 * @return if success pointer to the address immediately after parsed sequence, nullptr otherwise.
 */
inline const char * parseIPv4(const char * src, const char * src_end, unsigned char * dst)
{
    if (src == nullptr || (src_end && src_end - src < 7))
        return nullptr;

    UInt32 result = 0;
    for (int offset = 24; offset >= 0; offset -= 8)
    {
        UInt32 value = 0;
        size_t len = 0;
        while (isNumericASCII(*src) && len <= 3)
        {
            value = value * 10 + (*src - '0');
            ++len;
            ++src;
            if (src_end && src == src_end)
                break;
        }
        if (len == 0 || value > 255 || (offset > 0 && *src != '.'))
            return nullptr;
        result |= value << offset;
        ++src;
    }

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    reverseMemcpy(dst, &result, sizeof(result));
#else
    memcpy(dst, &result, sizeof(result));
#endif
    return src - 1;
}

inline bool parseIPv4(const char * src, unsigned char * dst)
{
    return parseIPv4(src, nullptr, dst);
}

/** Unsafe (no bounds-checking for src nor dst), optimized version of parsing IPv6 string.
*
* Slightly altered implementation from http://svn.apache.org/repos/asf/apr/apr/trunk/network_io/unix/inet_pton.c
* Parses the input string `src` and stores binary big-endian value into buffer pointed by `dst`,
* which should be long enough. In case of failure zeroes
* IPV6_BINARY_LENGTH bytes of buffer pointed by `dst`.
*
* @param src     - beginning of the input string.
* @param src_end - optional, end of the input string, if nullptr string will be parsed until not valid symbol is met.
* @param dst     - where to put output bytes, expected to be non-null and at IPV6_BINARY_LENGTH-long.
* @return if success pointer to the address immediately after parsed sequence, nullptr otherwise.
*/
inline const char * parseIPv6(const char * src, const char * src_end, unsigned char * dst)
{
    const auto clear_dst = [dst]()
    {
        memset(dst, '\0', IPV6_BINARY_LENGTH);
        return nullptr;
    };

    if (src == nullptr || (src_end && src_end - src < 2))
        return clear_dst();

    /// Leading :: requires some special handling.
    if (*src == ':')
        if (*++src != ':')
            return clear_dst();

    unsigned char tmp[IPV6_BINARY_LENGTH]{};
    unsigned char * tp = tmp;
    unsigned char * endp = tp + IPV6_BINARY_LENGTH;
    const char * curtok = src;
    bool saw_xdigit = false;
    UInt32 val{};
    unsigned char * colonp = nullptr;

    /// Assuming zero-terminated string if src_size==0 or otherwise max src_size symbols
    for (; true; ++src)
    {
        if (src_end && src == src_end)
            break;

        UInt8 num = unhex(*src);

        if (num != 0xFF)
        {
            val <<= 4;
            val |= num;
            if (val > 0xffffu)
                return clear_dst();

            saw_xdigit = true;
            continue;
        }

        if (*src == ':')
        {
            curtok = src + 1;
            if (!saw_xdigit)
            {
                if (colonp)
                    return clear_dst();

                colonp = tp;
                continue;
            }

            if (tp + sizeof(UInt16) > endp)
                return clear_dst();

            *tp++ = static_cast<unsigned char>((val >> 8) & 0xffu);
            *tp++ = static_cast<unsigned char>(val & 0xffu);
            saw_xdigit = false;
            val = 0;
            continue;
        }

        if (*src == '.' && (tp + IPV4_BINARY_LENGTH) <= endp)
        {
            src = parseIPv4(curtok, src_end, tp);
            if (src == nullptr)
                return clear_dst();
            std::reverse(tp, tp + IPV4_BINARY_LENGTH);

            tp += IPV4_BINARY_LENGTH;
            saw_xdigit = false;
            break;    /* '\0' was seen by ipv4_scan(). */
        }

        break;
    }

    if (saw_xdigit)
    {
        if (tp + sizeof(UInt16) > endp)
            return clear_dst();

        *tp++ = static_cast<unsigned char>((val >> 8) & 0xffu);
        *tp++ = static_cast<unsigned char>(val & 0xffu);
    }

    if (colonp)
    {
        /*
         * Since some memmove()'s erroneously fail to handle
         * overlapping regions, we'll do the shift by hand.
         */
        const auto n = tp - colonp;

        for (int i = 1; i <= n; ++i)
        {
            endp[- i] = colonp[n - i];
            colonp[n - i] = 0;
        }
        tp = endp;
    }

    if (tp != endp)
        return clear_dst();

    memcpy(dst, tmp, sizeof(tmp));
    return src;
}

inline bool parseIPv6(const char * src, unsigned char * dst)
{
    return parseIPv6(src, nullptr, dst);
}

/** Format 4-byte binary sequesnce as IPv4 text: 'aaa.bbb.ccc.ddd',
  * expects in out to be in BE-format, that is 0x7f000001 => "127.0.0.1".
  *
  * Any number of the tail bytes can be masked with given mask string.
  *
  * Assumptions:
  *     src is IPV4_BINARY_LENGTH long,
  *     dst is IPV4_MAX_TEXT_LENGTH long,
  *     mask_tail_octets <= IPV4_BINARY_LENGTH
  *     mask_string is NON-NULL, if mask_tail_octets > 0.
  *
  * Examples:
  *     formatIPv4(&0x7f000001, dst, mask_tail_octets = 0, nullptr);
  *         > dst == "127.0.0.1"
  *     formatIPv4(&0x7f000001, dst, mask_tail_octets = 1, "xxx");
  *         > dst == "127.0.0.xxx"
  *     formatIPv4(&0x7f000001, dst, mask_tail_octets = 1, "0");
  *         > dst == "127.0.0.0"
  */
inline void formatIPv4(const unsigned char * src, char *& dst, uint8_t mask_tail_octets = 0, const char * mask_string = "xxx")
{
    extern const char one_byte_to_string_lookup_table[256][4];

    const size_t mask_length = mask_string ? strlen(mask_string) : 0;
    const size_t limit = std::min(IPV4_BINARY_LENGTH, IPV4_BINARY_LENGTH - mask_tail_octets);
    for (size_t octet = 0; octet < limit; ++octet)
    {
        const uint8_t value = static_cast<uint8_t>(src[IPV4_BINARY_LENGTH - octet - 1]);
        const auto * rep = one_byte_to_string_lookup_table[value];
        const uint8_t len = rep[0];
        const char* str = rep + 1;

        memcpy(dst, str, len);
        dst += len;
        *dst++ = '.';
    }

    for (size_t mask = 0; mask < mask_tail_octets; ++mask)
    {
        memcpy(dst, mask_string, mask_length);
        dst += mask_length;

        *dst++ = '.';
    }

    dst[-1] = '\0';
}

}
