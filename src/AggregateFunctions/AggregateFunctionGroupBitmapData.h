#pragma once

#include <algorithm>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <boost/noncopyable.hpp>
#include <Common/HashTable/SmallTable.h>
#include <Common/PODArray.h>

// Include this header last, because it is an auto-generated dump of questionable
// garbage that breaks the build (e.g. it changes _POSIX_C_SOURCE).
// TODO: find out what it is. On github, they have proper interface headers like
// this one: https://github.com/RoaringBitmap/CRoaring/blob/master/include/roaring/roaring.h
#include <roaring/roaring.h>
#include "roaring.hh"
#include <roaring64map.hh>

namespace DB
{
/**
  * For a small number of values - an array of fixed size "on the stack".
  * For large, roaring_bitmap_t is allocated.
  * For a description of the roaring_bitmap_t, see: https://github.com/RoaringBitmap/CRoaring
  */
template <typename T, UInt8 small_set_size>
class RoaringBitmapWithSmallSet : private boost::noncopyable
{
private:
    using Small = SmallSet<T, small_set_size>;
    using ValueBuffer = std::vector<T>;
    bool bUInt64 = sizeof(T) >= 8;

    Small small;
    roaring_bitmap_t * rb = nullptr;
    std::shared_ptr<Roaring64Map> rb64 = nullptr;

    void toLarge()
    {
        if (isUInt64())
            rb64 = std::make_shared<Roaring64Map>();
        else
            rb = roaring_bitmap_create();

        for (const auto & x : small)
        {
            if (isUInt64())
                rb64->add(static_cast<UInt64>(x.getValue()));
            else
                roaring_bitmap_add(rb, x.getValue());
        }
    }

public:
    bool isLarge() const { return rb != nullptr || rb64 != nullptr; }

    bool isSmall() const { return rb == nullptr && rb64 == nullptr; }

    bool isUInt64() const { return bUInt64; }

    ~RoaringBitmapWithSmallSet()
    {
        if (isLarge() && !isUInt64())
            roaring_bitmap_free(rb);
    }

    void add(T value)
    {
        if (isSmall())
        {
            if (small.find(value) == small.end())
            {
                if (!small.full())
                    small.insert(value);
                else
                {
                    toLarge();
                    if (isUInt64())
                        rb64->add(static_cast<UInt64>(value));
                    else
                        roaring_bitmap_add(rb, value);
                }
            }
        }
        else
        {
            if (isUInt64())
                rb64->add(static_cast<UInt64>(value));
            else
                roaring_bitmap_add(rb, value);
        }
    }

    UInt64 size() const
    {
        if (isSmall())
        {
            return small.size();
        }
        else
        {
            if (isUInt64())
                return rb64->cardinality();
            else
                return roaring_bitmap_get_cardinality(rb);
        }
    }

    void merge(const RoaringBitmapWithSmallSet & r1)
    {
        if (r1.isLarge())
        {
            if (isSmall())
                toLarge();

            if (isUInt64())
                *rb64 |= *r1.rb64;
            else
                roaring_bitmap_or_inplace(rb, r1.rb);
        }
        else
        {
            for (const auto & x : r1.small)
                add(x.getValue());
        }
    }

    void read(DB::ReadBuffer & in)
    {
        // Container type:  0: SmallSet,  1: RoaringBitmap32,  2: RoaringBitmap64
        UInt8 containerType = 0;
        readBinary(containerType, in);

        if (0 == containerType)
        {
            small.read(in);
        }
        else if (1 == containerType)
        {
            std::string s;
            readStringBinary(s, in);
            rb = roaring_bitmap_portable_deserialize(s.c_str());
            // It has been persisted in the bitmap and does not need to merge from small
            //            for (const auto & x : small) // merge from small
            //                roaring_bitmap_add(rb, x.getValue());
        }
        else
        {
            std::string s;
            readStringBinary(s, in);
            rb64 = std::make_shared<Roaring64Map>(Roaring64Map::read(s.c_str()));
        }
    }

    void write(DB::WriteBuffer & out) const
    {
        // Container type:  0: SmallSet,  1: RoaringBitmap32,  2: RoaringBitmap64
        UInt8 containerType = 0;
        if (isLarge())
        {
            if (isUInt64())
                containerType = 2;
            else
                containerType = 1;
        }

        writeBinary(containerType, out);

        if (isLarge())
        {
            if (isUInt64())
            {
                uint32_t expectedsize = rb64->getSizeInBytes();
                std::string s(expectedsize, 0);
                rb64->write(const_cast<char *>(s.data()));
                writeStringBinary(s, out);
            }
            else
            {
                uint32_t expectedsize = roaring_bitmap_portable_size_in_bytes(rb);
                std::string s(expectedsize, 0);
                roaring_bitmap_portable_serialize(rb, const_cast<char *>(s.data()));
                writeStringBinary(s, out);
            }
        }
        else
            small.write(out);
    }

    roaring_bitmap_t * getRb() const { return rb; }

    std::shared_ptr<Roaring64Map> getRb64() const { return rb64; }

    Small & getSmall() const { return small; }

    /**
     * Get a new roaring_bitmap_t from elements of small
     */
    roaring_bitmap_t * getNewRbFromSmall() const
    {
        roaring_bitmap_t * smallRb = roaring_bitmap_create();
        for (const auto & x : small)
            roaring_bitmap_add(smallRb, x.getValue());
        return smallRb;
    }

    /**
     * Get a new Roaring64Map from elements of small
     */
    std::shared_ptr<Roaring64Map> getNewRb64FromSmall() const
    {
        std::shared_ptr<Roaring64Map> smallRb64 = std::make_shared<Roaring64Map>();
        for (const auto & x : small)
            smallRb64->add(static_cast<UInt64>(x.getValue()));
        return smallRb64;
    }

    /**
     * Computes the intersection between two bitmaps
     */
    void rb_and(const RoaringBitmapWithSmallSet & r1)
    {
        ValueBuffer buffer;
        if (isSmall() && r1.isSmall())
        {
            // intersect
            for (const auto & x : small)
                if (r1.small.find(x.getValue()) != r1.small.end())
                    buffer.push_back(x.getValue());

            // Clear out the original values
            small.clear();

            for (const auto & value : buffer)
                small.insert(value);

            buffer.clear();
        }
        else if (isSmall() && r1.isLarge())
        {
            for (const auto & x : small)
            {
                if (isUInt64())
                {
                    if (rb64->contains(static_cast<UInt64>(x.getValue())))
                        buffer.push_back(x.getValue());
                }
                else
                {
                    if (roaring_bitmap_contains(r1.rb, x.getValue()))
                        buffer.push_back(x.getValue());
                }
            }

            // Clear out the original values
            small.clear();

            for (const auto & value : buffer)
                small.insert(value);

            buffer.clear();
        }
        else
        {
            if (isUInt64())
            {
                std::shared_ptr<Roaring64Map> newRb64 = r1.isSmall() ? r1.getNewRb64FromSmall() : r1.getRb64();
                *rb64 &= *newRb64;
            }
            else
            {
                roaring_bitmap_t * rb1 = r1.isSmall() ? r1.getNewRbFromSmall() : r1.getRb();
                roaring_bitmap_and_inplace(rb, rb1);
                if (r1.isSmall())
                    roaring_bitmap_free(rb1);
            }
        }
    }

    /**
     * Computes the union between two bitmaps.
     */
    void rb_or(const RoaringBitmapWithSmallSet & r1) { merge(r1); }

    /**
     * Computes the symmetric difference (xor) between two bitmaps.
     */
    void rb_xor(const RoaringBitmapWithSmallSet & r1)
    {
        if (isSmall())
            toLarge();

        if (isUInt64())
        {
            std::shared_ptr<Roaring64Map> newRb64 = r1.isSmall() ? r1.getNewRb64FromSmall() : r1.getRb64();
            *rb64 ^= *newRb64;
        }
        else
        {
            roaring_bitmap_t * rb1 = r1.isSmall() ? r1.getNewRbFromSmall() : r1.getRb();
            roaring_bitmap_xor_inplace(rb, rb1);
            if (r1.isSmall())
                roaring_bitmap_free(rb1);
        }
    }

    /**
     * Computes the difference (andnot) between two bitmaps
     */
    void rb_andnot(const RoaringBitmapWithSmallSet & r1)
    {
        ValueBuffer buffer;
        if (isSmall() && r1.isSmall())
        {
            // subtract
            for (const auto & x : small)
                if (r1.small.find(x.getValue()) == r1.small.end())
                    buffer.push_back(x.getValue());

            // Clear out the original values
            small.clear();

            for (const auto & value : buffer)
                small.insert(value);

            buffer.clear();
        }
        else if (isSmall() && r1.isLarge())
        {
            for (const auto & x : small)
            {
                if (isUInt64())
                {
                    if (!rb64->contains(static_cast<UInt64>(x.getValue())))
                        buffer.push_back(x.getValue());
                }
                else
                {
                    if (!roaring_bitmap_contains(r1.rb, x.getValue()))
                        buffer.push_back(x.getValue());
                }
            }

            // Clear out the original values
            small.clear();

            for (const auto & value : buffer)
                small.insert(value);

            buffer.clear();
        }
        else
        {
            if (isUInt64())
            {
                std::shared_ptr<Roaring64Map> newRb64 = r1.isSmall() ? r1.getNewRb64FromSmall() : r1.getRb64();
                *rb64 -= *newRb64;
            }
            else
            {
                roaring_bitmap_t * rb1 = r1.isSmall() ? r1.getNewRbFromSmall() : r1.getRb();
                roaring_bitmap_andnot_inplace(rb, rb1);
                if (r1.isSmall())
                    roaring_bitmap_free(rb1);
            }
        }
    }

    /**
     * Computes the cardinality of the intersection between two bitmaps.
     */
    UInt64 rb_and_cardinality(const RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 retSize = 0;
        if (isSmall() && r1.isSmall())
        {
            for (const auto & x : small)
                if (r1.small.find(x.getValue()) != r1.small.end())
                    ++retSize;
        }
        else if (isSmall() && r1.isLarge())
        {
            for (const auto & x : small)
            {
                if (isUInt64())
                {
                    if (rb64->contains(static_cast<UInt64>(x.getValue())))
                        ++retSize;
                }
                else
                {
                    if (roaring_bitmap_contains(r1.rb, x.getValue()))
                        ++retSize;
                }
            }
        }
        else
        {
            if (isUInt64())
            {
                std::shared_ptr<Roaring64Map> newRb64 = r1.isSmall() ? r1.getNewRb64FromSmall() : r1.getRb64();
                retSize = (*rb64 & *newRb64).cardinality();
            }
            else
            {
                roaring_bitmap_t * rb1 = r1.isSmall() ? r1.getNewRbFromSmall() : r1.getRb();
                retSize = roaring_bitmap_and_cardinality(rb, rb1);
                if (r1.isSmall())
                    roaring_bitmap_free(rb1);
            }
        }
        return retSize;
    }

    /**
     * Computes the cardinality of the union between two bitmaps.
    */
    UInt64 rb_or_cardinality(const RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 c1 = size();
        UInt64 c2 = r1.size();
        UInt64 inter = rb_and_cardinality(r1);
        return c1 + c2 - inter;
    }

    /**
     * Computes the cardinality of the symmetric difference (andnot) between two bitmaps.
    */
    UInt64 rb_xor_cardinality(const RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 c1 = size();
        UInt64 c2 = r1.size();
        UInt64 inter = rb_and_cardinality(r1);
        return c1 + c2 - 2 * inter;
    }

    /**
     * Computes the cardinality of the difference (andnot) between two bitmaps.
     */
    UInt64 rb_andnot_cardinality(const RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 c1 = size();
        UInt64 inter = rb_and_cardinality(r1);
        return c1 - inter;
    }

    /**
     * Return 1 if the two bitmaps contain the same elements.
     */
    UInt8 rb_equals(const RoaringBitmapWithSmallSet & r1)
    {
        if (isSmall())
            toLarge();

        UInt8 is_true = 0;
        if (isUInt64())
        {
            std::shared_ptr<Roaring64Map> newRb64 = r1.isSmall() ? r1.getNewRb64FromSmall() : r1.getRb64();
            is_true = *rb64 == *newRb64;
        }
        else
        {
            roaring_bitmap_t * rb1 = r1.isSmall() ? r1.getNewRbFromSmall() : r1.getRb();
            is_true = roaring_bitmap_equals(rb, rb1);
            if (r1.isSmall())
                roaring_bitmap_free(rb1);
        }

        return is_true;
    }

    /**
     * Check whether two bitmaps intersect.
     * Intersection with an empty set is always 0 (consistent with hasAny).
     */
    UInt8 rb_intersect(const RoaringBitmapWithSmallSet & r1) const
    {
        if (isSmall())
        {
            if (r1.isSmall())
            {
                for (const auto & x : r1.small)
                    if (small.find(x.getValue()) != small.end())
                        return 1;
            }
            else
            {
                for (const auto & x : small)
                {
                    if (isUInt64())
                    {
                        if (r1.rb64->contains(static_cast<UInt64>(x.getValue())))
                            return 1;
                    }
                    else
                    {
                        if (roaring_bitmap_contains(r1.rb, x.getValue()))
                            return 1;
                    }
                }
            }
        }
        else if (r1.isSmall())
        {
            for (const auto & x : r1.small)
            {
                if (isUInt64())
                {
                    if (rb64->contains(static_cast<UInt64>(x.getValue())))
                        return 1;
                }
                else
                {
                    if (roaring_bitmap_contains(rb, x.getValue()))
                        return 1;
                }
            }
        }
        else
        {
            if (isUInt64())
            {
                if ((*rb64 & *r1.rb64).cardinality() > 0)
                    return 1;
            }
            else
            {
                if (roaring_bitmap_intersect(rb, r1.rb))
                    return 1;
            }
        }

        return 0;
    }

    /**
     * Check whether the argument is the subset of this set.
     * Empty set is a subset of any other set (consistent with hasAll).
     */
    UInt8 rb_is_subset(const RoaringBitmapWithSmallSet & r1) const
    {
        if (isSmall())
        {
            if (r1.isSmall())
            {
                for (const auto & x : r1.small)
                    if (small.find(x.getValue()) == small.end())
                        return 0;
            }
            else
            {
                UInt64 r1_size = r1.size();

                if (r1_size > small.size())
                    return 0; // A bigger set can not be a subset of ours.

                // This is a rare case with a small number of elements on
                // both sides: r1 was promoted to large for some reason and
                // it is still not larger than our small set.
                // If r1 is our subset then our size must be equal to
                // r1_size + number of not found elements, if this sum becomes
                // greater then r1 is not a subset.
                for (const auto & x : small)
                {
                    if (isUInt64())
                    {
                        if (!r1.rb64->contains(static_cast<UInt64>(x.getValue())) && ++r1_size > small.size())
                            return 0;
                    }
                    else
                    {
                        if (!roaring_bitmap_contains(r1.rb, x.getValue()) && ++r1_size > small.size())
                            return 0;
                    }
                }
            }
        }
        else if (r1.isSmall())
        {
            for (const auto & x : r1.small)
            {
                if (isUInt64())
                {
                    if (!rb64->contains(static_cast<UInt64>(x.getValue())))
                        return 0;
                }
                else
                {
                    if (!roaring_bitmap_contains(rb, x.getValue()))
                        return 0;
                }
            }
        }
        else
        {
            if (isUInt64())
            {
                if (!r1.rb64->isSubset(*rb64))
                    return 0;
            }
            else
            {
                if (!roaring_bitmap_is_subset(r1.rb, rb))
                    return 0;
            }
        }
        return 1;
    }

    /**
     * Check whether this bitmap contains the argument.
     */
    UInt8 rb_contains(const UInt64 x) const
    {
        if (isSmall())
        {
            return small.find(x) != small.end();
        }
        else
        {
            if (isUInt64())
            {
                return rb64->contains(x);
            }
            else
            {
                return roaring_bitmap_contains(rb, x);
            }
        }
    }

    /**
     * Remove value
     */
    void rb_remove(UInt64 offsetid)
    {
        if (isSmall())
            toLarge();

        if (isUInt64())
            rb64->remove(offsetid);
        else
            roaring_bitmap_remove(rb, offsetid);
    }

    /**
     * compute (in place) the negation of the roaring bitmap within a specified
     * interval: [range_start, range_end). The number of negated values is
     * range_end - range_start.
     * Areas outside the range are passed through unchanged.
     */
    void rb_flip(UInt64 offsetstart, UInt64 offsetend)
    {
        if (isSmall())
            toLarge();
        if (isUInt64())
            rb64->flip(offsetstart, offsetend);
        else
            roaring_bitmap_flip_inplace(rb, offsetstart, offsetend);
    }

    /**
     * returns the number of integers that are smaller or equal to offsetid.
     */
    UInt64 rb_rank(UInt64 offsetid)
    {
        if (isSmall())
            toLarge();

        if (isUInt64())
            return rb64->rank(offsetid);
        else
            return roaring_bitmap_rank(rb, offsetid);
    }

    /**
     * Convert elements to integer array, return number of elements
     */
    template <typename Element>
    UInt64 rb_to_array(PaddedPODArray<Element> & res_data) const
    {
        UInt64 count = 0;
        if (isSmall())
        {
            for (const auto & x : small)
            {
                res_data.emplace_back(x.getValue());
                count++;
            }
        }
        else
        {
            if (isUInt64())
            {
                for (Roaring64Map::const_iterator iterator = rb64->begin(); iterator != rb64->end(); iterator++)
                {
                    res_data.emplace_back(*iterator);
                    count++;
                }
            }
            else
            {
                roaring_uint32_iterator_t iterator;
                roaring_init_iterator(rb, &iterator);
                while (iterator.has_value)
                {
                    res_data.emplace_back(iterator.current_value);
                    roaring_advance_uint32_iterator(&iterator);
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Return new set with specified range (not include the range_end)
     */
    UInt64 rb_range(UInt64 range_start, UInt64 range_end, RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 count = 0;
        if (range_start >= range_end)
            return count;
        if (isSmall())
        {
            for (const auto & x : small)
            {
                T val = x.getValue();
                if (UInt32(val) >= range_start && UInt32(val) < range_end)
                {
                    r1.add(val);
                    ++count;
                }
            }
        }
        else
        {
            if (isUInt64())
            {
                for (Roaring64Map::const_iterator iterator = rb64->begin(); iterator != rb64->end(); iterator++)
                {
                    if (*iterator < range_start)
                        continue;

                    if (*iterator < range_end)
                    {
                        r1.add(*iterator);
                        ++count;
                    }
                    else
                        break;
                }
            }
            else
            {
                roaring_uint32_iterator_t iterator;
                roaring_init_iterator(rb, &iterator);
                roaring_move_uint32_iterator_equalorlarger(&iterator, range_start);
                while (iterator.has_value && UInt32(iterator.current_value) < range_end)
                {
                    r1.add(iterator.current_value);
                    roaring_advance_uint32_iterator(&iterator);
                    ++count;
                }
            }
        }
        return count;
    }

    /**
     * Return new set of the smallest `limit` values in set which is no less than `range_start`.
     */
    UInt64 rb_limit(UInt64 range_start, UInt64 limit, RoaringBitmapWithSmallSet & r1) const
    {
        UInt64 count = 0;
        if (isSmall())
        {
            std::vector<T> answer;
            for (const auto & x : small)
            {
                T val = x.getValue();
                if (UInt32(val) >= range_start)
                {
                    answer.push_back(val);
                }
            }
            sort(answer.begin(), answer.end());
            if (limit > answer.size())
                limit = answer.size();
            for (size_t i = 0; i < limit; ++i)
                r1.add(answer[i]);
            count = UInt64(limit);
        }
        else
        {
            if (isUInt64())
            {
                for (Roaring64Map::const_iterator iterator = rb64->begin(); iterator != rb64->end(); iterator++)
                {
                    if (*iterator < range_start)
                        continue;

                    if (count < limit)
                    {
                        r1.add(*iterator);
                        ++count;
                    }
                    else
                        break;
                }
            }
            else
            {
                roaring_uint32_iterator_t iterator;
                roaring_init_iterator(rb, &iterator);
                roaring_move_uint32_iterator_equalorlarger(&iterator, range_start);
                while (UInt32(count) < limit && iterator.has_value)
                {
                    r1.add(iterator.current_value);
                    roaring_advance_uint32_iterator(&iterator);
                    ++count;
                }
            }
        }
        return count;
    }

    UInt64 rb_min() const
    {
        UInt64 min_val = UINT32_MAX;
        if (isSmall())
        {
            for (const auto & x : small)
            {
                T val = x.getValue();
                if (UInt64(val) < min_val)
                {
                    min_val = UInt64(val);
                }
            }
        }
        else
        {
            if (isUInt64())
                min_val = rb64->minimum();
            else
                min_val = UInt64(roaring_bitmap_minimum(rb));
        }
        return min_val;
    }

    UInt64 rb_max() const
    {
        UInt64 max_val = 0;
        if (isSmall())
        {
            for (const auto & x : small)
            {
                T val = x.getValue();
                if (UInt64(val) > max_val)
                {
                    max_val = UInt64(val);
                }
            }
        }
        else
        {
            if (isUInt64())
                max_val = rb64->maximum();
            else
                max_val = UInt64(roaring_bitmap_maximum(rb));
        }
        return max_val;
    }

    /**
     * Replace value
     */
    void rb_replace(const UInt32 * from_vals, const UInt32 * to_vals, size_t num)
    {
        if (isSmall())
            toLarge();
        for (size_t i = 0; i < num; ++i)
        {
            if (from_vals[i] == to_vals[i])
                continue;

            if (isUInt64())
            {
                bool changed = rb64->removeChecked(from_vals[i]);
                if (changed)
                    rb64->add(to_vals[i]);
            }
            else
            {
                bool changed = roaring_bitmap_remove_checked(rb, from_vals[i]);
                if (changed)
                    roaring_bitmap_add(rb, to_vals[i]);
            }
        }
    }

private:
 
};

template <typename T>
struct AggregateFunctionGroupBitmapData
{
    bool doneFirst = false;
    RoaringBitmapWithSmallSet<T, 32> rbs;
    static const char * name() { return "groupBitmap"; }
};


}
