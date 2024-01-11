#pragma once

#include <memory> // for std::unique_ptr
#include <cmath>
#pragma once

#include <memory> // for std::unique_ptr
#include <cmath>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <base/types.h>

#include <IO/ReadBuffer.h>
#include <IO/WriteBuffer.h>

// We start with 128 bins and grow the number of bins by 128
// each time we need to extend the range of the bins.
// This is done to avoid reallocating the bins vector too often.
constexpr UInt64 chunk_size = 128;

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int INCORRECT_DATA;
}

struct DDSketchBins
{
    Poco::Logger * logger = &Poco::Logger::get("DDSketch");
    struct BinRange
    {
        Int64 min_key = INT64_MAX;
        Int64 max_key = INT64_MIN;
        Int64 offset = 0;
    };

    /// Avoid memory allocations for very small sketches.
    PODArrayWithStackMemory<Float64, 1024> bins;
    BinRange range;
    Float64 count = 0;

    DDSketchBins() : bins(128) {}

    void add(Int64 key, Float64 weight)
    {
        Int64 idx = getIndex(key);
        bins[idx] += weight;
        count += weight;
    }

    Int64 keyAtRank(Float64 rank, bool lower) const
    {
        Float64 running_ct = 0.0;
        for (size_t i = 0; i < bins.size(); ++i)
        {
            running_ct += bins[i];
            if ((lower && running_ct > rank) || (!lower && running_ct >= rank + 1))
            {
                return static_cast<Int64>(i) + range.offset;
            }
        }
        return range.max_key;
    }

    Int64 getIndex(Int64 key)
    {
        if (key < range.min_key || key > range.max_key)
        {
            extendRange(key, key);
        }
        return key - range.offset;
    }

    UInt64 getNewLength(Int64 new_min_key, Int64 new_max_key)
    {
        Int64 desired_length = new_max_key - new_min_key + 1;
        return static_cast<UInt64>(chunk_size * std::ceil(static_cast<Float64>(desired_length) / chunk_size)); // Fixed float conversion
    }

    void extendRange(Int64 min_key, Int64 max_key)
    {
        Int64 new_min_key = std::min(min_key, range.min_key);
        Int64 new_max_key = std::max(max_key, range.max_key);

        if (new_min_key >= range.offset && new_max_key < range.offset + static_cast<Int64>(bins.size()))
        {
            range.min_key = new_min_key;
            range.max_key = new_max_key;
        }
        else
        {
            UInt64 new_length = getNewLength(new_min_key, new_max_key);
            std::size_t old_size = bins.size();
            bins.resize(new_length);
            // fill the new bins with zeros
            std::fill(bins.begin() + old_size, bins.end(), 0.0);
            adjust(new_min_key, new_max_key);
        }
    }

    void adjust(Int64 new_min_key, Int64 new_max_key)
    {
        centerBins(new_min_key, new_max_key);
        range.min_key = new_min_key;
        range.max_key = new_max_key;
    }

    void shiftBins(Int64 shift)
    {
        Int64 new_offset = range.offset - shift;
        if (new_offset > range.offset)
            std::rotate(bins.begin(), bins.begin() + (new_offset - range.offset) % bins.size(), bins.end());
        else
            std::rotate(bins.begin(), bins.end() - (range.offset - new_offset) % bins.size(), bins.end());
        range.offset = new_offset;
    }

    void centerBins(Int64 new_min_key, Int64 new_max_key)
    {
        Int64 margins = static_cast<Int64>(bins.size()) - (new_max_key - new_min_key + 1);
        Int64 new_offset = new_min_key - margins / 2;
        shiftBins(range.offset - new_offset);
    }

    void merge(const DDSketchBins& other)
    {
        if (other.range.min_key < range.min_key || other.range.max_key > range.max_key)
        {
            extendRange(other.range.min_key, other.range.max_key);
        }
        for (size_t i = 0; i < other.bins.size(); ++i)
        {
            bins[i] += other.bins[i];
        }
        count += other.count;
    }

    void serialize(WriteBuffer& buf) const
    {
        writeBinary(range.min_key, buf);
        writeBinary(range.max_key, buf);
        writeBinary(range.offset, buf);
        writeBinary(count, buf);
        writeBinary(bins.size(), buf);
        for (size_t i = 0; i < bins.size(); ++i)
        {
            writeBinary(bins[i], buf);
        }
    }

    void deserialize(ReadBuffer& buf)
    {
        readBinary(range.min_key, buf);
        readBinary(range.max_key, buf);
        readBinary(range.offset, buf);
        readBinary(count, buf);
        UInt64 bins_size;
        readBinary(bins_size, buf);
        bins.resize(bins_size);
        for (size_t i = 0; i < bins.size(); ++i)
        {
            readBinary(bins[i], buf);
        }
    }
};

struct DDSketchNew
{
    Poco::Logger * logger = &Poco::Logger::get("DDSketch");

    DDSketchNew(double s = 0.01) : relative_accuracy(s) {
        gamma = (1 + s) / (1 - s);
        multiplier = 1 / std::log(gamma);
        min_possible = std::numeric_limits<Float64>::min() * gamma;
        max_possible = std::numeric_limits<Float64>::max() / gamma;
    }

    DDSketchBins positive_bins;
    DDSketchBins negative_bins;
    Float64 zero_count;
    Float64 count;
    Float64 relative_accuracy;
    Float64 gamma;
    Float64 min_possible;
    Float64 max_possible;
    Float64 multiplier;
    Float64 offset;

    Int64 key(Float64 value) const
    {
        if (value < min_possible || value > max_possible)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Value {} is out of range [{}, {}]", value, min_possible, max_possible);
        }
        return static_cast<Int64>(logGamma(value) + offset);
    }

    Float64 to_value(Int64 key) const
    {
        return lowerBound(key) * (1 + relative_accuracy);
    }

    Float64 logGamma(Float64 value) const
    {
        return std::log(value) * multiplier;
    }

    Float64 powGamma(Float64 value) const
    {
        return std::exp(value / multiplier);
    }

    Float64 lowerBound(Int64 index) const
    {
        return powGamma(static_cast<Float64>(index) - offset);
    }


    void add(Float64 val, UInt64 weight)
    {
        if (weight <= 0.0)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "weight must be a positive Float64");
        }

        if (val > min_possible)
        {
            positive_bins.add(key(val), weight);
        }
        else if (val < -min_possible)
        {
            negative_bins.add(key(-val), weight);
        }
        else
        {
            zero_count += weight;
        }

        count += weight;
    }

    Float64 get(Float64 quantile) const
    {
        if (quantile < 0 || quantile > 1 || count == 0)
        {
            return std::numeric_limits<Float64>::quiet_NaN(); // Return NaN if the conditions are not met
        }

        Float64 rank = quantile * (count - 1);
        Float64 quantile_value;
        LOG_INFO(logger, "DDSketchStruct::get rank: {}, negative_bins.count: {}", toString(rank), toString(negative_bins.count));
        if (rank < negative_bins.count)
        {
            Float64 reversed_rank = negative_bins.count - rank - 1;
            Int64 key = negative_bins.keyAtRank(reversed_rank, false);
            quantile_value = to_value(key);
        }
        else if (rank < zero_count + negative_bins.count)
        {
            quantile_value = 0;
        }
        else
        {
            Int64 key = positive_bins.keyAtRank(rank - zero_count - negative_bins.count, true);
            quantile_value = to_value(key);
        }
        return quantile_value;
    }

    void merge(const DDSketchNew& other)
    {
        positive_bins.merge(other.positive_bins);
        negative_bins.merge(other.negative_bins);
        zero_count += other.zero_count;
        count += other.count;
    }

    void serialize(WriteBuffer& buf) const
    {
        writeBinary(relative_accuracy, buf);
        writeBinary(count, buf);
        writeBinary(zero_count, buf);
        positive_bins.serialize(buf);
        negative_bins.serialize(buf);
    }

    void deserialize(ReadBuffer& buf)
    {
        readBinary(relative_accuracy, buf);
        readBinary(count, buf);
        readBinary(zero_count, buf);
        positive_bins.deserialize(buf);
        negative_bins.deserialize(buf);
    }
};


}
