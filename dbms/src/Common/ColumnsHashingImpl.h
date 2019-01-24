#pragma once

#include <Columns/IColumn.h>
#include <Interpreters/AggregationCommon.h>

namespace DB
{

namespace ColumnsHashing
{

namespace columns_hashing_impl
{

template <typename Value, bool consecutive_keys_optimization_>
struct LastElementCache
{
    static constexpr bool consecutive_keys_optimization = consecutive_keys_optimization_;
    Value value;
    bool empty = true;
    bool found = false;

    bool check(const Value & value_) { return !empty && value == value_; }

    template <typename Key>
    bool check(const Key & key) { return !empty && value.first == key; }
};

template <typename Data>
struct LastElementCache<Data, false>
{
    static constexpr bool consecutive_keys_optimization = false;
};

template <typename Mapped>
class EmplaceResultImpl
{
    Mapped & value;
    Mapped & cached_value;
    bool inserted;

public:
    EmplaceResultImpl(Mapped & value, Mapped & cached_value, bool inserted)
            : value(value), cached_value(cached_value), inserted(inserted) {}

    bool isInserted() const { return inserted; }
    auto & getMapped() const { return value; }
    void setMapped(const Mapped & mapped) { value = cached_value = mapped; }
};

template <>
class EmplaceResultImpl<void>
{
    bool inserted;

public:
    explicit EmplaceResultImpl(bool inserted) : inserted(inserted) {}
    bool isInserted() const { return inserted; }
};

template <typename Mapped>
class FindResultImpl
{
    Mapped * value;
    bool found;

public:
    FindResultImpl(Mapped * value, bool found) : value(value), found(found) {}
    bool isFound() const { return found; }
    Mapped & getMapped() const { return *value; }
};

template <>
class FindResultImpl<void>
{
    bool found;

public:
    explicit FindResultImpl(bool found) : found(found) {}
    bool isFound() const { return found; }
};

template <typename Value, typename Mapped, bool consecutive_keys_optimization>
struct HashMethodBase
{
    using EmplaceResult = EmplaceResultImpl<Mapped>;
    using FindResult = FindResultImpl<Mapped>;
    static constexpr bool has_mapped = !std::is_same<Mapped, void>::value;
    using Cache = LastElementCache<Value, consecutive_keys_optimization>;

protected:
    Cache cache;

    HashMethodBase()
    {
        if constexpr (has_mapped && consecutive_keys_optimization)
        {
            /// Init PairNoInit elements.
            cache.value.second = Mapped();
            using Key = decltype(cache.value.first);
            cache.value.first = Key();
        }
    }

    template <typename Data, typename Key>
    ALWAYS_INLINE EmplaceResult emplaceKeyImpl(Key key, Data & data, typename Data::iterator & it)
    {
        if constexpr (Cache::consecutive_keys_optimization)
        {
            if (cache.found && cache.check(key))
            {
                if constexpr (has_mapped)
                    return EmplaceResult(cache.value.second, cache.value.second, false);
                else
                    return EmplaceResult(false);
            }
        }

        bool inserted = false;
        data.emplace(key, it, inserted);
        Mapped * cached = &it->second;

        if constexpr (consecutive_keys_optimization)
        {
            cache.value = *it;
            cache.found = true;
            cache.empty = false;
            cached = &cache.value.second;
        }

        if constexpr (has_mapped)
            return EmplaceResult(it->second, *cached, inserted);
        else
            return EmplaceResult(inserted);
    }

    template <typename Data, typename Key>
    ALWAYS_INLINE FindResult findKeyImpl(Key key, Data & data)
    {
        if constexpr (Cache::consecutive_keys_optimization)
        {
            if (cache.check(key))
            {
                if constexpr (has_mapped)
                    return FindResult(&cache.value.second, cache.found);
                else
                    return FindResult(cache.found);
            }
        }

        auto it = data.find(key);
        bool found = it != data.end();

        if constexpr (consecutive_keys_optimization)
        {
            cache.found = found;
            cache.empty = false;

            if (found)
                cache.value = *it;
            else
            {
                if constexpr (has_mapped)
                    cache.value.first = key;
                else
                    cache.value = key;
            }
        }

        if constexpr (has_mapped)
            return FindResult(found ? &it->second : nullptr, found);
        else
            return FindResult(found);
    }
};


template <typename T>
struct MappedCache : public PaddedPODArray<T> {};

template <>
struct MappedCache<void> {};


/// This class is designed to provide the functionality that is required for
/// supporting nullable keys in HashMethodKeysFixed. If there are
/// no nullable keys, this class is merely implemented as an empty shell.
template <typename Key, bool has_nullable_keys>
class BaseStateKeysFixed;

/// Case where nullable keys are supported.
template <typename Key>
class BaseStateKeysFixed<Key, true>
{
protected:
    void init(const ColumnRawPtrs & key_columns)
    {
        null_maps.reserve(key_columns.size());
        actual_columns.reserve(key_columns.size());

        for (const auto & col : key_columns)
        {
            if (col->isColumnNullable())
            {
                const auto & nullable_col = static_cast<const ColumnNullable &>(*col);
                actual_columns.push_back(&nullable_col.getNestedColumn());
                null_maps.push_back(&nullable_col.getNullMapColumn());
            }
            else
            {
                actual_columns.push_back(col);
                null_maps.push_back(nullptr);
            }
        }
    }

    /// Return the columns which actually contain the values of the keys.
    /// For a given key column, if it is nullable, we return its nested
    /// column. Otherwise we return the key column itself.
    inline const ColumnRawPtrs & getActualColumns() const
    {
        return actual_columns;
    }

    /// Create a bitmap that indicates whether, for a particular row,
    /// a key column bears a null value or not.
    KeysNullMap<Key> createBitmap(size_t row) const
    {
        KeysNullMap<Key> bitmap{};

        for (size_t k = 0; k < null_maps.size(); ++k)
        {
            if (null_maps[k] != nullptr)
            {
                const auto & null_map = static_cast<const ColumnUInt8 &>(*null_maps[k]).getData();
                if (null_map[row] == 1)
                {
                    size_t bucket = k / 8;
                    size_t offset = k % 8;
                    bitmap[bucket] |= UInt8(1) << offset;
                }
            }
        }

        return bitmap;
    }

private:
    ColumnRawPtrs actual_columns;
    ColumnRawPtrs null_maps;
};

/// Case where nullable keys are not supported.
template <typename Key>
class BaseStateKeysFixed<Key, false>
{
protected:
    void init(const ColumnRawPtrs & columns) { actual_columns = columns; }

    const ColumnRawPtrs & getActualColumns() const { return actual_columns; }

    KeysNullMap<Key> createBitmap(size_t) const
    {
        throw Exception{"Internal error: calling createBitmap() for non-nullable keys"
                        " is forbidden", ErrorCodes::LOGICAL_ERROR};
    }

private:
    ColumnRawPtrs actual_columns;
};

}

}

}
