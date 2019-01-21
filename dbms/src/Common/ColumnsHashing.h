#pragma once


#include <Common/ColumnsHashingImpl.h>
#include <Common/Arena.h>
#include <Common/LRUCache.h>
#include <common/unaligned.h>

#include <Columns/ColumnString.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnLowCardinality.h>

#include <Core/Defines.h>
#include <memory>

namespace DB
{

namespace ColumnsHashing
{

/// Generic context for HashMethod. Context is shared between multiple threads, all methods must be thread-safe.
/// Is used for caching.
class HashMethodContext
{
public:
    virtual ~HashMethodContext() = default;

    struct Settings
    {
        size_t max_threads;
    };
};

using HashMethodContextPtr = std::shared_ptr<HashMethodContext>;


/// For the case where there is one numeric key.
template <typename Value, typename Mapped, typename FieldType>    /// UInt8/16/32/64 for any type with corresponding bit width.
struct HashMethodOneNumber : public columns_hashing_impl::HashMethodBase<Value, Mapped, true>
{
    using Base = columns_hashing_impl::HashMethodBase<Value, Mapped, true>;
    const char * vec;

    /// If the keys of a fixed length then key_sizes contains their lengths, empty otherwise.
    HashMethodOneNumber(const ColumnRawPtrs & key_columns, const Sizes & /*key_sizes*/, const HashMethodContextPtr &)
    {
        vec = key_columns[0]->getRawData().data;
    }

    /// Creates context. Method is called once and result context is used in all threads.
    static HashMethodContextPtr createContext(const HashMethodContext::Settings &) { return nullptr; }

    FieldType getKey(size_t row) const { return unalignedLoad<FieldType>(vec + row * sizeof(FieldType)); }

    /// Emplace key into HashTable or HashMap. If Data is HashMap, returns ptr to value, otherwise nullptr.
    template <typename Data>
    ALWAYS_INLINE typename Base::EmplaceResult emplaceKey(
        Data & data, /// HashTable
        size_t row, /// From which row of the block insert the key
        Arena & /*pool*/) /// For Serialized method, key may be placed in pool.
    {
        typename Data::iterator it;
        return Base::emplaceKeyImpl(getKey(row), data, it);
    }

    /// Find key into HashTable or HashMap. If Data is HashMap and key was found, returns ptr to value, otherwise nullptr.
    template <typename Data>
    ALWAYS_INLINE typename Base::FindResult findKey(Data & data, size_t row, Arena & /*pool*/)
    {
        return Base::findKeyImpl(getKey(row), data);
    }

    /// Get hash value of row.
    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & /*pool*/)
    {
        return data.hash(getKey(row));
    }

    /// Get StringRef from value which can be inserted into column.
    static StringRef getValueRef(const Value & value)
    {
        return StringRef(reinterpret_cast<const char *>(&value.first), sizeof(value.first));
    }

protected:
    static ALWAYS_INLINE void onNewKey(Value & /*value*/, Arena & /*pool*/) {}
};


/// For the case where there is one string key.
template <typename Value, typename Mapped>
struct HashMethodString : public columns_hashing_impl::HashMethodBase<Value, Mapped, true>
{
    using Base = columns_hashing_impl::HashMethodBase<Value, Mapped, true>;
    const IColumn::Offset * offsets;
    const UInt8 * chars;

    HashMethodString(const ColumnRawPtrs & key_columns, const Sizes & /*key_sizes*/, const HashMethodContextPtr &)
    {
        const IColumn & column = *key_columns[0];
        const ColumnString & column_string = static_cast<const ColumnString &>(column);
        offsets = column_string.getOffsets().data();
        chars = column_string.getChars().data();
    }

    static HashMethodContextPtr createContext(const HashMethodContext::Settings &) { return nullptr; }

    StringRef getKey(size_t row) const { return StringRef(chars + offsets[row - 1], offsets[row] - offsets[row - 1] - 1); }

    template <typename Data>
    ALWAYS_INLINE typename Base::EmplaceResult emplaceKey(Data & data, size_t row, Arena & pool)
    {
        auto key = getKey(row);
        typename Data::iterator it;
        auto result = Base::emplaceKeyImpl(key, data, it);
        if (result.isInserted())
        {
            if (key.size)
                it->first.data = pool.insert(key.data, key.size);
        }
        return result;
    }

    template <typename Data>
    ALWAYS_INLINE typename Base::FindResult findKey(Data & data, size_t row, Arena & /*pool*/)
    {
        return Base::findKeyImpl(getKey(row), data);
    }

    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & /*pool*/)
    {
        return data.hash(getKey(row));
    }

    static StringRef getValueRef(const Value & value)
    {
        return StringRef(value.first.data, value.first.size);
    }

protected:
    static ALWAYS_INLINE void onNewKey(Value & value, Arena & pool)
    {
        if (value.first.size)
            value.first.data = pool.insert(value.first.data, value.first.size);
    }
};


/// For the case where there is one fixed-length string key.
template <typename Value, typename Mapped>
struct HashMethodFixedString : public columns_hashing_impl::HashMethodBase<Value, Mapped, true>
{
    using Base = columns_hashing_impl::HashMethodBase<Value, Mapped, true>;
    size_t n;
    const ColumnFixedString::Chars * chars;

    HashMethodFixedString(const ColumnRawPtrs & key_columns, const Sizes & /*key_sizes*/, const HashMethodContextPtr &)
    {
        const IColumn & column = *key_columns[0];
        const ColumnFixedString & column_string = static_cast<const ColumnFixedString &>(column);
        n = column_string.getN();
        chars = &column_string.getChars();
    }

    static HashMethodContextPtr createContext(const HashMethodContext::Settings &) { return nullptr; }

    StringRef getKey(size_t row) const { return StringRef(&(*chars)[row * n], n); }

    template <typename Data>
    ALWAYS_INLINE typename Base::EmplaceResult emplaceKey(Data & data, size_t row, Arena & pool)
    {
        auto key = getKey(row);
        typename Data::iterator it;
        auto res = Base::emplaceKeyImpl(key, data, it);
        if (res.isInserted())
            it->first.data = pool.insert(key.data, key.size);

        return res;
    }

    template <typename Data>
    ALWAYS_INLINE typename Base::FindResult findKey(Data & data, size_t row, Arena & /*pool*/)
    {
        return Base::findKeyImpl(getKey(row), data);
    }

    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & /*pool*/)
    {
        return data.hash(getKey(row));
    }

    static StringRef getValueRef(const Value & value)
    {
        return StringRef(value.first.data, value.first.size);
    }

protected:
    static ALWAYS_INLINE void onNewKey(Value & value, Arena & pool)
    {
        value.first.data = pool.insert(value.first.data, value.first.size);
    }
};


/// Cache stores dictionaries and saved_hash per dictionary key.
class LowCardinalityDictionaryCache : public HashMethodContext
{
public:
    /// Will assume that dictionaries with same hash has the same keys.
    /// Just in case, check that they have also the same size.
    struct DictionaryKey
    {
        UInt128 hash;
        UInt64 size;

        bool operator== (const DictionaryKey & other) const { return hash == other.hash && size == other.size; }
    };

    struct DictionaryKeyHash
    {
        size_t operator()(const DictionaryKey & key) const
        {
            SipHash hash;
            hash.update(key.hash.low);
            hash.update(key.hash.high);
            hash.update(key.size);
            return hash.get64();
        }
    };

    struct CachedValues
    {
        /// Store ptr to dictionary to be sure it won't be deleted.
        ColumnPtr dictionary_holder;
        /// Hashes for dictionary keys.
        const UInt64 * saved_hash = nullptr;
    };

    using CachedValuesPtr = std::shared_ptr<CachedValues>;

    explicit LowCardinalityDictionaryCache(const HashMethodContext::Settings & settings) : cache(settings.max_threads) {}

    CachedValuesPtr get(const DictionaryKey & key) { return cache.get(key); }
    void set(const DictionaryKey & key, const CachedValuesPtr & mapped) { cache.set(key, mapped); }

private:
    using Cache = LRUCache<DictionaryKey, CachedValues, DictionaryKeyHash>;
    Cache cache;
};


/// Single low cardinality column.
template <typename SingleColumnMethod, typename Mapped, bool use_cache>
struct HashMethodSingleLowCardinalityColumn : public SingleColumnMethod
{
    using Base = SingleColumnMethod;

    enum class VisitValue
    {
        Empty = 0,
        Found = 1,
        NotFound = 2,
    };

    static constexpr bool has_mapped = !std::is_same<Mapped, void>::value;
    using EmplaceResult = columns_hashing_impl::EmplaceResultImpl<Mapped>;
    using FindResult = columns_hashing_impl::FindResultImpl<Mapped>;

    static HashMethodContextPtr createContext(const HashMethodContext::Settings & settings)
    {
        return std::make_shared<LowCardinalityDictionaryCache>(settings);
    }

    ColumnRawPtrs key_columns;
    const IColumn * positions = nullptr;
    size_t size_of_index_type = 0;

    /// saved hash is from current column or from cache.
    const UInt64 * saved_hash = nullptr;
    /// Hold dictionary in case saved_hash is from cache to be sure it won't be deleted.
    ColumnPtr dictionary_holder;

    /// Cache AggregateDataPtr for current column in order to decrease the number of hash table usages.
    columns_hashing_impl::MappedCache<Mapped> mapped_cache;
    PaddedPODArray<VisitValue> visit_cache;

    /// If initialized column is nullable.
    bool is_nullable = false;

    static const ColumnLowCardinality & getLowCardinalityColumn(const IColumn * low_cardinality_column)
    {
        auto column = typeid_cast<const ColumnLowCardinality *>(low_cardinality_column);
        if (!column)
            throw Exception("Invalid aggregation key type for HashMethodSingleLowCardinalityColumn method. "
                            "Excepted LowCardinality, got " + column->getName(), ErrorCodes::LOGICAL_ERROR);
        return *column;
    }

    HashMethodSingleLowCardinalityColumn(
        const ColumnRawPtrs & key_columns_low_cardinality, const Sizes & key_sizes, const HashMethodContextPtr & context)
        : Base({getLowCardinalityColumn(key_columns_low_cardinality[0]).getDictionary().getNestedNotNullableColumn().get()}, key_sizes, context)
    {
        auto column = &getLowCardinalityColumn(key_columns_low_cardinality[0]);

        if (!context)
            throw Exception("Cache wasn't created for HashMethodSingleLowCardinalityColumn",
                            ErrorCodes::LOGICAL_ERROR);

        LowCardinalityDictionaryCache * cache;
        if constexpr (use_cache)
        {
            cache = typeid_cast<LowCardinalityDictionaryCache *>(context.get());
            if (!cache)
            {
                const auto & cached_val = *context;
                throw Exception("Invalid type for HashMethodSingleLowCardinalityColumn cache: "
                                + demangle(typeid(cached_val).name()), ErrorCodes::LOGICAL_ERROR);
            }
        }

        auto * dict = column->getDictionary().getNestedNotNullableColumn().get();
        is_nullable = column->getDictionary().nestedColumnIsNullable();
        key_columns = {dict};
        bool is_shared_dict = column->isSharedDictionary();

        typename LowCardinalityDictionaryCache::DictionaryKey dictionary_key;
        typename LowCardinalityDictionaryCache::CachedValuesPtr cached_values;

        if (is_shared_dict)
        {
            dictionary_key = {column->getDictionary().getHash(), dict->size()};
            if constexpr (use_cache)
                cached_values = cache->get(dictionary_key);
        }

        if (cached_values)
        {
            saved_hash = cached_values->saved_hash;
            dictionary_holder = cached_values->dictionary_holder;
        }
        else
        {
            saved_hash = column->getDictionary().tryGetSavedHash();
            dictionary_holder = column->getDictionaryPtr();

            if constexpr (use_cache)
            {
                if (is_shared_dict)
                {
                    cached_values = std::make_shared<typename LowCardinalityDictionaryCache::CachedValues>();
                    cached_values->saved_hash = saved_hash;
                    cached_values->dictionary_holder = dictionary_holder;

                    cache->set(dictionary_key, cached_values);
                }
            }
        }

        if constexpr (has_mapped)
            mapped_cache.resize(key_columns[0]->size());

        VisitValue empty(VisitValue::Empty);
        visit_cache.assign(key_columns[0]->size(), empty);

        size_of_index_type = column->getSizeOfIndexType();
        positions = column->getIndexesPtr().get();
    }

    ALWAYS_INLINE size_t getIndexAt(size_t row) const
    {
        switch (size_of_index_type)
        {
            case sizeof(UInt8): return static_cast<const ColumnUInt8 *>(positions)->getElement(row);
            case sizeof(UInt16): return static_cast<const ColumnUInt16 *>(positions)->getElement(row);
            case sizeof(UInt32): return static_cast<const ColumnUInt32 *>(positions)->getElement(row);
            case sizeof(UInt64): return static_cast<const ColumnUInt64 *>(positions)->getElement(row);
            default: throw Exception("Unexpected size of index type for low cardinality column.", ErrorCodes::LOGICAL_ERROR);
        }
    }

    /// Get the key from the key columns for insertion into the hash table.
    ALWAYS_INLINE auto getKey(size_t row) const
    {
        return Base::getKey(getIndexAt(row));
    }

    template <typename Data>
    ALWAYS_INLINE EmplaceResult emplaceKey(Data & data, size_t row_, Arena & pool)
    {
        size_t row = getIndexAt(row_);

        if (is_nullable && row == 0)
        {
            visit_cache[row] = VisitValue::Found;
            if constexpr (has_mapped)
                return EmplaceResult(data.getNullKeyData(), mapped_cache[0], !data.hasNullKeyData());
            else
                return EmplaceResult(!data.hasNullKeyData());
        }

        if (visit_cache[row] == VisitValue::Found)
        {
            if constexpr (has_mapped)
                return EmplaceResult(mapped_cache[row], mapped_cache[row], false);
            else
                return EmplaceResult(false);
        }

        auto key = getKey(row_);

        bool inserted = false;
        typename Data::iterator it;
        if (saved_hash)
            data.emplace(key, it, inserted, saved_hash[row]);
        else
            data.emplace(key, it, inserted);

        visit_cache[row] = VisitValue::Found;

        if (inserted)
            Base::onNewKey(*it, pool);

        if constexpr (has_mapped)
            return EmplaceResult(it->second, mapped_cache[row], inserted);
        else
            return EmplaceResult(inserted);
    }

    ALWAYS_INLINE bool isNullAt(size_t i)
    {
        if (!is_nullable)
            return false;

        return getIndexAt(i) == 0;
    }

    template <typename Data>
    ALWAYS_INLINE FindResult findFromRow(Data & data, size_t row_, Arena &)
    {
        size_t row = getIndexAt(row_);

        if (is_nullable && row == 0)
        {
            if constexpr (has_mapped)
                return FindResult(data.hasNullKeyData() ? data.getNullKeyData() : Mapped(), data.hasNullKeyData());
            else
                return FindResult(data.hasNullKeyData());
        }

        if (visit_cache[row] != VisitValue::Empty)
        {
            if constexpr (has_mapped)
                return FindResult(mapped_cache[row], visit_cache[row] == VisitValue::Found);
            else
                return FindResult(visit_cache[row] == VisitValue::Found);
        }

        auto key = getKey(row_);

        typename Data::iterator it;
        if (saved_hash)
            it = data.find(key, saved_hash[row]);
        else
            it = data.find(key);

        bool found = it != data.end();
        visit_cache[row] = found ? VisitValue::Found : VisitValue::NotFound;

        if constexpr (has_mapped)
        {
            if (found)
                mapped_cache[row] = it->second;
        }

        if constexpr (has_mapped)
            return FindResult(mapped_cache[row], found);
        else
            return FindResult(found);
    }

    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & pool)
    {
        row = getIndexAt(row);
        if (saved_hash)
            return saved_hash[row];

        return Base::getHash(data, row, pool);
    }
};


// Optional mask for low cardinality columns.
template <bool has_low_cardinality>
struct LowCardinalityKeys
{
    ColumnRawPtrs nested_columns;
    ColumnRawPtrs positions;
    Sizes position_sizes;
};

template <>
struct LowCardinalityKeys<false> {};

/// For the case where all keys are of fixed length, and they fit in N (for example, 128) bits.
template <typename Value, typename Key, typename Mapped, bool has_nullable_keys_ = false, bool has_low_cardinality_ = false>
struct HashMethodKeysFixed
    : private columns_hashing_impl::BaseStateKeysFixed<Key, has_nullable_keys_>
    , public columns_hashing_impl::HashMethodBase<Value, Mapped, true>
{
    static constexpr bool has_nullable_keys = has_nullable_keys_;
    static constexpr bool has_low_cardinality = has_low_cardinality_;

    LowCardinalityKeys<has_low_cardinality> low_cardinality_keys;
    Sizes key_sizes;
    size_t keys_size;

    using Base = columns_hashing_impl::BaseStateKeysFixed<Key, has_nullable_keys>;
    using BaseHashed = columns_hashing_impl::HashMethodBase<Value, Mapped, true>;

    HashMethodKeysFixed(const ColumnRawPtrs & key_columns, const Sizes & key_sizes, const HashMethodContextPtr &)
        : key_sizes(std::move(key_sizes)), keys_size(key_columns.size())
    {
        if constexpr (has_low_cardinality)
        {
            low_cardinality_keys.nested_columns.resize(key_columns.size());
            low_cardinality_keys.positions.assign(key_columns.size(), nullptr);
            low_cardinality_keys.position_sizes.resize(key_columns.size());
            for (size_t i = 0; i < key_columns.size(); ++i)
            {
                if (auto * low_cardinality_col = typeid_cast<const ColumnLowCardinality *>(key_columns[i]))
                {
                    low_cardinality_keys.nested_columns[i] = low_cardinality_col->getDictionary().getNestedColumn().get();
                    low_cardinality_keys.positions[i] = &low_cardinality_col->getIndexes();
                    low_cardinality_keys.position_sizes[i] = low_cardinality_col->getSizeOfIndexType();
                }
                else
                    low_cardinality_keys.nested_columns[i] = key_columns[i];
            }
        }

        Base::init(key_columns);
    }

    static HashMethodContextPtr createContext(const HashMethodContext::Settings &) { return nullptr; }

    ALWAYS_INLINE Key getKey(size_t row) const
    {
        if (has_nullable_keys)
        {
            auto bitmap = Base::createBitmap(row);
            return packFixed<Key>(row, keys_size, Base::getActualColumns(), key_sizes, bitmap);
        }
        else
        {
            if constexpr (has_low_cardinality)
                return packFixed<Key, true>(row, keys_size, low_cardinality_keys.nested_columns, key_sizes,
                                            &low_cardinality_keys.positions, &low_cardinality_keys.position_sizes);

            return packFixed<Key>(row, keys_size, Base::getActualColumns(), key_sizes);
        }
    }

    template <typename Data>
    ALWAYS_INLINE typename BaseHashed::EmplaceResult emplaceKey(Data & data, size_t row, Arena & /*pool*/)
    {
        typename Data::iterator it;
        return BaseHashed::emplaceKeyImpl(getKey(row), data, it);
    }

    template <typename Data>
    ALWAYS_INLINE typename BaseHashed::FindResult findKey(Data & data, size_t row, Arena & /*pool*/)
    {
        return BaseHashed::findKeyImpl(getKey(row), data);
    }

    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & /*pool*/)
    {
        return data.hash(getKey(row));
    }
};

/** Hash by concatenating serialized key values.
  * The serialized value differs in that it uniquely allows to deserialize it, having only the position with which it starts.
  * That is, for example, for strings, it contains first the serialized length of the string, and then the bytes.
  * Therefore, when aggregating by several strings, there is no ambiguity.
  */
template <typename Value, typename Mapped>
struct HashMethodSerialized : public columns_hashing_impl::HashMethodBase<Value, Mapped, false>
{
    using Base = columns_hashing_impl::HashMethodBase<Value, Mapped, false>;
    ColumnRawPtrs key_columns;
    size_t keys_size;

    HashMethodSerialized(const ColumnRawPtrs & key_columns, const Sizes & /*key_sizes*/, const HashMethodContextPtr &)
        : key_columns(key_columns), keys_size(key_columns.size()) {}

    static HashMethodContextPtr createContext(const HashMethodContext::Settings &) { return nullptr; }

    template <typename Data>
    ALWAYS_INLINE typename Base::EmplaceResult emplaceKey(Data & data, size_t row, Arena & pool)
    {
        auto key = getKey(row, pool);
        typename Data::iterator it;
        auto res = Base::emplaceKeyImpl(key, data, it);
        if (!res.isInserted())
            pool.rollback(key.size);

        return res;
    }

    template <typename Data>
    ALWAYS_INLINE typename Base::FindResult findKey(Data & data, size_t row, Arena & pool)
    {
        auto key = getKey(row, pool);
        auto res = Base::findKeyImpl(key, data);
        pool.rollback(key.size);

        return res;
    }

    template <typename Data>
    ALWAYS_INLINE size_t getHash(const Data & data, size_t row, Arena & pool)
    {
        auto key = getKey(row, pool);
        auto hash = data.hash(key);
        pool.rollback(key.size);

        return hash;
    }

protected:
    ALWAYS_INLINE StringRef getKey(size_t row, Arena & pool) const
    {
        return serializeKeysToPoolContiguous(row, keys_size, key_columns, pool);
    }
};

}
}
