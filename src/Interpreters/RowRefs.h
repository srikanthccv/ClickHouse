#pragma once

#include <algorithm>
#include <cassert>
#include <list>
#include <mutex>
#include <optional>
#include <variant>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn.h>
#include <Interpreters/asof.h>
#include <base/sort.h>
#include <Common/Arena.h>


#include <base/logger_useful.h>

namespace DB
{

class Block;

/// Reference to the row in block.
struct __attribute__((__packed__)) RowRef
{
    using SizeT = uint32_t; /// Do not use size_t cause of memory economy

    const Block * block = nullptr;
    SizeT row_num = 0;

    RowRef() = default;
    RowRef(const Block * block_, size_t row_num_) : block(block_), row_num(row_num_) {}
};

/// Single linked list of references to rows. Used for ALL JOINs (non-unique JOINs)
struct RowRefList : RowRef
{
    /// Portion of RowRefs, 16 * (MAX_SIZE + 1) bytes sized.
    struct Batch
    {
        static constexpr size_t MAX_SIZE = 7; /// Adequate values are 3, 7, 15, 31.

        SizeT size = 0; /// It's smaller than size_t but keeps align in Arena.
        Batch * next;
        RowRef row_refs[MAX_SIZE];

        Batch(Batch * parent)
            : next(parent)
        {}

        bool full() const { return size == MAX_SIZE; }

        Batch * insert(RowRef && row_ref, Arena & pool)
        {
            if (full())
            {
                auto batch = pool.alloc<Batch>();
                *batch = Batch(this);
                batch->insert(std::move(row_ref), pool);
                return batch;
            }

            row_refs[size++] = std::move(row_ref);
            return this;
        }
    };

    class ForwardIterator
    {
    public:
        ForwardIterator(const RowRefList * begin)
            : root(begin)
            , first(true)
            , batch(root->next)
            , position(0)
        {}

        const RowRef * operator -> () const
        {
            if (first)
                return root;
            return &batch->row_refs[position];
        }

        const RowRef * operator * () const
        {
            if (first)
                return root;
            return &batch->row_refs[position];
        }

        void operator ++ ()
        {
            if (first)
            {
                first = false;
                return;
            }

            if (batch)
            {
                ++position;
                if (position >= batch->size)
                {
                    batch = batch->next;
                    position = 0;
                }
            }
        }

        bool ok() const { return first || batch; }

    private:
        const RowRefList * root;
        bool first;
        Batch * batch;
        size_t position;
    };

    RowRefList() {}
    RowRefList(const Block * block_, size_t row_num_) : RowRef(block_, row_num_) {}

    ForwardIterator begin() const { return ForwardIterator(this); }

    /// insert element after current one
    void insert(RowRef && row_ref, Arena & pool)
    {
        if (!next)
        {
            next = pool.alloc<Batch>();
            *next = Batch(nullptr);
        }
        next = next->insert(std::move(row_ref), pool);
    }

private:
    Batch * next = nullptr;
};

/**
 * This class is intended to push sortable data into.
 * When looking up values the container ensures that it is sorted for log(N) lookup
 * After calling any of the lookup methods, it is no longer allowed to insert more data as this would invalidate the
 * references that can be returned by the lookup methods
 */

template <typename TEntry>
class SortedLookupVector
{
public:
    using Base = std::vector<TEntry>;
    using TKey = decltype(TEntry::asof_value);
    using Keys = std::vector<TKey>;

    void insert(const IColumn & asof_column, const Block * block, size_t row_num)
    {
        using ColumnType = ColumnVectorOrDecimal<TKey>;
        const auto & column = assert_cast<const ColumnType &>(asof_column);
        TKey k = column.getElement(row_num);

        assert(!sorted.load(std::memory_order_acquire));
        array.emplace_back(k, block, row_num);
    }

    /// Find an element based on the inequality rules
    /// Note that this function uses 2 arrays, one with only the keys (so it's smaller and more memory efficient)
    /// and a second one with both the key and the Rowref to be returned
    /// Both are sorted only once, in a concurrent safe manner
    const RowRef * find(ASOF::Inequality inequality, const IColumn & asof_column, size_t row_num)
    {
        sort();

        using ColumnType = ColumnVectorOrDecimal<TKey>;
        const auto & column = assert_cast<const ColumnType &>(asof_column);
        TKey k = column.getElement(row_num);

        auto it = keys.cend();
        switch (inequality)
        {
            case ASOF::Inequality::LessOrEquals:
            {
                it = std::lower_bound(keys.cbegin(), keys.cend(), k);
                break;
            }
            case ASOF::Inequality::Less:
            {
                it = std::upper_bound(keys.cbegin(), keys.cend(), k);
                break;
            }
            case ASOF::Inequality::GreaterOrEquals:
            {
                auto first_ge = std::upper_bound(keys.cbegin(), keys.cend(), k);
                if (first_ge == keys.cend() && keys.size())
                    first_ge--;
                while (first_ge != keys.cbegin() && *first_ge > k)
                    first_ge--;
                if (*first_ge <= k)
                    it = first_ge;
                break;
            }
            case ASOF::Inequality::Greater:
            {
                auto first_ge = std::upper_bound(keys.cbegin(), keys.cend(), k);
                if (first_ge == keys.cend() && keys.size())
                    first_ge--;
                while (first_ge != keys.cbegin() && *first_ge >= k)
                    first_ge--;
                if (*first_ge < k)
                    it = first_ge;
                break;
            }
            default:
                throw Exception("Invalid ASOF Join order", ErrorCodes::LOGICAL_ERROR);
        }

        if (it != keys.cend())
            return &((array.cbegin() + (it - keys.begin()))->row_ref);

        return nullptr;
    }

private:
    std::atomic<bool> sorted = false;
    mutable std::mutex lock;

    Base array;
    /// We keep a separate copy of just the keys to make the searches more memory efficient
    Keys keys;

    // Double checked locking with SC atomics works in C++
    // https://preshing.com/20130930/double-checked-locking-is-fixed-in-cpp11/
    // The first thread that calls one of the lookup methods sorts the data
    // After calling the first lookup method it is no longer allowed to insert any data
    // the array becomes immutable
    void sort()
    {
        if (!sorted.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> l(lock);
            if (!sorted.load(std::memory_order_relaxed))
            {
                if (!array.empty())
                {
                    ::sort(array.begin(), array.end());
                    keys.reserve(array.size());
                    for (auto & e : array)
                        keys.push_back(e.asof_value);
                }

                sorted.store(true, std::memory_order_release);
            }
        }
    }
};

struct AsofRowRefsBase
{
    AsofRowRefsBase() = default;
    virtual ~AsofRowRefsBase() { }

    static std::optional<TypeIndex> getTypeSize(const IColumn & asof_column, size_t & type_size);
    virtual void insert(const IColumn &, const Block *, size_t) = 0;
    virtual const RowRef * findAsof(ASOF::Inequality, const IColumn &, size_t) = 0;
};

template <typename T>
class AsofRowRefDerived : public AsofRowRefsBase
{
public:
    template <typename EntryType>
    struct Entry
    {
        T asof_value;
        RowRef row_ref;

        Entry() = delete;
        explicit Entry(T v) : asof_value(v) { }
        Entry(T v, RowRef rr) : asof_value(v), row_ref(rr) { }
        Entry(T v, const Block * block, size_t row_num) : asof_value(v), row_ref(block, row_num) { }

        bool operator<(const Entry & other) const { return asof_value < other.asof_value; }
    };

    AsofRowRefDerived() { }

    // This will be synchronized by the rwlock mutex in Join.h
    void insert(const IColumn & asof_column, const Block * block, size_t row_num) override { lookups.insert(asof_column, block, row_num); }

    // This will internally synchronize
    const RowRef * findAsof(ASOF::Inequality inequality, const IColumn & asof_column, size_t row_num) override
    {
        return lookups.find(inequality, asof_column, row_num);
    }

private:
    SortedLookupVector<Entry<T>> lookups;
};

// It only contains a std::unique_ptr, which contains a single pointer, which is memmovable.
// Source: https://github.com/ClickHouse/ClickHouse/issues/4906
using AsofRowRefs = std::unique_ptr<AsofRowRefsBase>;
AsofRowRefs createAsofRowRef(TypeIndex type);
}
