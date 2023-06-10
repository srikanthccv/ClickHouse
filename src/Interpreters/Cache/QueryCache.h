#pragma once

#include <Common/CacheBase.h>
#include <Core/Block.h>
#include <Parsers/IAST_fwd.h>
#include <Processors/Sources/SourceFromChunks.h>
#include <Poco/Util/LayeredConfiguration.h>
#include <Processors/Chunk.h>
#include <QueryPipeline/Pipe.h>

namespace DB
{

/// Does AST contain non-deterministic functions like rand() and now()?
bool astContainsNonDeterministicFunctions(ASTPtr ast, ContextPtr context);

/// Maps queries to query results. Useful to avoid repeated query calculation.
///
/// The cache does not aim to be transactionally consistent (which is difficult to get right). For example, the cache is not invalidated
/// when data is inserted/deleted into/from tables referenced by queries in the cache. In such situations, incorrect results may be
/// returned. In order to still obtain sufficiently up-to-date query results, a expiry time (TTL) must be specified for each cache entry
/// after which it becomes stale and is ignored. Stale entries are removed opportunistically from the cache, they are only evicted when a
/// new entry is inserted and the cache has insufficient capacity.
class QueryCache
{
public:
    /// Represents a query result in the cache.
    struct Key
    {
        /// ----------------------------------------------------
        /// The actual key (data which gets hashed):

        /// Unlike the query string, the AST is agnostic to lower/upper case (SELECT vs. select)
        const ASTPtr ast;

        /// Note: For a transactionally consistent cache, we would need to include the system settings in the cache key or invalidate the
        /// cache whenever the settings change. This is because certain settings (e.g. "additional_table_filters") can affect the query
        /// result.

        /// ----------------------------------------------------
        /// Additional stuff data stored in the key, not hashed:

        /// Result metadata for constructing the pipe.
        const Block header;

        /// The user who executed the query.
        const String user_name;

        /// If the associated entry can be read by other users. In general, sharing is a bad idea: First, it is unlikely that different
        /// users pose the same queries. Second, sharing potentially breaches security. E.g. User A should not be able to bypass row
        /// policies on some table by running the same queries as user B for whom no row policies exist.
        bool is_shared;

        /// When does the entry expire?
        const std::chrono::time_point<std::chrono::system_clock> expires_at;

        /// Are the chunks in the entry compressed?
        /// (we could theoretically apply compression also to the totals and extremes but it's an obscure use case)
        const bool is_compressed;

        Key(ASTPtr ast_,
            Block header_,
            const String & user_name_, bool is_shared_,
            std::chrono::time_point<std::chrono::system_clock> expires_at_,
            bool is_compressed);

        bool operator==(const Key & other) const;
        String queryStringFromAst() const;
    };

    struct Entry
    {
        Chunks chunks;
        std::optional<Chunk> totals = std::nullopt;
        std::optional<Chunk> extremes = std::nullopt;
    };

private:
    struct KeyHasher
    {
        size_t operator()(const Key & key) const;
    };

    struct QueryCacheEntryWeight
    {
        size_t operator()(const Entry & entry) const;
    };

    struct IsStale
    {
        bool operator()(const Key & key) const;
    };

    /// query --> query result
    using Cache = CacheBase<Key, Entry, KeyHasher, QueryCacheEntryWeight>;

    /// query --> query execution count
    using TimesExecuted = std::unordered_map<Key, size_t, KeyHasher>;

public:
    /// Buffers multiple partial query result chunks (buffer()) and eventually stores them as cache entry (finalizeWrite()).
    ///
    /// Implementation note: Queries may throw exceptions during runtime, e.g. out-of-memory errors. In this case, no query result must be
    /// written into the query cache. Unfortunately, neither the Writer nor the special transform added on top of the query pipeline which
    /// holds the Writer know whether they are destroyed because the query ended successfully or because of an exception (otherwise, we
    /// could simply implement a check in their destructors). To handle exceptions correctly nevertheless, we do the actual insert in
    /// finalizeWrite() as opposed to the Writer destructor. This function is then called only for successful queries in finish_callback()
    /// which runs before the transform and the Writer are destroyed, whereas for unsuccessful queries we do nothing (the Writer is
    /// destroyed w/o inserting anything).
    /// Queries may also be cancelled by the user, in which case IProcessor's cancel bit is set. FinalizeWrite() is only called if the
    /// cancel bit is not set.
    class Writer
    {
    public:

        Writer(const Writer & other);

        enum class ChunkType {Result, Totals, Extremes};
        void buffer(Chunk && chunk, ChunkType chunk_type);

        void finalizeWrite();
    private:
        std::mutex mutex;
        Cache & cache;
        const Key key;
        const size_t max_entry_size_in_bytes;
        const size_t max_entry_size_in_rows;
        const std::chrono::time_point<std::chrono::system_clock> query_start_time = std::chrono::system_clock::now(); /// Writer construction and finalizeWrite() coincide with query start/end
        const std::chrono::milliseconds min_query_runtime;
        const bool squash_partial_results;
        const size_t max_block_size;
        Cache::MappedPtr query_result TSA_GUARDED_BY(mutex) = std::make_shared<Entry>();
        std::atomic<bool> skip_insert = false;
        bool was_finalized = false;

        Writer(Cache & cache_, const Key & key_,
            size_t max_entry_size_in_bytes_, size_t max_entry_size_in_rows_,
            std::chrono::milliseconds min_query_runtime_,
            bool squash_partial_results_,
            size_t max_block_size_);

        friend class QueryCache; /// for createWriter()
    };

    /// Reader's constructor looks up a query result for a key in the cache. If found, it constructs source processors (that generate the
    /// cached result) for use in a pipe or query pipeline.
    class Reader
    {
    public:
        bool hasCacheEntryForKey() const;
        /// getSource*() moves source processors out of the Reader. Call each of these method just once.
        std::unique_ptr<SourceFromChunks> getSource();
        std::unique_ptr<SourceFromChunks> getSourceTotals();
        std::unique_ptr<SourceFromChunks> getSourceExtremes();
    private:
        Reader(Cache & cache_, const Key & key, const std::lock_guard<std::mutex> &);
        void buildSourceFromChunks(Block header, Chunks && chunks, const std::optional<Chunk> & totals, const std::optional<Chunk> & extremes);
        std::unique_ptr<SourceFromChunks> source_from_chunks;
        std::unique_ptr<SourceFromChunks> source_from_chunks_totals;
        std::unique_ptr<SourceFromChunks> source_from_chunks_extremes;
        friend class QueryCache; /// for createReader()
    };

    QueryCache();

    void updateConfiguration(const Poco::Util::AbstractConfiguration & config);

    Reader createReader(const Key & key);
    Writer createWriter(const Key & key, std::chrono::milliseconds min_query_runtime, bool squash_partial_results, size_t max_block_size, size_t max_query_cache_size_in_bytes_quota, size_t max_query_cache_entries_quota);

    void reset();

    /// Record new execution of query represented by key. Returns number of executions so far.
    size_t recordQueryRun(const Key & key);

    /// For debugging and system tables
    std::vector<QueryCache::Cache::KeyMapped> dump() const;

private:
    Cache cache;

    mutable std::mutex mutex;
    TimesExecuted times_executed TSA_GUARDED_BY(mutex);

    /// Cache configuration
    size_t max_entry_size_in_bytes TSA_GUARDED_BY(mutex) = 0;
    size_t max_entry_size_in_rows TSA_GUARDED_BY(mutex) = 0;

    size_t cache_size_in_bytes TSA_GUARDED_BY(mutex) = 0; /// Updated in each cache insert/delete

    friend class StorageSystemQueryCache;
};

using QueryCachePtr = std::shared_ptr<QueryCache>;

}
