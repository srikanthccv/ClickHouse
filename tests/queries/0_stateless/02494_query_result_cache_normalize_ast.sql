-- { echoOn }

SYSTEM DROP QUERY RESULT CACHE;
DROP TABLE IF EXISTS old;

-- save current event counts for query result cache
CREATE TABLE old (event String, value UInt64) ENGINE=MergeTree ORDER BY event;
INSERT INTO old SELECT event, value FROM system.events WHERE event LIKE 'QueryResultCache%';

-- Run query whose result gets cached in the query result cache (QRC).
-- Besides "enable_experimental_query_result_cache", pass two knobs. We just care *that* they are passed and not about their effect.
-- One QRC-specific knob and one non-QRC-specific knob
SELECT 1 SETTINGS enable_experimental_query_result_cache = true, query_result_cache_store_results_of_queries_with_nondeterministic_functions = true, max_threads = 16;
SELECT COUNT(*) FROM system.queryresult_cache;

-- Run same query again. We want its result to be served from the QRC.
-- Technically, both SELECT 1 queries have different ASTs.
SELECT 1 SETTINGS enable_experimental_query_result_cache_passive_usage = true,  max_threads = 16;

-- Different ASTs lead to different QRC keys. Internal normalization makes sure that the keys match regardless.
-- Verify by checking that we had a cache hit.
SELECT value = (SELECT value FROM old WHERE event = 'QueryResultCacheHits') + 1
FROM system.events
WHERE event = 'QueryResultCacheHits';

SYSTEM DROP QUERY RESULT CACHE;
DROP TABLE old;

-- { echoOff }
