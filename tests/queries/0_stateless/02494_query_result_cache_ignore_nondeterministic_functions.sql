-- { echoOn }

SYSTEM DROP QUERY RESULT CACHE;

-- rand() is non-deterministic, with default settings no entry in the query result cache should be created
SELECT count(rand(1)) SETTINGS enable_experimental_query_result_cache = true;
SELECT count(*) FROM system.queryresult_cache;

-- force an entry
SELECT count(rand(1)) SETTINGS enable_experimental_query_result_cache = true, query_result_cache_ignore_nondeterministic_functions = false;
SELECT count(*) FROM system.queryresult_cache;

SYSTEM DROP QUERY RESULT CACHE;

-- { echoOff }
