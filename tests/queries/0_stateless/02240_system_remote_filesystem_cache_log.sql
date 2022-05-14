-- Tags: no-parallel, no-fasttest, no-s3-storage

-- { echo }

SYSTEM DROP FILESYSTEM CACHE;
SET enable_filesystem_cache_log=1;
SET enable_filesystem_cache_on_write_operations=0;

DROP TABLE IF EXISTS test;
DROP TABLE IF EXISTS system.filesystem_cache_log;
CREATE TABLE test (key UInt32, value String) Engine=MergeTree() ORDER BY key SETTINGS storage_policy='s3_cache', min_bytes_for_wide_part = 10485760;
INSERT INTO test SELECT number, toString(number) FROM numbers(100000);

SELECT 2240, * FROM test FORMAT Null;
SYSTEM FLUSH LOGS;
SELECT file_segment_range, read_type FROM system.filesystem_cache_log WHERE read_type='READ_FROM_FS_AND_DOWNLOADED_TO_CACHE' AND query_id = (SELECT query_id from system.query_log where query LIKE '%SELECT 2240%' AND current_database = currentDatabase() ORDER BY event_time desc LIMIT 1);

SELECT 2241, * FROM test FORMAT Null;
SYSTEM FLUSH LOGS;
SELECT file_segment_range, read_type FROM system.filesystem_cache_log WHERE read_type='READ_FROM_CACHE' AND query_id = (SELECT query_id from system.query_log where query LIKE '%SELECT 2241%' AND current_database = currentDatabase() ORDER BY event_time desc LIMIT 1);
