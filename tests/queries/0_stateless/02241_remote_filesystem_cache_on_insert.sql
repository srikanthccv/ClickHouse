-- Tags: no-parallel, no-fasttest

-- { echo }

SET remote_fs_cache_on_write_operations=1;

DROP TABLE IF EXISTS test;
CREATE TABLE test (key UInt32, value String) Engine=MergeTree() ORDER BY key SETTINGS storage_policy='s3_cache';

SYSTEM DROP REMOTE FILESYSTEM CACHE;

SELECT file_segment_range_begin, file_segment_range_end, size, state FROM (SELECT file_segment_range_begin, file_segment_range_end, size, state, local_path FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path) WHERE endsWith(local_path, 'data.bin') FORMAT Vertical;
SELECT count() FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path;
SELECT count() FROM system.remote_filesystem_cache;

INSERT INTO test SELECT number, toString(number) FROM numbers(100) SETTINGS remote_fs_cache_on_write_operations=1;

SELECT file_segment_range_begin, file_segment_range_end, size, state FROM (SELECT file_segment_range_begin, file_segment_range_end, size, state, local_path FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path) WHERE endsWith(local_path, 'data.bin') FORMAT Vertical;
SELECT count() FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path;
SELECT count() FROM system.remote_filesystem_cache;

SELECT count() FROM system.remote_filesystem_cache WHERE cache_hits > 0;

SELECT  * FROM test FORMAT Null;
SELECT count() FROM system.remote_filesystem_cache WHERE cache_hits > 0;

SELECT  * FROM test FORMAT Null;
SELECT count() FROM system.remote_filesystem_cache WHERE cache_hits > 0;

SELECT count() size FROM system.remote_filesystem_cache;

SYSTEM DROP REMOTE FILESYSTEM CACHE;

INSERT INTO test SELECT number, toString(number) FROM numbers(100, 200);

SELECT file_segment_range_begin, file_segment_range_end, size, state FROM (SELECT file_segment_range_begin, file_segment_range_end, size, state, local_path FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path) WHERE endsWith(local_path, 'data.bin') FORMAT Vertical;
SELECT count() FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.remote_filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path;
SELECT count() FROM system.remote_filesystem_cache;

SELECT count() FROM system.remote_filesystem_cache;
INSERT INTO test SELECT number, toString(number) FROM numbers(100) SETTINGS remote_fs_cache_on_write_operations=0;
SELECT count() FROM system.remote_filesystem_cache;

INSERT INTO test SELECT number, toString(number) FROM numbers(100);
INSERT INTO test SELECT number, toString(number) FROM numbers(300, 10000);
SELECT count() FROM system.remote_filesystem_cache;
OPTIMIZE TABLE test FINAL;
SELECT count() FROM system.remote_filesystem_cache;

SET mutations_sync=2;
ALTER TABLE test UPDATE value = 'kek' WHERE key = 100;
SELECT count() FROM system.remote_filesystem_cache;
