-- Tags: no-fasttest, no-parallel, no-s3-storage, no-random-settings

-- { echo }

SET enable_filesystem_cache_on_write_operations=0;

SYSTEM DROP FILESYSTEM CACHE;

DROP TABLE IF EXISTS nopers;
CREATE TABLE nopers (key UInt32, value String) Engine=MergeTree() ORDER BY key SETTINGS storage_policy='s3_cache', min_bytes_for_wide_part = 10485760, compress_marks=false, compress_primary_key=false;
SYSTEM STOP MERGES nopers;

INSERT INTO nopers SELECT number, toString(number) FROM numbers(10);
SELECT * FROM nopers FORMAT Null;
SELECT sum(size) FROM system.filesystem_cache;

SELECT extract(local_path, '.*/([\w.]+)') as file, extract(cache_path, '.*/([\w.]+)') as cache, size
FROM
(
    SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path
    FROM system.remote_data_paths
) AS data_paths
INNER JOIN
    system.filesystem_cache AS caches
ON data_paths.cache_path = caches.cache_path
ORDER BY file, cache, size;

DROP TABLE IF EXISTS test;
CREATE TABLE test (key UInt32, value String) Engine=MergeTree() ORDER BY key SETTINGS storage_policy='s3_cache_small', min_bytes_for_wide_part = 10485760, compress_marks=false, compress_primary_key=false;
SYSTEM STOP MERGES test;

INSERT INTO test SELECT number, toString(number) FROM numbers(100);
SELECT * FROM test FORMAT Null;

SELECT sum(size) FROM system.filesystem_cache;

SELECT count() FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path;
SELECT count() FROM system.filesystem_cache;

SELECT extract(local_path, '.*/([\w.]+)') as file, extract(cache_path, '.*/([\w.]+)') as cache, size
FROM
(
    SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path
    FROM system.remote_data_paths
) AS data_paths
INNER JOIN
    system.filesystem_cache AS caches
ON data_paths.cache_path = caches.cache_path
ORDER BY file, cache, size;

DROP TABLE IF EXISTS test2;
CREATE TABLE test2 (key UInt32, value String) Engine=MergeTree() ORDER BY key SETTINGS storage_policy='s3_cache_small', min_bytes_for_wide_part = 10485760, compress_marks=false, compress_primary_key=false;
SYSTEM STOP MERGES test2;

INSERT INTO test2 SELECT number, toString(number) FROM numbers(100000);
SELECT * FROM test2 FORMAT Null;

SELECT sum(size) FROM system.filesystem_cache;

SELECT count() FROM (SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path FROM system.remote_data_paths ) AS data_paths INNER JOIN system.filesystem_cache AS caches ON data_paths.cache_path = caches.cache_path;
SELECT count() FROM system.filesystem_cache;

SELECT extract(local_path, '.*/([\w.]+)') as file, extract(cache_path, '.*/([\w.]+)') as cache, size
FROM
(
    SELECT arrayJoin(cache_paths) AS cache_path, local_path, remote_path
    FROM system.remote_data_paths
) AS data_paths
INNER JOIN
    system.filesystem_cache AS caches
ON data_paths.cache_path = caches.cache_path
ORDER BY file, cache, size;

DROP TABLE test;
DROP TABLE test2;
DROP TABLE nopers;
