-- Tags: no-s3-storage, no-replicated-database

DROP TABLE IF EXISTS test;

EXPLAIN SYNTAX
CREATE TABLE test (a Int32)
ENGINE = MergeTree() order by tuple()
SETTINGS disk = disk(type=local, path='/var/lib/clickhouse/disks/local/');

CREATE TABLE test (a Int32)
ENGINE = MergeTree() order by tuple()
SETTINGS disk = disk(type=local, path='/local/'); -- { serverError BAD_ARGUMENTS }

CREATE TABLE test (a Int32)
ENGINE = MergeTree() order by tuple()
SETTINGS disk = disk(type=local, path='/var/lib/clickhouse/disks/local/');

INSERT INTO test SELECT number FROM numbers(100);
SELECT count() FROM test;

DETACH TABLE test;
ATTACH TABLE test;

SHOW CREATE TABLE test;
DESCRIBE TABLE test;

INSERT INTO test SELECT number FROM numbers(100);
SELECT count() FROM test;

DROP TABLE test;

CREATE TABLE test (a Int32)
ENGINE = MergeTree() order by tuple()
SETTINGS disk = disk(type=cache, max_size='1Gi', path='/var/lib/clickhouse/custom_disk_cache/', disk=disk(type=local, path='/var/lib/clickhouse/disks/local/'));

SHOW CREATE TABLE test;

DROP TABLE test;
