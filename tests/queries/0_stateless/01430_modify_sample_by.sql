DROP TABLE IF EXISTS modify_sample;

SET min_insert_block_size_rows = 0, min_insert_block_size_bytes = 0;
SET max_block_size = 10;

CREATE TABLE modify_sample (d Date DEFAULT '2000-01-01', x UInt8) ENGINE = MergeTree PARTITION BY d ORDER BY x;
INSERT INTO modify_sample (x) SELECT toUInt8(number) AS x FROM system.numbers LIMIT 256;

SELECT count(), min(x), max(x), sum(x), uniqExact(x) FROM modify_sample SAMPLE 0.1; -- { serverError 141 }

ALTER TABLE modify_sample MODIFY SAMPLE BY x;
SELECT count(), min(x), max(x), sum(x), uniqExact(x) FROM modify_sample SAMPLE 0.1;

CREATE TABLE modify_sample_replicated (d Date DEFAULT '2000-01-01', x UInt8) ENGINE = ReplicatedMergeTree('/clickhouse/tables/test_01430', 'modify_sample') PARTITION BY d ORDER BY x;
INSERT INTO modify_sample_replicated (x) SELECT toUInt8(number) AS x FROM system.numbers LIMIT 256;

SELECT count(), min(x), max(x), sum(x), uniqExact(x) FROM modify_sample_replicated SAMPLE 0.1; -- { serverError 141 }

ALTER TABLE modify_sample_replicated MODIFY SAMPLE BY x;
SELECT count(), min(x), max(x), sum(x), uniqExact(x) FROM modify_sample_replicated SAMPLE 0.1;

DROP TABLE modify_sample;
