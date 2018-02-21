DROP TABLE IF EXISTS test.nullable;
CREATE TABLE test.nullable (x String) ENGINE = MergeTree ORDER BY x;
INSERT INTO test.nullable VALUES ('hello'), ('world');
SELECT * FROM test.nullable;
ALTER TABLE test.nullable ADD COLUMN n Nullable(UInt64);
SELECT * FROM test.nullable;
ALTER TABLE test.nullable DROP COLUMN n;
ALTER TABLE test.nullable ADD COLUMN n Nullable(UInt64) DEFAULT NULL;
SELECT * FROM test.nullable;
ALTER TABLE test.nullable DROP COLUMN n;
ALTER TABLE test.nullable ADD COLUMN n Nullable(UInt64) DEFAULT 0;
SELECT * FROM test.nullable;
DROP TABLE test.nullable;
