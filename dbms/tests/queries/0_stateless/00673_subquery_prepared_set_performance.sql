DROP TABLE IF EXISTS test.mergetree;

CREATE TABLE test.mergetree (x UInt64) ENGINE = MergeTree ORDER BY x;
INSERT INTO test.mergetree VALUES (1);

SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM test.mergetree WHERE x IN (SELECT * FROM numbers(10000000))))))))))));

SET force_primary_key = 1;

SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM test.mergetree WHERE x IN (SELECT * FROM numbers(10000000))))))))))));

DROP TABLE test.mergetree;
