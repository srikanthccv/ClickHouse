DROP TABLE IF EXISTS test_02559;
CREATE TABLE test_02559 (x UInt8, s String) ENGINE = MergeTree ORDER BY tuple();

INSERT INTO test_02559 VALUES (1, 'Hello, world!');

ALTER TABLE test_02559 ADD COLUMN y UInt8 DEFAULT 0;
INSERT INTO test_02559 VALUES (2, 'Goodbye.', 3);
SELECT * FROM test_02559 ORDER BY x;

SET enable_multiple_prewhere_read_steps=true, move_all_conditions_to_prewhere=true;

-- { echoOn }
SELECT s FROM test_02559 PREWHERE x AND y ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE y ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE NOT y ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE x AND y ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE x AND NOT y ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE y AND x ORDER BY s;
SELECT s, y FROM test_02559 PREWHERE (NOT y) AND x ORDER BY s;

ALTER TABLE test_02559 ADD COLUMN z UInt8 DEFAULT 10;
INSERT INTO test_02559 VALUES (3, 'So long, and thanks for all the fish.', 42, 0);
SELECT * FROM test_02559 ORDER BY x;

SELECT s FROM test_02559 PREWHERE z ORDER BY s;
SELECT s FROM test_02559 PREWHERE y AND z ORDER BY s;
SELECT s, z FROM test_02559 PREWHERE NOT y AND z ORDER BY s;
-- { echoOff }

DROP TABLE test_02559;
