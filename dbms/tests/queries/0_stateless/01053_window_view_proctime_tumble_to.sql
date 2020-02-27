SET allow_experimental_window_view = 1;
SET allow_experimental_window_view = 1;

DROP TABLE IF EXISTS test.mt;
DROP TABLE IF EXISTS test.dst;
DROP TABLE IF EXISTS test.wv;

CREATE TABLE test.dst(count UInt64) Engine=MergeTree ORDER BY tuple();
CREATE TABLE test.mt(a Int32) ENGINE=MergeTree ORDER BY tuple();
CREATE WINDOW VIEW test.wv TO test.dst AS SELECT count(a) AS count FROM test.mt GROUP BY TUMBLE(now(), INTERVAL '1' SECOND) AS wid;

INSERT INTO test.mt VALUES (1);
SELECT sleep(2);
SELECT count from test.dst;

DROP TABLE test.wv;
DROP TABLE test.mt;
DROP TABLE test.dst;
