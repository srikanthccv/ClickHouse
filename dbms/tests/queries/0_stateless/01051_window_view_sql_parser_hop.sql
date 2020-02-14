SET allow_experimental_window_view = 1;

DROP TABLE IF EXISTS test.mt;

CREATE TABLE test.mt(a Int32, timestamp DateTime) ENGINE=MergeTree ORDER BY tuple();

SELECT '---WATERMARK---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv WATERMARK=INTERVAL '1' SECOND AS SELECT count(a), HOP_START(wid) AS w_start, HOP_END(wid) AS w_end FROM test.mt GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid;

SELECT '---With w_end---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv AS SELECT count(a), HOP_START(wid) AS w_start, HOP_END(wid) AS w_end FROM test.mt GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid;

SELECT '---WithOut w_end---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv AS SELECT count(a), HOP_START(wid) AS w_start FROM test.mt GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid;

SELECT '---WITH---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv AS WITH toDateTime('2018-01-01 00:00:00') AS date_time SELECT count(a), HOP_START(wid) AS w_start, HOP_END(wid) AS w_end, date_time FROM test.mt GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid;

SELECT '---WHERE---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv AS SELECT count(a), HOP_START(wid) AS w_start FROM test.mt WHERE a != 1 GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid;

SELECT '---ORDER_BY---';
DROP TABLE IF EXISTS test.wv;
CREATE WINDOW VIEW test.wv AS SELECT count(a), HOP_START(wid) AS w_start FROM test.mt WHERE a != 1 GROUP BY HOP(timestamp, INTERVAL '3' SECOND, INTERVAL '5' SECOND) AS wid ORDER BY w_start;

DROP TABLE test.mt;
DROP TABLE test.wv;