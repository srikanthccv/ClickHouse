SET allow_experimental_window_view = 1;
SET allow_experimental_window_view = 1;

DROP TABLE IF EXISTS mt;
DROP TABLE IF EXISTS dst;
DROP TABLE IF EXISTS wv;

CREATE TABLE dst(count UInt64, w_end DateTime) Engine=MergeTree ORDER BY tuple();
CREATE TABLE mt(a Int32, timestamp DateTime) ENGINE=MergeTree ORDER BY tuple();
CREATE WINDOW VIEW wv TO dst ENGINE=AggregatingMergeTree ORDER BY wid WATERMARK=STRICTLY_ASCENDING AS SELECT count(a) AS count, TUMBLE_END(wid) as w_end FROM mt GROUP BY TUMBLE(timestamp, INTERVAL '5' SECOND) AS wid;

INSERT INTO mt VALUES (1, '1990/01/01 12:00:00');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:01');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:02');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:05');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:10');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:11');
INSERT INTO mt VALUES (1, '1990/01/01 12:00:30');

SELECT sleep(1);
SELECT * from dst order by w_end;

DROP TABLE wv;
DROP TABLE mt;
DROP TABLE dst;
