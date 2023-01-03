DROP TABLE IF EXISTS t;
CREATE TABLE t (x Decimal(18, 3)) ENGINE = MergeTree ORDER BY x;
INSERT INTO t VALUES (1.1);
SELECT * FROM t WHERE toUInt64(x) = 1;
DROP TABLE t;
