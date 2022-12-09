DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS tj;
DROP TABLE IF EXISTS tjj;

CREATE TABLE t1 (key1 UInt64, key2 UInt64, key3 UInt64) ENGINE = Memory;
INSERT INTO t1 VALUES (11, 12, 13), (21, 22, 23), (31, 32, 33), (41, 42, 43), (51, 52, 53);

CREATE TABLE tj (key2 UInt64, key1 UInt64, key3 UInt64, attr UInt64) ENGINE = Join(ALL, INNER, key3, key2, key1);
INSERT INTO tj VALUES (22, 21, 23, 2000), (32, 31, 33, 3000), (42, 41, 43, 4000), (52, 51, 53, 5000), (62, 61, 63, 6000);

SELECT * FROM t1 ALL INNER JOIN tj USING (key1, key2, key3) ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj USING (key2, key3, key1) ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj USING (key3, key2, key1) ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj USING (key1, key3, key2) ORDER BY key1;

SELECT * FROM t1 ALL INNER JOIN tj ON t1.key3 = tj.key3 AND t1.key2 = tj.key2 AND t1.key1 = tj.key1 ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key2 = tj.key2 AND t1.key3 = tj.key3 AND t1.key1 = tj.key1 ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key3 = tj.key3 AND t1.key1 = tj.key1 AND t1.key2 = tj.key2 ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key1 = tj.key1 AND t1.key3 = tj.key3 AND t1.key2 = tj.key2 ORDER BY key1;

SELECT * FROM (SELECT key3 AS c, key1 AS a, key2 AS b FROM t1) AS t1 ALL INNER JOIN tj ON t1.a = tj.key1 AND t1.c = tj.key3 AND t1.b = tj.key2 ORDER BY t1.a;
SELECT * FROM (SELECT key3 AS c, key1 AS a, key2 AS b FROM t1) AS t1 ALL INNER JOIN tj ON t1.a = tj.key1 AND t1.b = tj.key2 AND t1.c = tj.key3 ORDER BY t1.a;
SELECT * FROM (SELECT key3 AS c, key1 AS a, key2 AS b FROM t1) AS t1 ALL INNER JOIN tj ON t1.c = tj.key3 AND t1.a = tj.key1 AND t1.b = tj.key2 ORDER BY t1.a;

SELECT * FROM t1 ALL INNER JOIN tj ON 1; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON 0; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON NULL; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON 1 == 1; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON 1 != 1; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }

SELECT * FROM t1 ALL INNER JOIN tj USING (key2, key3); -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj USING (key1, key2, attr); -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj USING (key1, key2, key3, attr); -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key1 = tj.attr; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key1 = tj.key1; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key1 = tj.key1 AND t1.key2 = tj.key2 AND t1.key3 = tj.attr; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }
SELECT * FROM t1 ALL INNER JOIN tj ON t1.key1 = tj.key1 AND t1.key2 = tj.key2 AND t1.key3 = tj.key3 AND t1.key1 = tj.key1; -- { serverError INCOMPATIBLE_TYPE_OF_JOIN }


CREATE TABLE tjj (key2 UInt64, key1 UInt64, key3 UInt64, attr UInt64) ENGINE = Join(ALL, INNER, key3, key2, key1);
INSERT INTO tjj VALUES (11, 11, 11, 1000), (21, 21, 21, 2000), (31, 31, 31, 3000), (41, 41, 41, 4000), (51, 51, 51, 5000), (61, 61, 61, 6000);

SELECT * FROM t1 ALL INNER JOIN tjj ON t1.key1 = tjj.key1 AND t1.key1 = tjj.key2 AND t1.key1 = tjj.key3 ORDER BY key1;
SELECT * FROM t1 ALL INNER JOIN tjj ON t1.key1 = tjj.key1 AND t1.key1 = tjj.key3 AND t1.key1 = tjj.key2 ORDER BY key1;

DROP TABLE IF EXISTS t1;
DROP TABLE IF EXISTS tj;
DROP TABLE IF EXISTS tjj;
