SET enable_debug_queries = 1;
SET enable_optimize_predicate_expression = 0;
SET multiple_joins_rewriter_version = 1;

DROP TABLE IF EXISTS t1_00849;
DROP TABLE IF EXISTS t2_00849;
DROP TABLE IF EXISTS t3_00849;
DROP TABLE IF EXISTS t4_00849;

CREATE TABLE t1_00849 (a UInt32, b Nullable(Int32)) ENGINE = Memory;
CREATE TABLE t2_00849 (a UInt32, b Nullable(Int32)) ENGINE = Memory;
CREATE TABLE t3_00849 (a UInt32, b Nullable(Int32)) ENGINE = Memory;
CREATE TABLE t4_00849 (a UInt32, b Nullable(Int32)) ENGINE = Memory;

ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849 WHERE t1_00849.a = t2_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849 WHERE t1_00849.b = t2_00849.b;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849 WHERE t1_00849.a = t2_00849.a AND t1_00849.a = t3_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849 WHERE t1_00849.b = t2_00849.b AND t1_00849.b = t3_00849.b;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t1_00849.a = t2_00849.a AND t1_00849.a = t3_00849.a AND t1_00849.a = t4_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t1_00849.b = t2_00849.b AND t1_00849.b = t3_00849.b AND t1_00849.b = t4_00849.b;

ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t2_00849.a = t1_00849.a AND t2_00849.a = t3_00849.a AND t2_00849.a = t4_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t3_00849.a = t1_00849.a AND t3_00849.a = t2_00849.a AND t3_00849.a = t4_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t4_00849.a = t1_00849.a AND t4_00849.a = t2_00849.a AND t4_00849.a = t3_00849.a;
ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849 WHERE t1_00849.a = t2_00849.a AND t2_00849.a = t3_00849.a AND t3_00849.a = t4_00849.a;

ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849, t3_00849, t4_00849;
ANALYZE SELECT t1_00849.a FROM t1_00849 CROSS JOIN t2_00849 CROSS JOIN t3_00849 CROSS JOIN t4_00849;

ANALYZE SELECT t1_00849.a FROM t1_00849, t2_00849 CROSS JOIN t3_00849; -- { serverError 48 }
ANALYZE SELECT t1_00849.a FROM t1_00849 JOIN t2_00849 USING a CROSS JOIN t3_00849; -- { serverError 48 }
ANALYZE SELECT t1_00849.a FROM t1_00849 JOIN t2_00849 ON t1_00849.a = t2_00849.a CROSS JOIN t3_00849;

INSERT INTO t1_00849 values (1,1), (2,2), (3,3), (4,4);
INSERT INTO t2_00849 values (1,1), (1, Null);
INSERT INTO t3_00849 values (1,1), (1, Null);
INSERT INTO t4_00849 values (1,1), (1, Null);

SELECT 'SELECT * FROM t1, t2';
SELECT * FROM t1_00849, t2_00849
ORDER BY t1_00849.a, t2_00849.b;
SELECT 'SELECT * FROM t1, t2 WHERE t1.a = t2.a';
SELECT * FROM t1_00849, t2_00849 WHERE t1_00849.a = t2_00849.a
ORDER BY t1_00849.a, t2_00849.b;
SELECT 'SELECT t1.a, t2.a FROM t1, t2 WHERE t1.b = t2.b';
SELECT t1_00849.a, t2_00849.b FROM t1_00849, t2_00849 WHERE t1_00849.b = t2_00849.b;
SELECT 'SELECT t1.a, t2.b, t3.b FROM t1, t2, t3 WHERE t1.a = t2.a AND t1.a = t3.a';
SELECT t1_00849.a, t2_00849.b, t3_00849.b FROM t1_00849, t2_00849, t3_00849
WHERE t1_00849.a = t2_00849.a AND t1_00849.a = t3_00849.a
ORDER BY t2_00849.b, t3_00849.b;
SELECT 'SELECT t1.a, t2.b, t3.b FROM t1, t2, t3 WHERE t1.b = t2.b AND t1.b = t3.b';
SELECT t1_00849.a, t2_00849.b, t3_00849.b FROM t1_00849, t2_00849, t3_00849 WHERE t1_00849.b = t2_00849.b AND t1_00849.b = t3_00849.b;
SELECT 'SELECT t1.a, t2.b, t3.b, t4.b FROM t1, t2, t3, t4 WHERE t1.a = t2.a AND t1.a = t3.a AND t1.a = t4.a';
SELECT t1_00849.a, t2_00849.b, t3_00849.b, t4_00849.b FROM t1_00849, t2_00849, t3_00849, t4_00849
WHERE t1_00849.a = t2_00849.a AND t1_00849.a = t3_00849.a AND t1_00849.a = t4_00849.a
ORDER BY t2_00849.b, t3_00849.b, t4_00849.b;
SELECT 'SELECT t1.a, t2.b, t3.b, t4.b FROM t1, t2, t3, t4 WHERE t1.b = t2.b AND t1.b = t3.b AND t1.b = t4.b';
SELECT t1_00849.a, t2_00849.b, t3_00849.b, t4_00849.b FROM t1_00849, t2_00849, t3_00849, t4_00849
WHERE t1_00849.b = t2_00849.b AND t1_00849.b = t3_00849.b AND t1_00849.b = t4_00849.b;
SELECT 'SELECT t1.a, t2.b, t3.b, t4.b FROM t1, t2, t3, t4 WHERE t1.a = t2.a AND t2.a = t3.a AND t3.a = t4.a';
SELECT t1_00849.a, t2_00849.b, t3_00849.b, t4_00849.b FROM t1_00849, t2_00849, t3_00849, t4_00849
WHERE t1_00849.a = t2_00849.a AND t2_00849.a = t3_00849.a AND t3_00849.a = t4_00849.a
ORDER BY t2_00849.b, t3_00849.b, t4_00849.b;

DROP TABLE t1_00849;
DROP TABLE t2_00849;
DROP TABLE t3_00849;
DROP TABLE t4_00849;
