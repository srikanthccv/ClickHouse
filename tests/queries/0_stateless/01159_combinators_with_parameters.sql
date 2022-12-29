SELECT toTypeName(topKArrayState(10)([toString(number)])) FROM numbers(100);
SELECT toTypeName(topKDistinctState(10)(toString(number))) FROM numbers(100);
SELECT toTypeName(topKForEachState(10)([toString(number)])) FROM numbers(100);
SELECT toTypeName(topKIfState(10)(toString(number), number % 2)) FROM numbers(100);
SELECT toTypeName(topKMergeState(10)(state)) FROM (SELECT topKState(10)(toString(number)) as state FROM numbers(100));
SELECT toTypeName(topKOrNullState(10)(toString(number))) FROM numbers(100); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT toTypeName(topKOrDefaultState(10)(toString(number))) FROM numbers(100);
SELECT toTypeName(topKResampleState(10, 1, 2, 42)(toString(number), number)) FROM numbers(100);
SELECT toTypeName(topKState(10)(toString(number))) FROM numbers(100);
SELECT toTypeName(topKArrayResampleOrDefaultIfState(10, 1, 2, 42)([toString(number)], number, number % 2)) FROM numbers(100);

CREATE TEMPORARY TABLE t0 AS SELECT quantileArrayState(0.10)([number]) FROM numbers(100);
CREATE TEMPORARY TABLE t1 AS SELECT quantileDistinctState(0.10)(number) FROM numbers(100);
CREATE TEMPORARY TABLE t2 AS SELECT quantileForEachState(0.10)([number]) FROM numbers(100);
CREATE TEMPORARY TABLE t3 AS SELECT quantileIfState(0.10)(number, number % 2) FROM numbers(100);
CREATE TEMPORARY TABLE t4 AS SELECT quantileMergeState(0.10)(state) FROM (SELECT quantileState(0.10)(number) as state FROM numbers(100));
CREATE TEMPORARY TABLE t5 AS SELECT quantileOrNullState(0.10)(number) FROM numbers(100);
CREATE TEMPORARY TABLE t6 AS SELECT quantileOrDefaultState(0.10)(number) FROM numbers(100);
CREATE TEMPORARY TABLE t7 AS SELECT quantileResampleState(0.10, 1, 2, 42)(number, number) FROM numbers(100);
CREATE TEMPORARY TABLE t8 AS SELECT quantileState(0.10)(number) FROM numbers(100);
CREATE TEMPORARY TABLE t9 AS SELECT quantileArrayResampleOrDefaultIfState(0.10, 1, 2, 42)([number], number, number % 2) FROM numbers(100);

INSERT INTO t0 SELECT quantileArrayState(0.10)([number]) FROM numbers(100);
INSERT INTO t1 SELECT quantileDistinctState(0.10)(number) FROM numbers(100);
INSERT INTO t2 SELECT quantileForEachState(0.10)([number]) FROM numbers(100);
INSERT INTO t3 SELECT quantileIfState(0.10)(number, number % 2) FROM numbers(100);
INSERT INTO t4 SELECT quantileMergeState(0.10)(state) FROM (SELECT quantileState(0.10)(number) as state FROM numbers(100));
INSERT INTO t5 SELECT quantileOrNullState(0.10)(number) FROM numbers(100);
INSERT INTO t6 SELECT quantileOrDefaultState(0.10)(number) FROM numbers(100);
INSERT INTO t7 SELECT quantileResampleState(0.10, 1, 2, 42)(number, number) FROM numbers(100);
INSERT INTO t8 SELECT quantileState(0.10)(number) FROM numbers(100);
INSERT INTO t9 SELECT quantileArrayResampleOrDefaultIfState(0.10, 1, 2, 42)([number], number, number % 2) FROM numbers(100);

SELECT round(quantileArrayMerge(0.10)((*,).1)) FROM t0;
SELECT round(quantileDistinctMerge(0.10)((*,).1)) FROM t1;
SELECT arrayMap(x -> round(x), quantileForEachMerge(0.10)((*,).1)) FROM t2;
SELECT round(quantileIfMerge(0.10)((*,).1)) FROM t3;
SELECT round(quantileMerge(0.10)((*,).1)) FROM t4;
SELECT round(quantileOrNullMerge(0.10)((*,).1)) FROM t5;
SELECT round(quantileOrDefaultMerge(0.10)((*,).1)) FROM t6;
SELECT arrayMap(x -> round(x), quantileResampleMerge(0.10, 1, 2, 42)((*,).1)) FROM t7;
SELECT round(quantileMerge(0.10)((*,).1)) FROM t8;
SELECT arrayMap(x -> round(x), quantileArrayResampleOrDefaultIfMerge(0.10, 1, 2, 42)((*,).1)) FROM t9;
