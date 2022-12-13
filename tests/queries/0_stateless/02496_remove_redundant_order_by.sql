SET allow_experimental_analyzer=1;
SET optimize_duplicate_order_by_and_distinct=0;

SELECT '-- Disable query_plan_remove_redundant_order_by';
SET query_plan_remove_redundant_order_by=0;

SELECT '-- ORDER BY clauses in subqueries are untouched';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
)
ORDER BY number ASC;

SELECT '-- Enable query_plan_remove_redundant_order_by';
SET query_plan_remove_redundant_order_by=1;

SELECT '-- ORDER BY removes ORDER BY clauses in subqueries';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
)
ORDER BY number ASC;

SELECT *
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
)
ORDER BY number ASC;

SELECT '-- ORDER BY cannot remove ORDER BY in subquery WITH FILL';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number DESC
    )
    ORDER BY number ASC WITH FILL STEP 1
)
ORDER BY number ASC;

SELECT '-- ORDER BY cannot remove ORDER BY in subquery with LIMIT BY';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number DESC
    )
    ORDER BY number ASC
    LIMIT 1 BY number
)
ORDER BY number ASC;

SELECT '-- CROSS JOIN with subqueries, nor ORDER BY nor GROUP BY in main query -> only ORDER BY clauses in most inner subqueries will be removed';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT number
    FROM
    (
        SELECT number
        FROM numbers(3)
        ORDER BY number DESC
    )
    ORDER BY number ASC
) AS t1,
(
    SELECT number
    FROM
    (
        SELECT number
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
) AS t2;

SELECT '-- CROSS JOIN with subqueries, ORDER BY in main query -> all ORDER BY clauses will be removed in subqueries';
EXPLAIN header=1
SELECT *
FROM
(
    SELECT number
    FROM
    (
        SELECT number
        FROM numbers(3)
        ORDER BY number DESC
    )
    ORDER BY number ASC
) AS t1,
(
    SELECT number
    FROM
    (
        SELECT number
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
) AS t2
ORDER BY t1.number ASC;

SELECT '-- GROUP BY with aggregation function which does NOT depend on order -> eliminate ORDER BY(s) in _all_ subqueries';
EXPLAIN header=1
SELECT sum(number)
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
)
GROUP BY number;

SELECT '-- GROUP BY with aggregation function which depends on order -> keep ORDER BY in first subquery, and eliminate in second subquery';
EXPLAIN header=1
SELECT any(number)
FROM
(
    SELECT *
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    ORDER BY number DESC
)
GROUP BY number;

SELECT '-- check that optimization is applied recursively to subqueries as well';
SELECT '-- GROUP BY with aggregation function which does NOT depend on order -> eliminate ORDER BY in most inner subquery here';
EXPLAIN header=1
SELECT a
FROM
(
    SELECT sum(number) AS a
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    GROUP BY number
)
ORDER BY a ASC;

SELECT '-- GROUP BY with aggregation function which depends on order -> ORDER BY in subquery is kept due to the aggregation function';
EXPLAIN header=1
SELECT a
FROM
(
    SELECT any(number) AS a
    FROM
    (
        SELECT *
        FROM numbers(3)
        ORDER BY number ASC
    )
    GROUP BY number
)
ORDER BY a ASC;

SELECT '-- Check that optimization works for subqueries as well, - main query have neither ORDER BY nor GROUP BY';
EXPLAIN header=1
SELECT a
FROM
(
    SELECT any(number) AS a
    FROM
    (
        SELECT *
        FROM
        (
            SELECT *
            FROM numbers(3)
            ORDER BY number DESC
        )
        ORDER BY number ASC
    )
    GROUP BY number
)
WHERE a > 0;

SELECT '-- disable common optimization to avoid functions to be lifted up (liftUpFunctions optimization), needed for testing with stateful function';
SET query_plan_enable_optimizations = 0;

SELECT '-- neighbor() as stateful function prevents removing inner ORDER BY since its result depends on order';
EXPLAIN
SELECT
    number,
    neighbor(number, 2)
FROM
(
    SELECT *
    FROM numbers(10)
    ORDER BY number DESC
)
ORDER BY number ASC;

SELECT '-- non-stateful function does _not_ prevent removing inner ORDER BY';
EXPLAIN
SELECT
    number,
    plus(number, 2)
FROM
(
    SELECT *
    FROM numbers(10)
    ORDER BY number DESC
)
ORDER BY number ASC;
