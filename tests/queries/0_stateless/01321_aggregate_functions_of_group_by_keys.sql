set optimize_aggregators_of_group_by_keys = 1;

SELECT min(number % 2) AS a, max(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
SELECT any(number % 2) AS a, anyLast(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
SELECT max((number % 5) * (number % 7)) AS a FROM numbers(10000000) GROUP BY number % 7, number % 5 ORDER BY a;
SELECT foo FROM (SELECT anyLast(number) AS foo FROM numbers(1) GROUP BY number);
SELECT anyLast(number) FROM numbers(1) GROUP BY number;

EXPLAIN SYNTAX SELECT min(number % 2) AS a, max(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
EXPLAIN SYNTAX SELECT any(number % 2) AS a, anyLast(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
EXPLAIN SYNTAX SELECT max((number % 5) * (number % 7)) AS a FROM numbers(10000000) GROUP BY number % 7, number % 5 ORDER BY a;
EXPLAIN SYNTAX SELECT foo FROM (SELECT anyLast(number) AS foo FROM numbers(1) GROUP BY number);

set optimize_aggregators_of_group_by_keys = 0;

SELECT min(number % 2) AS a, max(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
SELECT any(number % 2) AS a, anyLast(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
SELECT max((number % 5) * (number % 7)) AS a FROM numbers(10000000) GROUP BY number % 7, number % 5 ORDER BY a;
SELECT foo FROM (SELECT anyLast(number) AS foo FROM numbers(1) GROUP BY number);

EXPLAIN SYNTAX SELECT min(number % 2) AS a, max(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
EXPLAIN SYNTAX SELECT any(number % 2) AS a, anyLast(number % 3) AS b FROM numbers(10000000) GROUP BY number % 2, number % 3 ORDER BY a, b;
EXPLAIN SYNTAX SELECT max((number % 5) * (number % 7)) AS a FROM numbers(10000000) GROUP BY number % 7, number % 5 ORDER BY a;
EXPLAIN SYNTAX SELECT foo FROM (SELECT anyLast(number) AS foo FROM numbers(1) GROUP BY number);
