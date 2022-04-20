set enable_parallel_hash_join=true;
SELECT
    EventDate,
    hits,
    visits
FROM
(
    SELECT
        EventDate,
        count() AS hits
    FROM test.hits
    GROUP BY EventDate
) ANY LEFT JOIN
(
    SELECT
        StartDate AS EventDate,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY EventDate
) USING EventDate
ORDER BY hits DESC
LIMIT 10
SETTINGS joined_subquery_requires_alias = 0;


SELECT
    EventDate,
    count() AS hits,
    any(visits)
FROM test.hits ANY LEFT JOIN
(
    SELECT
        StartDate AS EventDate,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY EventDate
) USING EventDate
GROUP BY EventDate
ORDER BY hits DESC
LIMIT 10
SETTINGS joined_subquery_requires_alias = 0, enable_parallel_hash_join=true;


SELECT
    domain,
    hits,
    visits
FROM
(
    SELECT
        domain(URL) AS domain,
        count() AS hits
    FROM test.hits
    GROUP BY domain
) ANY LEFT JOIN
(
    SELECT
        domain(StartURL) AS domain,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY domain
) USING domain
ORDER BY hits DESC
LIMIT 10
SETTINGS joined_subquery_requires_alias = 0;

SELECT CounterID FROM test.visits ARRAY JOIN Goals.ID WHERE CounterID = 942285 ORDER BY CounterID;


SELECT
    CounterID,
    hits,
    visits
FROM
(
    SELECT
        (CounterID % 100000) AS CounterID,
        count() AS hits
    FROM test.hits
    GROUP BY CounterID
) ANY FULL OUTER JOIN
(
    SELECT
        (CounterID % 100000) AS CounterID,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY CounterID
    HAVING visits > 0
) USING CounterID
WHERE hits = 0 OR visits = 0
ORDER BY
    hits + visits * 10 DESC,
    CounterID ASC
LIMIT 20
SETTINGS any_join_distinct_right_table_keys = 1, joined_subquery_requires_alias = 0;


SELECT
    CounterID,
    hits,
    visits
FROM
(
    SELECT
        (CounterID % 100000) AS CounterID,
        count() AS hits
    FROM test.hits
    GROUP BY CounterID
) ANY LEFT JOIN
(
    SELECT
        (CounterID % 100000) AS CounterID,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY CounterID
    HAVING visits > 0
) USING CounterID
WHERE hits = 0 OR visits = 0
ORDER BY
    hits + visits * 10 DESC,
    CounterID ASC
LIMIT 20
SETTINGS any_join_distinct_right_table_keys = 1, joined_subquery_requires_alias = 0;


SELECT
    CounterID,
    hits,
    visits
FROM
(
    SELECT
        (CounterID % 100000) AS CounterID,
        count() AS hits
    FROM test.hits
    GROUP BY CounterID
) ANY RIGHT JOIN
(
    SELECT
        (CounterID % 100000) AS CounterID,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY CounterID
    HAVING visits > 0
) USING CounterID
WHERE hits = 0 OR visits = 0
ORDER BY
    hits + visits * 10 DESC,
    CounterID ASC
LIMIT 20
SETTINGS any_join_distinct_right_table_keys = 1, joined_subquery_requires_alias = 0;


SELECT
    CounterID,
    hits,
    visits
FROM
(
    SELECT
        (CounterID % 100000) AS CounterID,
        count() AS hits
    FROM test.hits
    GROUP BY CounterID
) ANY INNER JOIN
(
    SELECT
        (CounterID % 100000) AS CounterID,
        sum(Sign) AS visits
    FROM test.visits
    GROUP BY CounterID
    HAVING visits > 0
) USING CounterID
WHERE hits = 0 OR visits = 0
ORDER BY
    hits + visits * 10 DESC,
    CounterID ASC
LIMIT 20
SETTINGS any_join_distinct_right_table_keys = 1, joined_subquery_requires_alias = 0;

SELECT UserID, pp.Key1, pp.Key2, ParsedParams.Key1 FROM test.hits ARRAY JOIN ParsedParams AS pp WHERE CounterID = 1704509 ORDER BY UserID, EventTime, pp.Key1, pp.Key2 LIMIT 100;

SELECT UserID, pp.Key1, pp.Key2, ParsedParams.Key1 FROM test.hits LEFT ARRAY JOIN ParsedParams AS pp WHERE CounterID = 1704509 ORDER BY UserID, EventTime, pp.Key1, pp.Key2 LIMIT 100;

SELECT a.*, b.* FROM
(
    SELECT number AS k FROM system.numbers LIMIT 10
) AS a
ANY INNER JOIN
(
    SELECT number * 2 AS k, number AS joined FROM system.numbers LIMIT 10
) AS b
USING k
SETTINGS any_join_distinct_right_table_keys = 1;

SELECT a.*, b.* FROM
(
    SELECT number AS k FROM system.numbers LIMIT 10
) AS a
ALL INNER JOIN
(
    SELECT intDiv(number, 2) AS k, number AS joined FROM system.numbers LIMIT 10
) AS b
USING k;
