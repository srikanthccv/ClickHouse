SELECT *
FROM
(
    SELECT 1 AS a
) AS t1
INNER JOIN
(
    SELECT 1 AS a
    GROUP BY 1
        WITH TOTALS
    UNION ALL
    SELECT 1
    GROUP BY 1
        WITH TOTALS
) AS t2 USING (a)
