DROP TABLE IF EXISTS perf;
CREATE TABLE perf (site String, user_id UInt64, z Float64) ENGINE = Log;

SELECT * FROM (SELECT perf_1.z AS z_1 FROM perf AS perf_1);

SELECT sum(mul)/sqrt(sum(sqr_dif_1) * sum(sqr_dif_2)) AS z_r
FROM(
SELECT 
        (SELECT avg(z_1) AS z_1_avg, 
                avg(z_2) AS z_2_avg
        FROM ( 
            SELECT perf_1.site, perf_1.z AS z_1
            FROM perf AS perf_1
            WHERE user_id = 000
        ) jss1 ALL INNER JOIN (
            SELECT perf_2.site, perf_2.z AS z_2
            FROM perf AS perf_2
            WHERE user_id = 999
        ) jss2 USING site) as avg_values,
       z_1 - avg_values.1 AS dif_1, 
       z_2 - avg_values.2 AS dif_2, 
       dif_1 * dif_2 AS mul, 
       dif_1*dif_1 AS sqr_dif_1, 
       dif_2*dif_2 AS sqr_dif_2
FROM (
            SELECT perf_1.site, perf_1.z AS z_1
            FROM perf AS perf_1
            WHERE user_id = 000
) js1 ALL INNER JOIN (
            SELECT perf_2.site, perf_2.z AS z_2
            FROM perf AS perf_2
            WHERE user_id = 999
) js2 USING site);

-- check order is preserved
SET enable_debug_queries = 1;
ANALYZE SELECT * FROM system.one HAVING dummy > 0 AND dummy < 0;

-- from #10613
SELECT name, count() AS cnt
FROM remote('127.{1,2}', system.settings)
GROUP BY name
HAVING (max(value) > '9') AND (min(changed) = 0)
FORMAT Null;

DROP TABLE perf;
