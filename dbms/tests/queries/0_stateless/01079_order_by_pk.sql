DROP TABLE IF EXISTS mt_pk;

CREATE TABLE mt_pk ENGINE = MergeTree PARTITION BY d ORDER BY x
AS SELECT toDate(number % 32) AS d, number AS x FROM system.numbers LIMIT 10000010;
SELECT x FROM mt_pk ORDER BY x ASC LIMIT 10000000, 1;

DROP TABLE mt_pk;
