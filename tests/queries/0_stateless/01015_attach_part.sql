-- Tags: no-replicated-database, no-parallel
-- Tag no-replicated-database: Unsupported type of ALTER query

DROP TABLE IF EXISTS table_01;

CREATE TABLE table_01 (
    date Date,
    n Int32
) ENGINE = MergeTree()
PARTITION BY date
ORDER BY date;

INSERT INTO table_01 SELECT toDate('2019-10-01'), number FROM system.numbers LIMIT 1000;

SELECT COUNT() FROM table_01;

ALTER TABLE table_01 DETACH PARTITION ID '20191001';

SELECT COUNT() FROM table_01;

ALTER TABLE table_01 ATTACH PART '20191001_1_1_0';

SELECT COUNT() FROM table_01;

DROP TABLE IF EXISTS table_01;
