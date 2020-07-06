DROP TABLE IF EXISTS tt_01373;

CREATE TABLE tt_01373
(a Int64, d Int64, val Int64) 
ENGINE = SummingMergeTree PARTITION BY (a) ORDER BY (d);

INSERT INTO tt_01373 SELECT number%13, number%17, 1 from numbers(1000000);

OPTIMIZE TABLE tt_01373 FINAL;

SELECT a, count() FROM tt_01373 GROUP BY a ORDER BY a;

DROP TABLE IF EXISTS tt_01373;
