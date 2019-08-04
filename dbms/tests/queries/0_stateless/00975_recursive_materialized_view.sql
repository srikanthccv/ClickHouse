DROP TABLE IF EXISTS test.src;
DROP TABLE IF EXISTS test.dst1;
DROP TABLE IF EXISTS test.dst2;

USE test;

CREATE TABLE src (x UInt8) ENGINE Memory;
CREATE TABLE dst1 (x UInt8) ENGINE Memory;
CREATE MATERIALIZED VIEW src_to_dst1 TO dst1 AS SELECT x + 1 as x FROM src;
CREATE MATERIALIZED VIEW dst2 ENGINE Memory AS SELECT x + 1 as x FROM dst1;

INSERT INTO src VALUES (1), (2);
SELECT * FROM dst1 ORDER BY x;
SELECT * FROM dst2 ORDER BY x;

DROP TABLE src;
DROP TABLE dst1;
DROP TABLE dst2;
