DROP TABLE IF EXISTS r1;
DROP TABLE IF EXISTS r2;

CREATE TABLE r1 (x String) ENGINE = ReplicatedMergeTree('/clickhouse/tables/r', 'r1') ORDER BY x;
CREATE TABLE r2 (x String) ENGINE = ReplicatedMergeTree('/clickhouse/tables/r', 'r2') ORDER BY x;

SYSTEM STOP REPLICATED SENDS;

INSERT INTO r1 VALUES ('Hello, world');
SELECT * FROM r1;
SELECT * FROM r2;
INSERT INTO r2 VALUES ('Hello, world');
SELECT '---';
SELECT * FROM r1;
SELECT * FROM r2;

SYSTEM START REPLICATED SENDS;
SYSTEM SYNC REPLICA r1;
SYSTEM SYNC REPLICA r2;

SELECT * FROM r1;
SELECT * FROM r2;

DROP TABLE r1;
DROP TABLE r2;
