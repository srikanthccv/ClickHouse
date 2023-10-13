DROP TABLE IF EXISTS test;

CREATE TABLE test
(
  EventDate Date
)
ENGINE = MergeTree
ORDER BY tuple()
PARTITION BY toMonday(EventDate);

INSERT INTO test VALUES(toDate('2023-10-09'));

SET param_partition='2023-10-09';

ALTER TABLE test DROP PARTITION {partition:String};

SELECT count() FROM test;

INSERT INTO test VALUES(toDate('2023-10-09'));

ALTER TABLE test DROP PARTITION tuple(toMonday({partition:Date}));

SELECT count() FROM test;

INSERT INTO test VALUES(toDate('2023-10-09'));

-- for some reason only tuples are allowed as non-string arguments
ALTER TABLE test DROP PARTITION toMonday({partition:String}); --{clientError 62}

set param_partition_id = '20231009';

ALTER TABLE test DROP PARTITION ID {partition_id:String};

SELECT count() FROM test;

DROP TABLE IF EXISTS test;

DROP TABLE IF EXISTS test2;

CREATE TABLE test2
(
  a UInt32,
  b Int64
)
ENGINE = MergeTree
ORDER BY tuple()
PARTITION BY (a * b, b * b);


INSERT INTO test2 VALUES(1, 2);

SET param_first='2';
SET param_second='4';

ALTER TABLE test2 DROP PARTITION tuple({first:UInt32},{second:Int64});

SELECT count() FROM test2;

DROP TABLE IF EXISTS test2;
DROP TABLE IF EXISTS test3;

CREATE TABLE test3
(
  a UInt32,
  b Int64
)
ENGINE = MergeTree
ORDER BY tuple()
PARTITION BY a;

INSERT INTO test3 VALUES(1, 2);

SET param_simple='1';

ALTER TABLE test3 DROP PARTITION {simple:String};

SELECT count() FROM test3;

DROP TABLE IF EXISTS test3;


DROP TABLE IF EXISTS test4 ON CLUSTER 'test_shard_localhost';

CREATE TABLE test4 ON CLUSTER 'test_shard_localhost' (EventDate Date) ENGINE = MergeTree() ORDER BY tuple() PARTITION BY EventDate;

INSERT INTO test4 VALUES(toDate('2023-10-09'));

SET param_partition='2023-10-09';

ALTER TABLE test4 ON CLUSTER 'test_shard_localhost' DROP PARTITION {partition:String};

SELECT count() FROM test4;

DROP TABLE IF EXISTS test4 ON CLUSTER 'test_shard_localhost';
