DROP TABLE IF EXISTS test;
CREATE TABLE test(a Int, b Int) Engine=ReplacingMergeTree order by a;
INSERT INTO test select number, number from numbers(5);
INSERT INTO test select number, number from numbers(5,2);
set max_threads =1;
explain pipeline select * from test final;
select * from test final;
set max_threads =2;
explain pipeline select * from test final;
DROP TABLE test;
