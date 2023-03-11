DROP TABLE IF EXISTS t1;
CREATE TABLE t1(ID UInt64, name String) engine=MergeTree order by ID;

insert into t1(ID, name) values (1, 'abc'), (2, 'bbb');

explain pipeline graph=1 select count(ID) from t1;
explain pipeline graph=1 select sum(1) from t1;
explain pipeline graph=1 select min(ID) from t1;
explain pipeline graph=1 select max(ID) from t1;

DROP TABLE t1;
