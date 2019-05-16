CREATE DATABASE IF NOT EXISTS test;
DROP TABLE IF EXISTS test.defaults;
CREATE TABLE IF NOT EXISTS test.defaults
(
    param1 Float64,
    param2 Float64,
    target Float64,
    predict1 Float64,
    predict2 Float64
) ENGINE = Memory;
insert into test.defaults values (1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2),(1,2,1,-1,-2),(-1,-2,-1,1,2)
DROP TABLE IF EXISTS test.model;
create table test.model engine = Memory as select LogisticRegressionState(0.1, 0.0, 1.0, 'SGD')(target, param1, param2) as state from test.defaults;

select ans < 1.1 and ans > 0.9 from
(with (select state from test.model) as model select evalMLMethod(model, predict1, predict2) as ans from test.defaults limit 2);

select ans > -0.1 and ans < 0.1 from
(with (select state from test.model) as model select evalMLMethod(model, predict1, predict2) as ans from test.defaults limit 2);
