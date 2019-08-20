drop table if exists funnel_test;

create table funnel_test (timestamp UInt32, event UInt32) engine=Memory;
insert into funnel_test values (0,1000),(1,1001),(2,1002),(3,1003),(4,1004),(5,1005),(6,1006),(7,1007),(8,1008);

select 1 = windowFunnel(10000)(timestamp, event = 1000) from funnel_test;
select 2 = windowFunnel(10000)(timestamp, event = 1000, event = 1001) from funnel_test;
select 3 = windowFunnel(10000)(timestamp, event = 1000, event = 1001, event = 1002) from funnel_test;
select 4 = windowFunnel(10000)(timestamp, event = 1000, event = 1001, event = 1002, event = 1008) from funnel_test;


select 1 = windowFunnel(1)(timestamp, event = 1000) from funnel_test;
select 3 = windowFunnel(2)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test;
select 4 = windowFunnel(3)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test;
select 5 = windowFunnel(4)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test;


drop table if exists funnel_test2;
create table funnel_test2 (uid UInt32 default 1,timestamp DateTime, event UInt32) engine=Memory;
insert into funnel_test2(timestamp, event) values  ('2018-01-01 01:01:01',1001),('2018-01-01 01:01:02',1002),('2018-01-01 01:01:03',1003),('2018-01-01 01:01:04',1004),('2018-01-01 01:01:05',1005),('2018-01-01 01:01:06',1006),('2018-01-01 01:01:07',1007),('2018-01-01 01:01:08',1008);


select 5 = windowFunnel(4)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test2;
select 2 = windowFunnel(10000)(timestamp, event = 1001, event = 1008) from funnel_test2;
select 1 = windowFunnel(10000)(timestamp, event = 1008, event = 1001) from funnel_test2;
select 5 = windowFunnel(4)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test2;
select 4 = windowFunnel(4)(timestamp, event <= 1007, event >= 1002, event <= 1006, event >= 1004) from funnel_test2;


drop table if exists funnel_test_u64;
create table funnel_test_u64 (uid UInt32 default 1,timestamp UInt64, event UInt32) engine=Memory;
insert into funnel_test_u64(timestamp, event) values  ( 1e14 + 1 ,1001),(1e14 + 2,1002),(1e14 + 3,1003),(1e14 + 4,1004),(1e14 + 5,1005),(1e14 + 6,1006),(1e14 + 7,1007),(1e14 + 8,1008);


select 5 = windowFunnel(4)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test_u64;
select 2 = windowFunnel(10000)(timestamp, event = 1001, event = 1008) from funnel_test_u64;
select 1 = windowFunnel(10000)(timestamp, event = 1008, event = 1001) from funnel_test_u64;
select 5 = windowFunnel(4)(timestamp, event = 1003, event = 1004, event = 1005, event = 1006, event = 1007) from funnel_test_u64;
select 4 = windowFunnel(4)(timestamp, event <= 1007, event >= 1002, event <= 1006, event >= 1004) from funnel_test_u64;


drop table if exists funnel_test_strict;
create table funnel_test_strict (timestamp UInt32, event UInt32) engine=Memory;
insert into funnel_test_strict values (00,1000),(10,1001),(20,1002),(30,1003),(40,1004),(50,1005),(51,1005),(60,1006),(70,1007),(80,1008);

select 6 = windowFunnel(10000, 'strict')(timestamp, event = 1000, event = 1001, event = 1002, event = 1003, event = 1004, event = 1005, event = 1006) from funnel_test_strict;
select 7 = windowFunnel(10000)(timestamp, event = 1000, event = 1001, event = 1002, event = 1003, event = 1004, event = 1005, event = 1006) from funnel_test_strict;


drop table funnel_test;
drop table funnel_test2;
drop table funnel_test_u64;
drop table funnel_test_strict;
