drop table if exists test.nested_map;

create table test.nested_map (d default today(), k UInt64, payload default rand(), SomeMap Nested(ID UInt32, Num Int64)) engine=SummingMergeTree(d, k, 8192);

insert into test.nested_map (k, `SomeMap.ID`, `SomeMap.Num`) values (0,[1],[100]),(1,[1],[100]),(2,[1],[100]),(3,[1,2],[100,150]);
insert into test.nested_map (k, `SomeMap.ID`, `SomeMap.Num`) values (0,[2],[150]),(1,[1],[150]),(2,[1,2],[150,150]),(3,[1],[-100]);
optimize table test.nested_map;
select `SomeMap.ID`, `SomeMap.Num` from test.nested_map;

drop table test.nested_map;

drop table if exists test.nested_map_explicit;

create table test.nested_map_explicit (d default today(), k UInt64, SomeIntExcluded UInt32, SomeMap Nested(ID UInt32, Num Int64)) engine=SummingMergeTree(d, k, 8192, (SomeMap));

insert into test.nested_map_explicit (k, `SomeIntExcluded`, `SomeMap.ID`, `SomeMap.Num`) values (0, 20, [1],[100]),(1, 20, [1],[100]),(2, 20, [1],[100]),(3, 20, [1,2],[100,150]);
insert into test.nested_map_explicit (k, `SomeIntExcluded`, `SomeMap.ID`, `SomeMap.Num`) values (0, 20, [2],[150]),(1, 20, [1],[150]),(2, 20, [1,2],[150,150]),(3, 20, [1],[-100]);
optimize table test.nested_map_explicit;
select `SomeIntExcluded`, `SomeMap.ID`, `SomeMap.Num` from test.nested_map_explicit;

drop table test.nested_map_explicit;
