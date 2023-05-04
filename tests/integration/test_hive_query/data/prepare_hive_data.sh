#!/bin/bash
hive -e "create database test"

hive -e "drop table if exists test.demo; create table test.demo(id string, score int) PARTITIONED BY(day string) ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'  STORED AS INPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'; create table test.demo_orc(id string, score int) PARTITIONED BY(day string) ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.orc.OrcSerde'  STORED AS INPUTFORMAT  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' OUTPUTFORMAT  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'; "
hive -e "drop table if exists test.parquet_demo; create table test.parquet_demo(id string, score int) PARTITIONED BY(day string, hour string) ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'  STORED AS INPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'"
hive -e "drop table if exists test.demo_text; create table test.demo_text(id string, score int, day string)row format delimited fields terminated by ','; load data local inpath '/demo_data.txt' into table test.demo_text "
hive -e "set hive.exec.dynamic.partition.mode=nonstrict;insert into test.demo partition(day) select * from test.demo_text; insert into test.demo_orc partition(day) select * from test.demo_text"
hive -e "set hive.exec.dynamic.partition.mode=nonstrict;insert into test.parquet_demo partition(day, hour) select id, score, day, '00' as hour from test.demo;"
hive -e "set hive.exec.dynamic.partition.mode=nonstrict;insert into test.parquet_demo partition(day, hour) select id, score, day, '01' as hour from test.demo;"

hive -e "drop table if exists test.test_hive_types; CREATE TABLE test.test_hive_types( f_tinyint tinyint, f_smallint smallint, f_int int, f_integer int, f_bigint bigint, f_float float, f_double double, f_decimal decimal(10,0), f_timestamp timestamp, f_date date, f_string string, f_varchar varchar(100), f_char char(100), f_bool boolean, f_array_int array<int>, f_array_string array<string>, f_array_float array<float>, f_map_int map<string, int>, f_map_string map<string, string>, f_map_float map<string, float>, f_struct struct<a:string, b:int, c:float, d: struct<x:int, y:string>>) PARTITIONED BY( day string) ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'  STORED AS INPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat';"
hive -e "insert into test.test_hive_types partition(day='2022-02-20') select 1, 2, 3, 4, 5, 6.11, 7.22, 8.333, '2022-02-20 14:47:04', '2022-02-20', 'hello world', 'hello world', 'hello world', true, array(1,2,3), array('hello world', 'hello world'), array(float(1.1),float(1.2)), map('a', 100, 'b', 200, 'c', 300), map('a', 'aa', 'b', 'bb', 'c', 'cc'), map('a', float(111.1), 'b', float(222.2), 'c', float(333.3)), named_struct('a', 'aaa', 'b', 200, 'c', float(333.3), 'd', named_struct('x', 10, 'y', 'xyz')); insert into test.test_hive_types partition(day='2022-02-19') select 1, 2, 3, 4, 5, 6.11, 7.22, 8.333, '2022-02-19 14:47:04', '2022-02-19', 'hello world', 'hello world', 'hello world', true, array(1,2,3), array('hello world', 'hello world'), array(float(1.1),float(1.2)), map('a', 100, 'b', 200, 'c', 300), map('a', 'aa', 'b', 'bb', 'c', 'cc'), map('a', float(111.1), 'b', float(222.2), 'c', float(333.3)), named_struct('a', 'aaa', 'b', 200, 'c', float(333.3), 'd', named_struct('x', 11, 'y', 'abc'));"

hive -e "drop table if exists test.null_as_default_orc; create table test.null_as_default_orc (x string, y string) PARTITIONED BY(day string) ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.orc.OrcSerde' STORED AS INPUTFORMAT 'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'"
hive -e "insert into test.null_as_default_orc partition(day='2023-05-01') select null, null;"
hive -e "drop table if exists test.null_as_default_parquet; create table test.null_as_default_parquet (x string, y string) PARTITIONED BY(day string) ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' STORED AS INPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'"
hive -e "set hive.exec.dynamic.partition.mode=nonstrict; insert into test.null_as_default_parquet partition(day) select x, y, day from test.null_as_default_orc;"
