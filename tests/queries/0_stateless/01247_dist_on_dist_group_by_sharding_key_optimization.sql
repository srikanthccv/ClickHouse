-- TODO: correct testing with real unique shards

drop table if exists dist_01247;
drop table if exists dist_layer_01247;
drop table if exists data_01247;

create table data_01247 as system.numbers engine=Memory();
-- since data is not inserted via distributed it will have duplicates
-- (and this is how we ensure that this optimization will work)
insert into data_01247 select * from system.numbers limit 2;

set max_distributed_connections=1;
set optimize_skip_unused_shards=1;

select 'Distributed(number)-over-Distributed(number)';
create table dist_layer_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), data_01247, number);
create table dist_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), dist_layer_01247, number);
select count(), * from dist_01247 group by number;
drop table if exists dist_01247;
drop table if exists dist_layer_01247;

select 'Distributed(rand)-over-Distributed(number)';
create table dist_layer_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), data_01247, number);
create table dist_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), dist_layer_01247, rand());
select count(), * from dist_01247 group by number;
drop table if exists dist_01247;
drop table if exists dist_layer_01247;

select 'Distributed(rand)-over-Distributed(rand)';
create table dist_layer_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), data_01247, rand());
create table dist_01247 as data_01247 engine=Distributed(test_cluster_two_shards, currentDatabase(), dist_layer_01247, number);
select count(), * from dist_01247 group by number;
drop table if exists dist_01247;
drop table if exists dist_layer_01247;
