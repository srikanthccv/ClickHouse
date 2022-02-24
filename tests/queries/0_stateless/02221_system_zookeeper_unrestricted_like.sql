-- Tags: zookeeper

DROP TABLE IF EXISTS sample_table;

SET allow_unrestricted_reads_from_keeper='true';

CREATE TABLE sample_table (
    key UInt64
)
ENGINE ReplicatedMergeTree('/clickhouse/{database}/02221_system_zookeeper_unrestricted_like/{shard}', '{replica}')
ORDER BY tuple();

DROP TABLE IF EXISTS sample_table_2;

CREATE TABLE sample_table_2 (
    key UInt64
)
ENGINE ReplicatedMergeTree('/clickhouse/{database}/02221_system_zookeeper_unrestricted_like_2/{shard}', '{replica}')
ORDER BY tuple();

SELECT name FROM (SELECT path, name FROM system.zookeeper WHERE path LIKE '/clickhouse%' ORDER BY name) WHERE path LIKE '%02221_system_zookeeper_unrestricted_like%';

SELECT '-------------------------';

SELECT name FROM (SELECT path, name FROM system.zookeeper WHERE path LIKE '/clickhouse/%' ORDER BY name) WHERE path LIKE '%02221_system_zookeeper_unrestricted_like%';

DROP TABLE IF EXISTS sample_table;
DROP TABLE IF EXISTS sample_table_2;
