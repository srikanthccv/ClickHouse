DROP TABLE IF EXISTS partitioned_table;

CREATE TABLE partitioned_table (
    key UInt64,
    partitioner UInt8,
    value String
)
ENGINE ReplicatedMergeTree('/clickhouse/test/01650_drop_part_and_deduplication/partitioned_table', '1')
ORDER BY key
PARTITION BY partitioner;

SYSTEM STOP MERGES partitioned_table;

INSERT INTO partitioned_table VALUES (1, 1, 'A'), (2, 2, 'B'), (3, 3, 'C');
INSERT INTO partitioned_table VALUES (11, 1, 'AA'), (22, 2, 'BB'), (33, 3, 'CC');

SELECT partition_id, name FROM system.parts WHERE table = 'partitioned_table' AND database = currentDatabase() ORDER BY name;

SELECT name, value FROM system.zookeeper WHERE path='/clickhouse/test/01650_drop_part_and_deduplication/partitioned_table/blocks/';

INSERT INTO partitioned_table VALUES (33, 3, 'CC'); -- must be deduplicated

SELECT partition_id, name FROM system.parts WHERE table = 'partitioned_table' AND database = currentDatabase() ORDER BY name;

SELECT name, value FROM system.zookeeper WHERE path='/clickhouse/test/01650_drop_part_and_deduplication/partitioned_table/blocks/';

ALTER TABLE partitioned_table DROP PART '3_1_1_0';

SELECT partition_id, name FROM system.parts WHERE table = 'partitioned_table' AND database = currentDatabase() ORDER BY name;

SELECT name, value FROM system.zookeeper WHERE path='/clickhouse/test/01650_drop_part_and_deduplication/partitioned_table/blocks/';

DROP TABLE IF EXISTS partitioned_table;
