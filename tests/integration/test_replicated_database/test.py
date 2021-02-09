import time
import re
import pytest

from helpers.cluster import ClickHouseCluster
from helpers.test_tools import assert_eq_with_retry

cluster = ClickHouseCluster(__file__)

main_node = cluster.add_instance('main_node', main_configs=['configs/config.xml'], with_zookeeper=True, stay_alive=True, macros={"shard": 1, "replica": 1})
dummy_node = cluster.add_instance('dummy_node', main_configs=['configs/config.xml'], with_zookeeper=True, stay_alive=True, macros={"shard": 1, "replica": 2})
competing_node = cluster.add_instance('competing_node', main_configs=['configs/config.xml'], with_zookeeper=True, macros={"shard": 1, "replica": 3})
snapshotting_node = cluster.add_instance('snapshotting_node', main_configs=['configs/config.xml'], with_zookeeper=True, macros={"shard": 2, "replica": 1})
snapshot_recovering_node = cluster.add_instance('snapshot_recovering_node', main_configs=['configs/config.xml'], with_zookeeper=True, macros={"shard": 2, "replica": 2})

all_nodes = [main_node, dummy_node, competing_node, snapshotting_node, snapshot_recovering_node]

uuid_regex = re.compile("[0-9a-f]{8}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{12}")
def assert_create_query(nodes, table_name, expected):
    replace_uuid = lambda x: re.sub(uuid_regex, "uuid", x)
    query = "show create table {}".format(table_name)
    for node in nodes:
        assert_eq_with_retry(node, query, expected, get_result=replace_uuid)

@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        main_node.query("CREATE DATABASE testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard1', 'replica1');")
        dummy_node.query("CREATE DATABASE testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard1', 'replica2');")
        yield cluster

    finally:
        cluster.shutdown()

def test_create_replicated_table(started_cluster):
    assert "Old syntax is not allowed" in \
           main_node.query_and_get_error("CREATE TABLE testdb.replicated_table (d Date, k UInt64, i32 Int32) ENGINE=ReplicatedMergeTree('/test/tmp', 'r', d, k, 8192);")

    main_node.query("CREATE TABLE testdb.replicated_table (d Date, k UInt64, i32 Int32) ENGINE=ReplicatedMergeTree ORDER BY k PARTITION BY toYYYYMM(d);")

    expected = "CREATE TABLE testdb.replicated_table\\n(\\n    `d` Date,\\n    `k` UInt64,\\n    `i32` Int32\\n)\\n" \
               "ENGINE = ReplicatedMergeTree(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')\\n" \
               "PARTITION BY toYYYYMM(d)\\nORDER BY k\\nSETTINGS index_granularity = 8192"
    assert_create_query([main_node, dummy_node], "testdb.replicated_table", expected)
    # assert without replacing uuid
    assert main_node.query("show create testdb.replicated_table") == dummy_node.query("show create testdb.replicated_table")

@pytest.mark.parametrize("engine", ['MergeTree', 'ReplicatedMergeTree'])
def test_simple_alter_table(started_cluster, engine):
    # test_simple_alter_table
    name  = "testdb.alter_test_{}".format(engine)
    main_node.query("CREATE TABLE {} "
                    "(CounterID UInt32, StartDate Date, UserID UInt32, VisitID UInt32, NestedColumn Nested(A UInt8, S String), ToDrop UInt32) "
                    "ENGINE = {} PARTITION BY StartDate ORDER BY (CounterID, StartDate, intHash32(UserID), VisitID);".format(name, engine))
    main_node.query("ALTER TABLE {} ADD COLUMN Added0 UInt32;".format(name))
    main_node.query("ALTER TABLE {} ADD COLUMN Added2 UInt32;".format(name))
    main_node.query("ALTER TABLE {} ADD COLUMN Added1 UInt32 AFTER Added0;".format(name))
    main_node.query("ALTER TABLE {} ADD COLUMN AddedNested1 Nested(A UInt32, B UInt64) AFTER Added2;".format(name))
    main_node.query("ALTER TABLE {} ADD COLUMN AddedNested1.C Array(String) AFTER AddedNested1.B;".format(name))
    main_node.query("ALTER TABLE {} ADD COLUMN AddedNested2 Nested(A UInt32, B UInt64) AFTER AddedNested1;".format(name))

    full_engine = engine if not "Replicated" in engine else engine + "(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')"
    expected = "CREATE TABLE {}\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n" \
               "    `ToDrop` UInt32,\\n    `Added0` UInt32,\\n    `Added1` UInt32,\\n    `Added2` UInt32,\\n" \
               "    `AddedNested1.A` Array(UInt32),\\n    `AddedNested1.B` Array(UInt64),\\n    `AddedNested1.C` Array(String),\\n" \
               "    `AddedNested2.A` Array(UInt32),\\n    `AddedNested2.B` Array(UInt64)\\n)\\n" \
               "ENGINE = {}\\nPARTITION BY StartDate\\nORDER BY (CounterID, StartDate, intHash32(UserID), VisitID)\\n" \
               "SETTINGS index_granularity = 8192".format(name, full_engine)

    assert_create_query([main_node, dummy_node], name, expected)

    # test_create_replica_after_delay
    competing_node.query("CREATE DATABASE IF NOT EXISTS testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard1', 'replica3');")

    name  = "testdb.alter_test_{}".format(engine)
    main_node.query("ALTER TABLE {} ADD COLUMN Added3 UInt32;".format(name))
    main_node.query("ALTER TABLE {} DROP COLUMN AddedNested1;".format(name))
    main_node.query("ALTER TABLE {} RENAME COLUMN Added1 TO AddedNested1;".format(name))

    full_engine = engine if not "Replicated" in engine else engine + "(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')"
    expected = "CREATE TABLE {}\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n" \
               "    `ToDrop` UInt32,\\n    `Added0` UInt32,\\n    `AddedNested1` UInt32,\\n    `Added2` UInt32,\\n" \
               "    `AddedNested2.A` Array(UInt32),\\n    `AddedNested2.B` Array(UInt64),\\n    `Added3` UInt32\\n)\\n" \
               "ENGINE = {}\\nPARTITION BY StartDate\\nORDER BY (CounterID, StartDate, intHash32(UserID), VisitID)\\n" \
               "SETTINGS index_granularity = 8192".format(name, full_engine)

    assert_create_query([main_node, dummy_node, competing_node], name, expected)


def test_alters_from_different_replicas(started_cluster):
    # test_alters_from_different_replicas
    competing_node.query("CREATE DATABASE IF NOT EXISTS testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard1', 'replica3');")

    main_node.query("CREATE TABLE testdb.concurrent_test "
                    "(CounterID UInt32, StartDate Date, UserID UInt32, VisitID UInt32, NestedColumn Nested(A UInt8, S String), ToDrop UInt32) "
                    "ENGINE = MergeTree(StartDate, intHash32(UserID), (CounterID, StartDate, intHash32(UserID), VisitID), 8192);")

    main_node.query("CREATE TABLE testdb.dist AS testdb.concurrent_test ENGINE = Distributed(cluster, testdb, concurrent_test, CounterID)")

    dummy_node.stop_clickhouse(kill=True)

    settings = {"distributed_ddl_task_timeout": 10}
    assert "There are 1 unfinished hosts (0 of them are currently active)" in \
        competing_node.query_and_get_error("ALTER TABLE testdb.concurrent_test ADD COLUMN Added0 UInt32;", settings=settings)
    dummy_node.start_clickhouse()
    main_node.query("ALTER TABLE testdb.concurrent_test ADD COLUMN Added2 UInt32;")
    competing_node.query("ALTER TABLE testdb.concurrent_test ADD COLUMN Added1 UInt32 AFTER Added0;")
    main_node.query("ALTER TABLE testdb.concurrent_test ADD COLUMN AddedNested1 Nested(A UInt32, B UInt64) AFTER Added2;")
    competing_node.query("ALTER TABLE testdb.concurrent_test ADD COLUMN AddedNested1.C Array(String) AFTER AddedNested1.B;")
    main_node.query("ALTER TABLE testdb.concurrent_test ADD COLUMN AddedNested2 Nested(A UInt32, B UInt64) AFTER AddedNested1;")

    expected = "CREATE TABLE testdb.concurrent_test\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n    `ToDrop` UInt32,\\n" \
               "    `Added0` UInt32,\\n    `Added1` UInt32,\\n    `Added2` UInt32,\\n    `AddedNested1.A` Array(UInt32),\\n" \
               "    `AddedNested1.B` Array(UInt64),\\n    `AddedNested1.C` Array(String),\\n    `AddedNested2.A` Array(UInt32),\\n" \
               "    `AddedNested2.B` Array(UInt64)\\n)\\n" \
               "ENGINE = MergeTree(StartDate, intHash32(UserID), (CounterID, StartDate, intHash32(UserID), VisitID), 8192)"

    assert_create_query([main_node, competing_node], "testdb.concurrent_test", expected)

    # test_create_replica_after_delay
    main_node.query("DROP TABLE testdb.concurrent_test")
    main_node.query("CREATE TABLE testdb.concurrent_test "
                    "(CounterID UInt32, StartDate Date, UserID UInt32, VisitID UInt32, NestedColumn Nested(A UInt8, S String), ToDrop UInt32) "
                    "ENGINE = ReplicatedMergeTree ORDER BY CounterID;")

    expected = "CREATE TABLE testdb.concurrent_test\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n    `ToDrop` UInt32\\n)\\n" \
               "ENGINE = ReplicatedMergeTree(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')\\nORDER BY CounterID\\nSETTINGS index_granularity = 8192"

    assert_create_query([main_node, competing_node], "testdb.concurrent_test", expected)

    main_node.query("INSERT INTO testdb.dist (CounterID, StartDate, UserID) SELECT number, addDays(toDate('2020-02-02'), number), intHash32(number) FROM numbers(10)")

    # test_replica_restart
    main_node.restart_clickhouse()

    expected = "CREATE TABLE testdb.concurrent_test\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n    `ToDrop` UInt32\\n)\\n" \
               "ENGINE = ReplicatedMergeTree(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')\\nORDER BY CounterID\\nSETTINGS index_granularity = 8192"


    # test_snapshot_and_snapshot_recover
    snapshotting_node.query("CREATE DATABASE testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard2', 'replica1');")
    snapshot_recovering_node.query("CREATE DATABASE testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard2', 'replica2');")
    assert_create_query(all_nodes, "testdb.concurrent_test", expected)

    main_node.query("SYSTEM FLUSH DISTRIBUTED testdb.dist")
    main_node.query("ALTER TABLE testdb.concurrent_test UPDATE StartDate = addYears(StartDate, 1) WHERE 1")
    res = main_node.query("ALTER TABLE testdb.concurrent_test DELETE WHERE UserID % 2")
    assert "shard1|replica1" in res and "shard1|replica2" in res and "shard1|replica3" in res
    assert "shard2|replica1" in res and "shard2|replica2" in res

    expected = "1\t1\tmain_node\n" \
               "1\t2\tdummy_node\n" \
               "1\t3\tcompeting_node\n" \
               "2\t1\tsnapshotting_node\n" \
               "2\t2\tsnapshot_recovering_node\n"
    assert main_node.query("SELECT shard_num, replica_num, host_name FROM system.clusters WHERE cluster='testdb'") == expected

    # test_drop_and_create_replica
    main_node.query("DROP DATABASE testdb")
    main_node.query("CREATE DATABASE testdb ENGINE = Replicated('/clickhouse/databases/test1', 'shard1', 'replica1');")

    expected = "CREATE TABLE testdb.concurrent_test\\n(\\n    `CounterID` UInt32,\\n    `StartDate` Date,\\n    `UserID` UInt32,\\n" \
               "    `VisitID` UInt32,\\n    `NestedColumn.A` Array(UInt8),\\n    `NestedColumn.S` Array(String),\\n    `ToDrop` UInt32\\n)\\n" \
               "ENGINE = ReplicatedMergeTree(\\'/clickhouse/tables/uuid/{shard}\\', \\'{replica}\\')\\nORDER BY CounterID\\nSETTINGS index_granularity = 8192"

    assert_create_query([main_node, competing_node], "testdb.concurrent_test", expected)
    assert_create_query(all_nodes, "testdb.concurrent_test", expected)

    for node in all_nodes:
        node.query("SYSTEM SYNC REPLICA testdb.concurrent_test")

    expected = "0\t2021-02-02\t4249604106\n" \
               "1\t2021-02-03\t1343103100\n" \
               "4\t2021-02-06\t3902320246\n" \
               "7\t2021-02-09\t3844986530\n" \
               "9\t2021-02-11\t1241149650\n"

    assert_eq_with_retry(dummy_node, "SELECT CounterID, StartDate, UserID FROM testdb.dist ORDER BY CounterID", expected)

