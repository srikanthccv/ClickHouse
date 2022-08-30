import multiprocessing
import pytest
from time import sleep
import random
from itertools import count
from sys import stdout

from multiprocessing import Pool

from helpers.cluster import ClickHouseCluster
from helpers.test_tools import assert_eq_with_retry, assert_logs_contain
from helpers.network import PartitionManager

test_recover_staled_replica_run = 1

cluster = ClickHouseCluster(__file__)

node = cluster.add_instance(
    "node",
    main_configs=[],
    with_zookeeper=True,
    stay_alive=True,
)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster

    finally:
        cluster.shutdown()


def get_genuine_zk():
    print("Zoo1", cluster.get_instance_ip("zoo1"))
    return cluster.get_kazoo_client("zoo1")


def remove_children(client, path):
    children = client.get_children(path)

    for child in children:
        child_path = f"{path}/{child}"
        remove_children(client, child_path)
        client.delete(child_path)


def test_create_keeper_map(started_cluster):
    assert "Path '/test1' doesn't exist" in node.query_and_get_error(
        "CREATE TABLE test_keeper_map (key UInt64, value UInt64) ENGINE = KeeperMap('/test1', 0) PRIMARY KEY(key);"
    )

    node.query(
        "CREATE TABLE test_keeper_map (key UInt64, value UInt64) ENGINE = KeeperMap('/test1') PRIMARY KEY(key);"
    )
    zk_client = get_genuine_zk()

    def assert_children_size(expected_size):
        assert len(zk_client.get_children("/test1")) == expected_size

    assert_children_size(1)

    node.query("INSERT INTO test_keeper_map VALUES (1, 11)")
    assert_children_size(2)

    node.query(
        "CREATE TABLE test_keeper_map_another (key UInt64, value UInt64) ENGINE = KeeperMap('/test1') PRIMARY KEY(key);"
    )
    assert_children_size(2)
    node.query("INSERT INTO test_keeper_map_another VALUES (1, 11)")
    assert_children_size(2)

    node.query("INSERT INTO test_keeper_map_another VALUES (2, 22)")
    assert_children_size(3)

    node.query("DROP TABLE test_keeper_map SYNC")
    assert_children_size(3)

    node.query("DROP TABLE test_keeper_map_another SYNC")
    assert_children_size(0)

    zk_client.stop()


def create_drop_loop(index, stop_event):
    table_name = f"test_keeper_map_{index}"

    for i in count(0, 1):
        if stop_event.is_set():
            return

        stdout.write(f"Trying with {i} for {index}\n")
        node.query(
            f"CREATE TABLE {table_name} (key UInt64, value UInt64) ENGINE = KeeperMap('/test') PRIMARY KEY(key);"
        )
        node.query(f"INSERT INTO {table_name} VALUES ({index}, {i})")
        result = node.query(f"SELECT value FROM {table_name} WHERE key = {index}")
        assert result.strip() == str(i)
        node.query(f"DROP TABLE {table_name} SYNC")


def test_create_drop_keeper_map_concurrent(started_cluster):
    pool = Pool()
    manager = multiprocessing.Manager()
    stop_event = manager.Event()
    results = []
    for i in range(multiprocessing.cpu_count()):
        sleep(0.2)
        results.append(
            pool.apply_async(
                create_drop_loop,
                args=(
                    i,
                    stop_event,
                ),
            )
        )

    sleep(60)
    stop_event.set()

    for result in results:
        result.get()

    pool.close()

    client = get_genuine_zk()
    assert len(client.get_children("/test")) == 0


def test_keeper_map_without_zk(started_cluster):
    def assert_keeper_exception_after_partition(query):
        with PartitionManager() as pm:
            pm.drop_instance_zk_connections(node)
            error = node.query_and_get_error(query)
            assert "Coordination::Exception" in error

    assert_keeper_exception_after_partition(
        "CREATE TABLE test_keeper_map (key UInt64, value UInt64) ENGINE = KeeperMap('/test1') PRIMARY KEY(key);"
    )

    node.query(
        "CREATE TABLE test_keeper_map (key UInt64, value UInt64) ENGINE = KeeperMap('/test1') PRIMARY KEY(key);"
    )

    assert_keeper_exception_after_partition(
        "INSERT INTO test_keeper_map VALUES (1, 11)"
    )
    node.query("INSERT INTO test_keeper_map VALUES (1, 11)")

    assert_keeper_exception_after_partition("SELECT * FROM test_keeper_map")
    node.query("SELECT * FROM test_keeper_map")

    with PartitionManager() as pm:
        pm.drop_instance_zk_connections(node)
        node.restart_clickhouse(60)
        error = node.query_and_get_error("SELECT * FROM test_keeper_map")
        assert "Failed to activate table because of connection issues" in error

    node.query("SELECT * FROM test_keeper_map")

    client = get_genuine_zk()
    remove_children(client, "/test1")
    node.restart_clickhouse(60)
    error = node.query_and_get_error("SELECT * FROM test_keeper_map")
    assert "Failed to activate table because of invalid metadata in ZooKeeper" in error

    node.query("DETACH TABLE test_keeper_map")

    client.stop()
