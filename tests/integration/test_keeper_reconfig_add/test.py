#!/usr/bin/env python3

import pytest
from helpers.cluster import ClickHouseCluster
import helpers.keeper_utils as ku
import os
from kazoo.client import KazooClient
from kazoo.exceptions import BadArgumentsException

cluster = ClickHouseCluster(__file__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

node1 = cluster.add_instance("node1", main_configs=["configs/keeper1.xml"])
node2 = cluster.add_instance("node2", stay_alive=True)
node3 = cluster.add_instance("node3", stay_alive=True)

server_join_msg = "confirms it will join"
part_of_cluster = "now this node is the part of cluster"
zk1, zk2, zk3 = None, None, None


def get_fake_zk(node):
    return ku.get_fake_zk(cluster, node)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()

        node2.stop_clickhouse()
        node2.copy_file_to_container(
            os.path.join(CONFIG_DIR, "keeper2.xml"),
            "/etc/clickhouse-server/config.d/keeper.xml",
        )

        node3.stop_clickhouse()
        node3.copy_file_to_container(
            os.path.join(CONFIG_DIR, "keeper3.xml"),
            "/etc/clickhouse-server/config.d/keeper.xml",
        )

        yield cluster

    finally:
        for conn in [zk1, zk2, zk3]:
            if conn:
                conn.stop()
                conn.close()

        cluster.shutdown()


def test_reconfig_add(started_cluster):
    """
    Add a node to another node. Then add another node to two.
    """

    zk1 = get_fake_zk(node1)
    config = ku.get_config_str(zk1)
    print("Initial config", config)

    assert len(config.split("\n")) == 1
    assert "node1" in config
    assert "node2" not in config
    assert "node3" not in config

    with pytest.raises(BadArgumentsException):
        # duplicate id with different endpoint
        zk1.reconfig(joining="server.1=localhost:1337", leaving=None, new_members=None)

    with pytest.raises(BadArgumentsException):
        # duplicate endpoint
        zk1.reconfig(joining="server.8=node1:9234", leaving=None, new_members=None)

    for i in range(100):
        zk1.create(f"/test_three_{i}", b"somedata")

    node2.start_clickhouse()
    config, _ = zk1.reconfig(
        joining="server.2=node2:9234", leaving=None, new_members=None
    )
    ku.wait_until_connected(cluster, node2)

    config = config.decode("utf-8")
    print("After adding 2", config)

    assert len(config.split("\n")) == 2
    assert "node1" in config
    assert "node2" in config
    assert "node3" not in config

    zk2 = get_fake_zk(node2)
    ku.wait_configs_equal(config, zk2)

    for i in range(100):
        assert zk2.exists(f"/test_three_{i}") is not None
        zk2.create(f"/test_three_{100 + i}", b"somedata")

    # Why not both?
    # One node will process add_srv request, other will pull out updated config, apply
    # and return true in config update thread (without calling add_srv again)
    assert node1.contains_in_log(server_join_msg) or node2.contains_in_log(
        server_join_msg
    )

    assert node2.contains_in_log(part_of_cluster)

    zk1.stop()
    zk1.close()
    zk1 = get_fake_zk(node1)
    zk1.sync("/test_three_0")

    for i in range(200):
        assert zk1.exists(f"/test_three_{i}") is not None

    for i in range(100):
        zk2.create(f"/test_four_{i}", b"somedata")

    node3.start_clickhouse()
    config, _ = zk2.reconfig(
        joining="server.3=node3:9234", leaving=None, new_members=None
    )
    ku.wait_until_connected(cluster, node3)

    config = config.decode("utf-8")
    print("After adding 3", config)

    assert len(config.split("\n")) == 3
    assert "node1" in config
    assert "node2" in config
    assert "node3" in config

    zk3 = get_fake_zk(node3)
    ku.wait_configs_equal(config, zk3)

    for i in range(100):
        assert zk3.exists(f"/test_four_{i}") is not None
        zk3.create(f"/test_four_{100 + i}", b"somedata")

    zk1.stop()
    zk1.close()
    zk1 = get_fake_zk(node1)
    zk1.sync("/test_four_0")

    zk2.stop()
    zk2.close()
    zk2 = get_fake_zk(node2)
    zk2.sync("/test_four_0")

    for i in range(200):
        assert zk1.exists(f"/test_four_{i}") is not None
        assert zk2.exists(f"/test_four_{i}") is not None

    assert node3.contains_in_log(part_of_cluster)
