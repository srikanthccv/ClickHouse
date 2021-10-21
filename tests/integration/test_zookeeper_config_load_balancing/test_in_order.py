import time
import pytest
import logging
from helpers.cluster import ClickHouseCluster

cluster = ClickHouseCluster(__file__, zookeeper_config_path='configs/zookeeper_config_in_order.xml')

node1 = cluster.add_instance('node1', with_zookeeper=True,
                                main_configs=["configs/remote_servers.xml", "configs/zookeeper_config_in_order.xml", "configs/zookeeper_log.xml"])
node2 = cluster.add_instance('node2', with_zookeeper=True,
                                main_configs=["configs/remote_servers.xml", "configs/zookeeper_config_in_order.xml", "configs/zookeeper_log.xml"])
node3 = cluster.add_instance('node3', with_zookeeper=True,
                                main_configs=["configs/remote_servers.xml", "configs/zookeeper_config_in_order.xml", "configs/zookeeper_log.xml"])


@pytest.fixture(scope="module", autouse=True)
def started_cluster():
    try:
        cluster.start()

        yield cluster

    finally:
        cluster.shutdown()

def wait_zookeeper_node_to_start(started_cluster, zk_nodes, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            for instance in zk_nodes:
                conn = started_cluster.get_kazoo_client(instance)
                conn.get_children('/')
            print("All instances of ZooKeeper started")
            return
        except Exception as ex:
            print(("Can't connect to ZooKeeper " + str(ex)))
            time.sleep(0.5)

def test_in_order(started_cluster):
    wait_zookeeper_node_to_start(started_cluster, ["zoo1", "zoo2", "zoo3"])
    time.sleep(2)
    zoo1_ip = started_cluster.get_instance_ip("zoo1")
    for i, node in enumerate([node1, node3]):
        node.query('DROP TABLE IF EXISTS simple SYNC')
        node.query('''
        CREATE TABLE simple (date Date, id UInt32)
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/0/simple', '{replica}', date, id, 8192);
        '''.format(replica=node.name))

    time.sleep(5)
    assert '::ffff:' + str(zoo1_ip) + '\n' == node1.query('SELECT IPv6NumToString(address) FROM system.zookeeper_log ORDER BY event_time DESC LIMIT 1')
    assert '::ffff:' + str(zoo1_ip) + '\n' == node2.query('SELECT IPv6NumToString(address) FROM system.zookeeper_log ORDER BY event_time DESC LIMIT 1')
    assert '::ffff:' + str(zoo1_ip) + '\n' == node3.query('SELECT IPv6NumToString(address) FROM system.zookeeper_log ORDER BY event_time DESC LIMIT 1')
