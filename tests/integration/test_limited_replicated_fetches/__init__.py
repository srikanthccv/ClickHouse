#!/usr/bin/env python3

import pytest
import time
from helpers.cluster import ClickHouseCluster
from helpers.network import PartitionManager
import random
import string

cluster = ClickHouseCluster(__file__)
node1 = cluster.add_instance('node1', with_zookeeper=True)
node2 = cluster.add_instance('node2', with_zookeeper=True)

DEFAULT_MAX_THREADS_FOR_FETCH = 3

@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()

        yield cluster

    finally:
        cluster.shutdown()


def get_random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def test_limited_fetches(started_cluster):
    node1.query("CREATE TABLE t (key UInt64, data String) ENGINE = ReplicatedMergeTree('/clickhouse/test/t', '1') ORDER BY tuple() PARTITION BY key")
    node2.query("CREATE TABLE t (key UInt64, data String) ENGINE = ReplicatedMergeTree('/clickhouse/test/t', '2') ORDER BY tuple() PARTITION BY key")

    with PartitionManager() as pm:
        node2.query("SYSTEM STOP FETCHES t")
        node1.query("INSERT INTO t SELECT 1, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        node1.query("INSERT INTO t SELECT 2, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        node1.query("INSERT INTO t SELECT 3, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        node1.query("INSERT INTO t SELECT 4, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        node1.query("INSERT INTO t SELECT 5, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        node1.query("INSERT INTO t SELECT 6, '{}' FROM numbers(5000)".format(get_random_string(104857)))
        pm.add_network_delay(node1, 80)
        node2.query("SYSTEM START FETCHES t")
        fetches_result = []
        for _ in range(1000):
            result = node2.query("SELECT result_part_name FROM system.replicated_fetches")
            if not result:
                if fetches_result:
                    break
                time.sleep(0.1)
            else:
                fetches_result.append(result.split('\n'))
                print(fetches_result[-1])
                time.sleep(0.1)

    for concurrently_fetching_parts in fetches_result:
        if len(concurrently_fetching_parts) > DEFAULT_MAX_THREADS_FOR_FETCH:
            assert False, "Found more than {} concurrently fetching parts: {}".format(DEFAULT_MAX_THREADS_FOR_FETCH, ', '.join(concurrently_fetching_parts))
