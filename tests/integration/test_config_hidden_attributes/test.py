import pytest
import os
from helpers.cluster import ClickHouseCluster


cluster = ClickHouseCluster(__file__)
node = cluster.add_instance(
    "node", main_configs=["configs/config.xml"], user_configs=["configs/users.xml"]
)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster

    finally:
        cluster.shutdown()


def test_hidden(started_cluster):
    assert (
        node.query(
            "select value from system.server_settings where name ='max_table_size_to_drop'"
        )
        == "60000000000\n"
    )
    assert (
        node.query(
            "select value from system.server_settings where name ='max_partition_size_to_drop'"
        )
        == "40000000000\n"
    )
    assert "key_1" in node.query("select collection from system.named_collections")
    out = node.exec_in_container(
        ["cat", "/var/lib/clickhouse/preprocessed_configs/config.xml"]
    )
    assert "max_table_size_to_drop" not in out
    assert "max_partition_size_to_drop" in out
    assert "named_collections" not in out
