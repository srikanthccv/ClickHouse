import pytest


from helpers.cluster import ClickHouseCluster
cluster = ClickHouseCluster(__file__)


node1 = cluster.add_instance('node1', with_zookeeper=True, image='yandex/clickhouse-server:19.4.5.35', stay_alive=True, with_installed_binary=True)
node2 = cluster.add_instance('node2', with_zookeeper=True, image='yandex/clickhouse-server:19.4.5.35', stay_alive=True, with_installed_binary=True)
node3 = cluster.add_instance('node3', with_zookeeper=True, image='yandex/clickhouse-server:19.4.5.35', stay_alive=True, with_installed_binary=True)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


def test_backup_from_old_version(started_cluster):
    node1.query("CREATE TABLE source_table(A Int64, B String) Engine = MergeTree order by tuple()")

    node1.query("INSERT INTO source_table VALUES(1, '1')")

    assert node1.query("SELECT COUNT() FROM source_table") == "1\n"

    node1.query("ALTER TABLE source_table ADD COLUMN Y String")

    node1.query("ALTER TABLE source_table FREEZE PARTITION tuple();")

    node1.restart_with_latest_version()

    node1.query("CREATE TABLE dest_table (A Int64,  B String,  Y String) ENGINE = ReplicatedMergeTree('/test/dest_table1', '1')  ORDER BY tuple()")

    node1.query("INSERT INTO dest_table VALUES(2, '2', 'Hello')")

    assert node1.query("SELECT COUNT() FROM dest_table") == "1\n"

    node1.exec_in_container(['bash', '-c', 'cp -r /var/lib/clickhouse/shadow/1/data/default/source_table/all_1_1_0/ /var/lib/clickhouse/data/default/dest_table/detached'])

    assert node1.query("SELECT COUNT() FROM dest_table") == "1\n"

    node1.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node1.query("SELECT sum(A) FROM dest_table") == "3\n"

    node1.query("ALTER TABLE dest_table DETACH PARTITION tuple()")

    node1.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node1.query("SELECT sum(A) FROM dest_table") == "3\n"

    assert node1.query("CHECK TABLE dest_table") == "1\n"


def test_backup_from_old_version_setting(started_cluster):
    node2.query("CREATE TABLE source_table(A Int64, B String) Engine = MergeTree order by tuple()")

    node2.query("INSERT INTO source_table VALUES(1, '1')")

    assert node2.query("SELECT COUNT() FROM source_table") == "1\n"

    node2.query("ALTER TABLE source_table ADD COLUMN Y String")

    node2.query("ALTER TABLE source_table FREEZE PARTITION tuple();")

    node2.restart_with_latest_version()

    node2.query("CREATE TABLE dest_table (A Int64,  B String,  Y String) ENGINE = ReplicatedMergeTree('/test/dest_table2', '1')  ORDER BY tuple() SETTINGS enable_mixed_granularity_parts = 1")

    node2.query("INSERT INTO dest_table VALUES(2, '2', 'Hello')")

    assert node2.query("SELECT COUNT() FROM dest_table") == "1\n"

    node2.exec_in_container(['bash', '-c', 'cp -r /var/lib/clickhouse/shadow/1/data/default/source_table/all_1_1_0/ /var/lib/clickhouse/data/default/dest_table/detached'])

    assert node2.query("SELECT COUNT() FROM dest_table") == "1\n"

    node2.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node2.query("SELECT sum(A) FROM dest_table") == "3\n"

    node2.query("ALTER TABLE dest_table DETACH PARTITION tuple()")

    node2.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node2.query("SELECT sum(A) FROM dest_table") == "3\n"

    assert node1.query("CHECK TABLE dest_table") == "1\n"


def test_backup_from_old_version_config(started_cluster):
    node3.query("CREATE TABLE source_table(A Int64, B String) Engine = MergeTree order by tuple()")

    node3.query("INSERT INTO source_table VALUES(1, '1')")

    assert node3.query("SELECT COUNT() FROM source_table") == "1\n"

    node3.query("ALTER TABLE source_table ADD COLUMN Y String")

    node3.query("ALTER TABLE source_table FREEZE PARTITION tuple();")

    def callback(n):
        n.replace_config("/etc/clickhouse-server/merge_tree_settings.xml", "<yandex><merge_tree><enable_mixed_granularity_parts>1</enable_mixed_granularity_parts></merge_tree></yandex>")

    node3.restart_with_latest_version(callback_onstop=callback)

    node3.query("CREATE TABLE dest_table (A Int64,  B String,  Y String) ENGINE = ReplicatedMergeTree('/test/dest_table3', '1')  ORDER BY tuple() SETTINGS enable_mixed_granularity_parts = 1")

    node3.query("INSERT INTO dest_table VALUES(2, '2', 'Hello')")

    assert node3.query("SELECT COUNT() FROM dest_table") == "1\n"

    node3.exec_in_container(['bash', '-c', 'cp -r /var/lib/clickhouse/shadow/1/data/default/source_table/all_1_1_0/ /var/lib/clickhouse/data/default/dest_table/detached'])

    assert node3.query("SELECT COUNT() FROM dest_table") == "1\n"

    node3.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node3.query("SELECT sum(A) FROM dest_table") == "3\n"

    node3.query("ALTER TABLE dest_table DETACH PARTITION tuple()")

    node3.query("ALTER TABLE dest_table ATTACH PARTITION tuple()")

    assert node3.query("SELECT sum(A) FROM dest_table") == "3\n"

    assert node1.query("CHECK TABLE dest_table") == "1\n"
