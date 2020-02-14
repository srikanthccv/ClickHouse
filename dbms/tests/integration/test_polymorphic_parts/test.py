import time
import pytest
import random
import string

from helpers.test_tools import TSV
from helpers.test_tools import assert_eq_with_retry
from helpers.cluster import ClickHouseCluster

cluster = ClickHouseCluster(__file__)

def get_random_array():
    return [random.randint(0, 1000) % 1000 for _ in range(random.randint(0, 1000))]

def get_random_string():
    length = random.randint(0, 1000)
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def insert_random_data(table, node, size):
    data = [
    '(' + ','.join((
        "'2019-10-11'",
        str(i),
        "'" + get_random_string() + "'",
        str(get_random_array()))) +
    ')' for i in range(size)
    ]
    
    node.query("INSERT INTO {} VALUES {}".format(table, ','.join(data)))

def create_tables(name, nodes, node_settings, shard):
    for i, (node, settings) in enumerate(zip(nodes, node_settings)):
        node.query(
        '''
        CREATE TABLE {name}(date Date, id UInt32, s String, arr Array(Int32))
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/test/{shard}/{name}', '{repl}')
        PARTITION BY toYYYYMM(date)
        ORDER BY id
        SETTINGS index_granularity = {index_granularity}, index_granularity_bytes = {index_granularity_bytes}, 
        min_rows_for_wide_part = {min_rows_for_wide_part}, min_bytes_for_wide_part = {min_bytes_for_wide_part}
        '''.format(name=name, shard=shard, repl=i, **settings)
        )


node1 = cluster.add_instance('node1', config_dir="configs", with_zookeeper=True)
node2 = cluster.add_instance('node2', config_dir="configs", with_zookeeper=True)

settings_default = {'index_granularity' : 64, 'index_granularity_bytes' : 10485760, 'min_rows_for_wide_part' : 512, 'min_bytes_for_wide_part' : 0}
settings_not_adaptive = {'index_granularity' : 64, 'index_granularity_bytes' : 0, 'min_rows_for_wide_part' : 512, 'min_bytes_for_wide_part' : 0}

node3 = cluster.add_instance('node3', config_dir="configs", with_zookeeper=True)
node4 = cluster.add_instance('node4', config_dir="configs", main_configs=['configs/no_leader.xml'], with_zookeeper=True)

settings_compact = {'index_granularity' : 64, 'index_granularity_bytes' : 10485760, 'min_rows_for_wide_part' : 512, 'min_bytes_for_wide_part' : 0}
settings_wide = {'index_granularity' : 64, 'index_granularity_bytes' : 10485760, 'min_rows_for_wide_part' : 0, 'min_bytes_for_wide_part' : 0}


@pytest.fixture(scope="module")
def start_cluster():
    try:
        cluster.start()

        create_tables('polymorphic_table', [node1, node2], [settings_default, settings_default], "shard1")
        create_tables('non_adaptive_table', [node1, node2], [settings_not_adaptive, settings_default], "shard1")
        create_tables('polymorphic_table_compact', [node3, node4], [settings_compact, settings_wide], "shard2")
        create_tables('polymorphic_table_wide', [node3, node4], [settings_wide, settings_compact], "shard2")

        yield cluster

    finally:
        cluster.shutdown()


def test_polymorphic_parts_basics(start_cluster):
    node1.query("SYSTEM STOP MERGES")
    node2.query("SYSTEM STOP MERGES")

    for size in [300, 300, 600]:
        insert_random_data('polymorphic_table', node1, size)
    node2.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)

    assert node1.query("SELECT count() FROM polymorphic_table") == "1200\n"
    assert node2.query("SELECT count() FROM polymorphic_table") == "1200\n"

    expected = "Compact\t2\nWide\t1\n"

    assert TSV(node1.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = 'polymorphic_table' AND active GROUP BY part_type ORDER BY part_type")) == TSV(expected)
    assert TSV(node2.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = 'polymorphic_table' AND active GROUP BY part_type ORDER BY part_type")) == TSV(expected)

    node1.query("SYSTEM START MERGES")
    node2.query("SYSTEM START MERGES")

    for _ in range(40):
        insert_random_data('polymorphic_table', node1, 10)
        insert_random_data('polymorphic_table', node2, 10)

    node1.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)
    node2.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)

    assert node1.query("SELECT count() FROM polymorphic_table") == "2000\n"
    assert node2.query("SELECT count() FROM polymorphic_table") == "2000\n"

    node1.query("OPTIMIZE TABLE polymorphic_table FINAL")
    node2.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)

    assert node1.query("SELECT count() FROM polymorphic_table") == "2000\n"
    assert node2.query("SELECT count() FROM polymorphic_table") == "2000\n"

    assert node1.query("SELECT DISTINCT part_type FROM system.parts WHERE table = 'polymorphic_table' AND active") == "Wide\n"
    assert node2.query("SELECT DISTINCT part_type FROM system.parts WHERE table = 'polymorphic_table' AND active") == "Wide\n"

    # Check alters and mutations also work
    node1.query("ALTER TABLE polymorphic_table ADD COLUMN ss String")
    node1.query("ALTER TABLE polymorphic_table UPDATE ss = toString(id) WHERE 1")

    node2.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)

    node1.query("SELECT count(ss) FROM polymorphic_table") == "2000\n"
    node1.query("SELECT uniqExact(ss) FROM polymorphic_table") == "600\n"

    node2.query("SELECT count(ss) FROM polymorphic_table") == "2000\n"
    node2.query("SELECT uniqExact(ss) FROM polymorphic_table") == "600\n"


# Check that follower replicas create parts of the same type, which leader has chosen at merge.
@pytest.mark.parametrize(
    ('table', 'part_type'),
    [
        ('polymorphic_table_compact', 'Compact'),
        ('polymorphic_table_wide', 'Wide')
    ]
)
def test_different_part_types_on_replicas(start_cluster, table, part_type):
    leader = node3
    follower = node4

    assert leader.query("SELECT is_leader FROM system.replicas WHERE table = '{}'".format(table)) == "1\n"
    assert node4.query("SELECT is_leader FROM system.replicas WHERE table = '{}'".format(table)) == "0\n"

    for _ in range(3):
        insert_random_data(table, leader, 100)

    leader.query("OPTIMIZE TABLE {} FINAL".format(table))
    follower.query("SYSTEM SYNC REPLICA {}".format(table), timeout=20)

    expected = "{}\t1\n".format(part_type)

    assert TSV(leader.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = '{}' AND active GROUP BY part_type ORDER BY part_type".format(table))) == TSV(expected)
    assert TSV(follower.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = '{}' AND active GROUP BY part_type ORDER BY part_type".format(table))) == TSV(expected)


node7 = cluster.add_instance('node7', config_dir="configs", with_zookeeper=True, image='yandex/clickhouse-server:19.17.8.54', stay_alive=True, with_installed_binary=True)
node8 = cluster.add_instance('node8', config_dir="configs", with_zookeeper=True)

settings7 = {'index_granularity' : 64, 'index_granularity_bytes' : 10485760}
settings8 = {'index_granularity' : 64, 'index_granularity_bytes' : 10485760, 'min_rows_for_wide_part' : 512, 'min_bytes_for_wide_part' : 0}

@pytest.fixture(scope="module")
def start_cluster_diff_versions():
    try:
        for name in ['polymorphic_table', 'polymorphic_table_2']:
            cluster.start()
            node7.query(
            '''
            CREATE TABLE {name}(date Date, id UInt32, s String, arr Array(Int32))
            ENGINE = ReplicatedMergeTree('/clickhouse/tables/test/shard4/{name}', '1')
            PARTITION BY toYYYYMM(date)
            ORDER BY id
            SETTINGS index_granularity = {index_granularity}, index_granularity_bytes = {index_granularity_bytes}
            '''.format(name=name, **settings7)
            )

            node8.query(
            '''
            CREATE TABLE {name}(date Date, id UInt32, s String, arr Array(Int32))
            ENGINE = ReplicatedMergeTree('/clickhouse/tables/test/shard4/{name}', '2')
            PARTITION BY toYYYYMM(date)
            ORDER BY id
            SETTINGS index_granularity = {index_granularity}, index_granularity_bytes = {index_granularity_bytes},
            min_rows_for_wide_part = {min_rows_for_wide_part}, min_bytes_for_wide_part = {min_bytes_for_wide_part}
            '''.format(name=name, **settings8)
            )

        yield cluster

    finally:
        cluster.shutdown()


def test_polymorphic_parts_diff_versions(start_cluster_diff_versions):
    # Check that replication with Wide parts works between different versions.

    node_old = node7
    node_new = node8

    insert_random_data('polymorphic_table', node7, 100)
    node8.query("SYSTEM SYNC REPLICA polymorphic_table", timeout=20)

    assert node8.query("SELECT count() FROM polymorphic_table") == "100\n"
    assert node8.query("SELECT DISTINCT part_type FROM system.parts WHERE table = 'polymorphic_table' and active") == "Wide\n"


def test_polymorphic_parts_diff_versions_2(start_cluster_diff_versions):
    # Replication doesn't work on old version if part is created in compact format, because 
    #  this version doesn't know anything about it. It's considered to be ok.

    node_old = node7
    node_new = node8

    insert_random_data('polymorphic_table_2', node_new, 100)

    assert node_new.query("SELECT count() FROM polymorphic_table_2") == "100\n"
    assert node_old.query("SELECT count() FROM polymorphic_table_2") == "0\n"
    assert node_old.contains_in_log("<Error> default.polymorphic_table_2")

    node_old.restart_with_latest_version()

    node_old.query("SYSTEM SYNC REPLICA polymorphic_table_2", timeout=20)

    # Works after update
    assert node_old.query("SELECT count() FROM polymorphic_table_2") == "100\n"
    assert node_old.query("SELECT DISTINCT part_type FROM system.parts WHERE table = 'polymorphic_table_2' and active") == "Compact\n"


def test_polymorphic_parts_non_adaptive(start_cluster):
    node1.query("SYSTEM STOP MERGES")
    node2.query("SYSTEM STOP MERGES")

    insert_random_data('non_adaptive_table', node1, 100)
    node2.query("SYSTEM SYNC REPLICA non_adaptive_table", timeout=20)

    insert_random_data('non_adaptive_table', node2, 100)
    node1.query("SYSTEM SYNC REPLICA non_adaptive_table", timeout=20)

    assert TSV(node1.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = 'non_adaptive_table' AND active GROUP BY part_type ORDER BY part_type")) == TSV("Wide\t2\n")
    assert TSV(node2.query("SELECT part_type, count() FROM system.parts " \
        "WHERE table = 'non_adaptive_table' AND active GROUP BY part_type ORDER BY part_type")) == TSV("Wide\t2\n")

    assert node1.contains_in_log("<Warning> default.non_adaptive_table: Table can't create parts with adaptive granularity")
