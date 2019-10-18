import pytest
import os
from helpers.cluster import ClickHouseCluster
from helpers.test_tools import TSV, assert_eq_with_retry
from helpers.client import QueryRuntimeException
import pymysql
import warnings
import time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

cluster = ClickHouseCluster(__file__, base_configs_dir=os.path.join(SCRIPT_DIR, 'configs'))
node1 = cluster.add_instance('node1', with_mysql=True, main_configs=['configs/dictionaries/simple_dictionary.xml'])
node2 = cluster.add_instance('node2', with_mysql=True, main_configs=['configs/dictionaries/simple_dictionary.xml', 'configs/dictionaries/lazy_load.xml'])


def create_mysql_conn(user, password, hostname, port):
    return pymysql.connect(
        user=user,
        password=password,
        host=hostname,
        port=port)

def execute_mysql_query(connection, query):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        for clickhouse in [node1, node2]:
            clickhouse.query("CREATE DATABASE test", user="admin")
            clickhouse.query("CREATE TABLE test.xml_dictionary_table (id UInt64, SomeValue1 UInt8, SomeValue2 String) ENGINE = MergeTree() ORDER BY id", user="admin")
            clickhouse.query("INSERT INTO test.xml_dictionary_table SELECT number, number % 23, hex(number) from numbers(1000)", user="admin")
        yield cluster

    finally:
        cluster.shutdown()



@pytest.mark.parametrize("clickhouse,name,layout", [
    (node1, 'complex_node1_hashed', 'LAYOUT(COMPLEX_KEY_HASHED())'),
    (node1, 'complex_node1_cache', 'LAYOUT(COMPLEX_KEY_CACHE(SIZE_IN_CELLS 10))'),
    (node2, 'complex_node2_hashed', 'LAYOUT(COMPLEX_KEY_HASHED())'),
    (node2, 'complex_node2_cache', 'LAYOUT(COMPLEX_KEY_CACHE(SIZE_IN_CELLS 10))'),
])
def test_crete_and_select_mysql(started_cluster, clickhouse, name, layout):
    mysql_conn = create_mysql_conn("root", "clickhouse", "localhost", 3308)
    execute_mysql_query(mysql_conn, "CREATE DATABASE IF NOT EXISTS clickhouse")
    execute_mysql_query(mysql_conn, "CREATE TABLE clickhouse.{} (key_field1 int, key_field2 bigint, value1 text, value2 float, PRIMARY KEY (key_field1, key_field2))".format(name))
    values = []
    for i in range(1000):
        values.append('(' + ','.join([str(i), str(i * i), str(i) * 5, str(i * 3.14)]) + ')')
    execute_mysql_query(mysql_conn, "INSERT INTO clickhouse.{} VALUES ".format(name) + ','.join(values))

    clickhouse.query("""
    CREATE DICTIONARY default.{} (
        key_field1 Int32,
        key_field2 Int64,
        value1 String DEFAULT 'xxx',
        value2 Float32 DEFAULT 'yyy'
    )
    PRIMARY KEY key_field1, key_field2
    SOURCE(MYSQL(
        USER 'root'
        PASSWORD 'clickhouse'
        DB 'clickhouse'
        TABLE '{}'
        REPLICA(PRIORITY 1 HOST '127.0.0.1' PORT 3333)
        REPLICA(PRIORITY 2 HOST 'mysql1' PORT 3306)
    ))
    {}
    LIFETIME(MIN 1 MAX 3)
    """.format(name, name, layout))

    for i in range(172, 200):
        assert clickhouse.query("SELECT dictGetString('default.{}', 'value1', tuple(toInt32({}), toInt64({})))".format(name, i, i * i)) == str(i) * 5 + '\n'
        stroka = clickhouse.query("SELECT dictGetFloat32('default.{}', 'value2', tuple(toInt32({}), toInt64({})))".format(name, i, i * i)).strip()
        value = float(stroka)
        assert int(value) == int(i * 3.14)


    for i in range(1000):
        values.append('(' + ','.join([str(i), str(i * i), str(i) * 3, str(i * 2.718)]) + ')')
    execute_mysql_query(mysql_conn, "REPLACE INTO clickhouse.{} VALUES ".format(name) + ','.join(values))

    clickhouse.query("SYSTEM RELOAD DICTIONARY 'default.{}'".format(name))

    for i in range(172, 200):
        assert clickhouse.query("SELECT dictGetString('default.{}', 'value1', tuple(toInt32({}), toInt64({})))".format(name, i, i * i)) == str(i) * 3 + '\n'
        stroka = clickhouse.query("SELECT dictGetFloat32('default.{}', 'value2', tuple(toInt32({}), toInt64({})))".format(name, i, i * i)).strip()
        value = float(stroka)
        assert int(value) == int(i * 2.718)

    clickhouse.query("select dictGetUInt8('xml_dictionary', 'SomeValue1', toUInt64(17))") == "17\n"
    clickhouse.query("select dictGetString('xml_dictionary', 'SomeValue2', toUInt64(977))") == str(hex(977))[2:] + '\n'


def test_restricted_database(started_cluster):
    for node in [node1, node2]:
        node.query("CREATE DATABASE IF NOT EXISTS restricted_db", user="admin")
        node.query("CREATE TABLE restricted_db.table_in_restricted_db AS test.xml_dictionary_table", user="admin")

    with pytest.raises(QueryRuntimeException):
        node1.query("""
        CREATE DICTIONARY restricted_db.some_dict(
            id UInt64,
            SomeValue1 UInt8,
            SomeValue2 String
        )
        PRIMARY KEY id
        LAYOUT(FLAT())
        SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 USER 'default' TABLE 'table_in_restricted_db' DB 'restricted_db'))
        LIFETIME(MIN 1 MAX 10)
        """)

    with pytest.raises(QueryRuntimeException):
        node1.query("""
        CREATE DICTIONARY default.some_dict(
            id UInt64,
            SomeValue1 UInt8,
            SomeValue2 String
        )
        PRIMARY KEY id
        LAYOUT(FLAT())
        SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 USER 'default' TABLE 'table_in_restricted_db' DB 'restricted_db'))
        LIFETIME(MIN 1 MAX 10)
        """)

        node1.query("SELECT dictGetUInt8('default.some_dict', 'SomeValue1', toUInt64(17))") == "17\n"

    # with lazy load we don't need query to get exception
    with pytest.raises(QueryRuntimeException):
        node2.query("""
        CREATE DICTIONARY restricted_db.some_dict(
            id UInt64,
            SomeValue1 UInt8,
            SomeValue2 String
        )
        PRIMARY KEY id
        LAYOUT(FLAT())
        SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 USER 'default' TABLE 'table_in_restricted_db' DB 'restricted_db'))
        LIFETIME(MIN 1 MAX 10)
        """)

    with pytest.raises(QueryRuntimeException):
        node2.query("""
        CREATE DICTIONARY default.some_dict(
            id UInt64,
            SomeValue1 UInt8,
            SomeValue2 String
        )
        PRIMARY KEY id
        LAYOUT(FLAT())
        SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 USER 'default' TABLE 'table_in_restricted_db' DB 'restricted_db'))
        LIFETIME(MIN 1 MAX 10)
        """)
