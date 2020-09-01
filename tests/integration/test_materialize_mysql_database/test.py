import os
import subprocess
import time

import pymysql.cursors
import pytest

import materialize_with_ddl
from helpers.cluster import ClickHouseCluster, get_docker_compose_path

DOCKER_COMPOSE_PATH = get_docker_compose_path()

cluster = ClickHouseCluster(__file__)
clickhouse_node = cluster.add_instance('node1', config_dir="configs", with_mysql=False)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


class MySQLNodeInstance:
    def __init__(self, user='root', password='clickhouse', hostname='127.0.0.1', port=3308):
        self.user = user
        self.port = port
        self.hostname = hostname
        self.password = password
        self.mysql_connection = None  # lazy init

    def alloc_connection(self):
        if self.mysql_connection is None:
            self.mysql_connection = pymysql.connect(user=self.user, password=self.password, host=self.hostname, port=self.port, autocommit=True)
        return self.mysql_connection

    def query(self, execution_query):
        with self.alloc_connection().cursor() as cursor:
            cursor.execute(execution_query)

    def close(self):
        if self.mysql_connection is not None:
            self.mysql_connection.close()

    def wait_mysql_to_start(self, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.alloc_connection()
                print "Mysql Started"
                return
            except Exception as ex:
                print "Can't connect to MySQL " + str(ex)
                time.sleep(0.5)

        subprocess.check_call(['docker-compose', 'ps', '--services', 'all'])
        raise Exception("Cannot wait MySQL container")


@pytest.fixture(scope="module")
def started_mysql_5_7():
    mysql_node = MySQLNodeInstance('root', 'clickhouse', '127.0.0.1', 33307)
    docker_compose = os.path.join(DOCKER_COMPOSE_PATH, 'docker_compose_mysql_5_7.yml')

    try:
        subprocess.check_call(['docker-compose', '-p', cluster.project_name, '-f', docker_compose, 'up', '--no-recreate', '-d'])
        mysql_node.wait_mysql_to_start(120)
        yield mysql_node
    finally:
        mysql_node.close()
        subprocess.check_call(['docker-compose', '-p', cluster.project_name, '-f', docker_compose, 'down', '--volumes', '--remove-orphans'])


@pytest.fixture(scope="module")
def started_mysql_8_0():
    mysql_node = MySQLNodeInstance('root', 'clickhouse', '127.0.0.1', 33308)
    docker_compose = os.path.join(DOCKER_COMPOSE_PATH, 'docker_compose_mysql_8_0.yml')

    try:
        subprocess.check_call(['docker-compose', '-p', cluster.project_name, '-f', docker_compose, 'up', '--no-recreate', '-d'])
        mysql_node.wait_mysql_to_start(120)
        yield mysql_node
    finally:
        mysql_node.close()
        subprocess.check_call(['docker-compose', '-p', cluster.project_name, '-f', docker_compose, 'down', '--volumes', '--remove-orphans'])


def test_materialize_database_dml_with_mysql_5_7(started_cluster, started_mysql_5_7):
    materialize_with_ddl.dml_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")


def test_materialize_database_dml_with_mysql_8_0(started_cluster, started_mysql_8_0):
    materialize_with_ddl.dml_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")

def test_materialize_database_ddl_with_mysql_5_7(started_cluster, started_mysql_5_7):
    try:
        materialize_with_ddl.drop_table_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.create_table_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.rename_table_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.alter_add_column_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.alter_drop_column_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        # mysql 5.7 cannot support alter rename column
        # materialize_with_ddl.alter_rename_column_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.alter_rename_table_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
        materialize_with_ddl.alter_modify_column_with_materialize_mysql_database(clickhouse_node, started_mysql_5_7, "mysql5_7")
    except:
        print(clickhouse_node.query("select '\n', thread_id, query_id, arrayStringConcat(arrayMap(x -> concat(demangle(addressToSymbol(x)), '\n    ', addressToLine(x)), trace), '\n') AS sym from system.stack_trace format TSVRaw"))
        raise


def test_materialize_database_ddl_with_mysql_8_0(started_cluster, started_mysql_8_0):
    materialize_with_ddl.drop_table_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.create_table_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.rename_table_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.alter_add_column_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.alter_drop_column_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.alter_rename_table_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.alter_rename_column_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")
    materialize_with_ddl.alter_modify_column_with_materialize_mysql_database(clickhouse_node, started_mysql_8_0, "mysql8_0")

def test_materialize_database_ddl_with_empty_transaction(started_cluster, started_mysql_5_7):
    mysql_node = started_mysql_5_7.alloc_connection()
    mysql_node.query("CREATE DATABASE test_database")


    mysql_node.query("RESET MASTER")
    mysql_node.query("CREATE TABLE test_database.t1(a INT NOT NULL PRIMARY KEY, b VARCHAR(255) DEFAULT 'BEGIN')")
    mysql_node.query("INSERT INTO test_database.t1(a) VALUES(1)")

    clickhouse_node.query(
        "CREATE DATABASE test_database ENGINE = MaterializeMySQL('{}:3306', 'test_database', 'root', 'clickhouse')".format(
            "mysql5_7"))

    # Reject one empty GTID QUERY event with 'BEGIN' and 'COMMIT'
    mysql_cursor = mysql_node.cursor(pymysql.cursors.DictCursor)
    mysql_cursor.execute("SHOW MASTER STATUS")
    (uuid, seqs) = mysql_cursor.fetchall()[0]["Executed_Gtid_Set"].split(":")
    (seq_begin, seq_end) = seqs.split("-")
    assert int(seq_begin) == 1
    assert int(seq_end) == 3
    next_gtid = uuid + ":" + str(int(seq_end) + 1)
    mysql_node.query("SET gtid_next='" + next_gtid + "'")
    mysql_node.query("BEGIN")
    mysql_node.query("COMMIT")
    mysql_node.query("SET gtid_next='AUTOMATIC'")

    # Reject one 'BEGIN' QUERY event and 'COMMIT' XID event.
    mysql_node.query("/* start */ begin /* end */")
    mysql_node.query("INSERT INTO test_database.t1(a) VALUES(2)")
    mysql_node.query("/* start */ commit /* end */")

    materialize_with_ddl.check_query(clickhouse_node, "SELECT * FROM test_database.t1 ORDER BY a FORMAT TSV", "1\tBEGIN\n2\tBEGIN\n")
    clickhouse_node.query("DROP DATABASE test_database")
    mysql_node.query("DROP DATABASE test_database")
