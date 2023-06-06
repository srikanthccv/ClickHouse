#!/usr/bin/env python3

import gzip
import json
import logging
import os
import io
import random
import threading
import time

from azure.storage.blob import BlobServiceClient
import helpers.client
import pytest
from helpers.cluster import ClickHouseCluster, ClickHouseInstance
from helpers.network import PartitionManager
from helpers.mock_servers import start_mock_servers
from helpers.test_tools import exec_query_with_retry


@pytest.fixture(scope="module")
def cluster():
    try:
        cluster = ClickHouseCluster(__file__)
        cluster.add_instance(
            "node",
            main_configs=["configs/named_collections.xml"],
            with_azurite=True,
        )
        cluster.start()

        yield cluster
    finally:
        cluster.shutdown()


def azure_query(node, query, try_num=3, settings={}):
    for i in range(try_num):
        try:
            return node.query(query, settings=settings)
        except Exception as ex:
            retriable_errors = [
                "DB::Exception: Azure::Core::Http::TransportException: Connection was closed by the server while trying to read a response"
            ]
            retry = False
            for error in retriable_errors:
                if error in str(ex):
                    retry = True
                    print(f"Try num: {i}. Having retriable error: {ex}")
                    break
            if not retry or i == try_num - 1:
                raise Exception(ex)
            continue


def get_azure_file_content(filename):
    container_name = "cont"
    connection_string = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(filename)
    download_stream = blob_client.download_blob()
    return download_stream.readall().decode("utf-8")


def test_create_table_connection_string(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_create_table_conn_string (key UInt64, data String) Engine = Azure('DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite1:10000/devstoreaccount1/;', 'cont', 'test_create_connection_string', 'CSV')",
    )


def test_create_table_account_string(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_create_table_account_url (key UInt64, data String) Engine = Azure('http://azurite1:10000/devstoreaccount1',  'cont', 'test_create_connection_string', 'devstoreaccount1', 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==', 'CSV')",
    )


def test_simple_write_account_string(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_simple_write (key UInt64, data String) Engine = Azure('http://azurite1:10000/devstoreaccount1', 'cont', 'test_simple_write.csv', 'devstoreaccount1', 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==', 'CSV')",
    )
    azure_query(node, "INSERT INTO test_simple_write VALUES (1, 'a')")
    print(get_azure_file_content("test_simple_write.csv"))
    assert get_azure_file_content("test_simple_write.csv") == '1,"a"\n'


def test_simple_write_connection_string(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_simple_write_connection_string (key UInt64, data String) Engine = Azure('DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite1:10000/devstoreaccount1;', 'cont', 'test_simple_write_c.csv', 'CSV')",
    )
    azure_query(node, "INSERT INTO test_simple_write_connection_string VALUES (1, 'a')")
    print(get_azure_file_content("test_simple_write_c.csv"))
    assert get_azure_file_content("test_simple_write_c.csv") == '1,"a"\n'


def test_simple_write_named_collection_1(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_simple_write_named_collection_1 (key UInt64, data String) Engine = Azure(azure_conf1)",
    )
    azure_query(
        node, "INSERT INTO test_simple_write_named_collection_1 VALUES (1, 'a')"
    )
    print(get_azure_file_content("test_simple_write_named.csv"))
    assert get_azure_file_content("test_simple_write_named.csv") == '1,"a"\n'


def test_simple_write_named_collection_2(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_simple_write_named_collection_2 (key UInt64, data String) Engine = Azure(azure_conf2, container='cont', blob_path='test_simple_write_named_2.csv', format='CSV')",
    )
    azure_query(
        node, "INSERT INTO test_simple_write_named_collection_2 VALUES (1, 'a')"
    )
    print(get_azure_file_content("test_simple_write_named_2.csv"))
    assert get_azure_file_content("test_simple_write_named_2.csv") == '1,"a"\n'


def test_partition_by(cluster):
    node = cluster.instances["node"]
    table_format = "column1 UInt32, column2 UInt32, column3 UInt32"
    partition_by = "column3"
    values = "(1, 2, 3), (3, 2, 1), (78, 43, 45)"
    filename = "test_{_partition_id}.csv"

    azure_query(
        node,
        f"CREATE TABLE test_partitioned_write ({table_format}) Engine = Azure('http://azurite1:10000/devstoreaccount1', 'cont', '{filename}', 'devstoreaccount1', 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==', 'CSV') PARTITION BY {partition_by}",
    )
    azure_query(node, f"INSERT INTO test_partitioned_write VALUES {values}")

    assert "1,2,3\n" == get_azure_file_content("test_3.csv")
    assert "3,2,1\n" == get_azure_file_content("test_1.csv")
    assert "78,43,45\n" == get_azure_file_content("test_45.csv")


def test_partition_by_string_column(cluster):
    node = cluster.instances["node"]
    table_format = "col_num UInt32, col_str String"
    partition_by = "col_str"
    values = "(1, 'foo/bar'), (3, 'йцук'), (78, '你好')"
    filename = "test_{_partition_id}.csv"
    azure_query(
        node,
        f"CREATE TABLE test_partitioned_string_write ({table_format}) Engine = Azure('http://azurite1:10000/devstoreaccount1', 'cont', '{filename}', 'devstoreaccount1', 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==', 'CSV') PARTITION BY {partition_by}",
    )
    azure_query(node, f"INSERT INTO test_partitioned_string_write VALUES {values}")

    assert '1,"foo/bar"\n' == get_azure_file_content("test_foo/bar.csv")
    assert '3,"йцук"\n' == get_azure_file_content("test_йцук.csv")
    assert '78,"你好"\n' == get_azure_file_content("test_你好.csv")


def test_partition_by_const_column(cluster):
    node = cluster.instances["node"]
    table_format = "column1 UInt32, column2 UInt32, column3 UInt32"
    values = "(1, 2, 3), (3, 2, 1), (78, 43, 45)"
    partition_by = "'88'"
    values_csv = "1,2,3\n3,2,1\n78,43,45\n"
    filename = "test_{_partition_id}.csv"
    azure_query(
        node,
        f"CREATE TABLE test_partitioned_const_write ({table_format}) Engine = Azure('http://azurite1:10000/devstoreaccount1', 'cont', '{filename}', 'devstoreaccount1', 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==', 'CSV') PARTITION BY {partition_by}",
    )
    azure_query(node, f"INSERT INTO test_partitioned_const_write VALUES {values}")
    assert values_csv == get_azure_file_content("test_88.csv")


def test_truncate(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_truncate (key UInt64, data String) Engine = Azure(azure_conf2, container='cont', blob_path='test_truncate.csv', format='CSV')",
    )
    azure_query(node, "INSERT INTO test_truncate VALUES (1, 'a')")
    assert get_azure_file_content("test_truncate.csv") == '1,"a"\n'
    azure_query(node, "TRUNCATE TABLE test_truncate")
    with pytest.raises(Exception):
        print(get_azure_file_content("test_truncate.csv"))


def test_simple_read_write(cluster):
    node = cluster.instances["node"]
    azure_query(
        node,
        "CREATE TABLE test_simple_read_write (key UInt64, data String) Engine = Azure(azure_conf2, container='cont', blob_path='test_simple_read_write.csv', format='CSV')",
    )

    azure_query(node, "INSERT INTO test_simple_read_write VALUES (1, 'a')")
    assert get_azure_file_content("test_simple_read_write.csv") == '1,"a"\n'
    print(azure_query(node, "SELECT * FROM test_simple_read_write"))
    assert azure_query(node, "SELECT * FROM test_simple_read_write") == "1\ta\n"


def test_create_new_files_on_insert(cluster):

    node = cluster.instances["node"]

    azure_query(node, f"create table test_multiple_inserts(a Int32, b String) ENGINE = Azure(azure_conf2, container='cont', blob_path='test_parquet', format='Parquet')")
    azure_query(node, "truncate table test_multiple_inserts")
    azure_query(node,
        f"insert into test_multiple_inserts select number, randomString(100) from numbers(10) settings azure_truncate_on_insert=1"
    )
    azure_query(node,
        f"insert into test_multiple_inserts select number, randomString(100) from numbers(20) settings azure_create_new_file_on_insert=1"
    )
    azure_query(node,
        f"insert into test_multiple_inserts select number, randomString(100) from numbers(30) settings azure_create_new_file_on_insert=1"
    )

    result = azure_query(node, f"select count() from test_multiple_inserts")
    assert int(result) == 60

    azure_query(node, f"drop table test_multiple_inserts")
