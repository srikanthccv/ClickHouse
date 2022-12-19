import os.path as p
import random
import threading
import time
import pytest
import logging

from helpers.cluster import ClickHouseCluster
from helpers.test_tools import TSV
from helpers.client import QueryRuntimeException

import json
import subprocess

import socket

cluster = ClickHouseCluster(__file__)
instance1 = cluster.add_instance(
    "instance1",
    main_configs=["configs/kerberos_with_keytab.xml"],
    user_configs=["configs/users.xml"],
    with_kerberos_kdc=True,
    clickhouse_path_dir="clickhouse_path",
)
instance2 = cluster.add_instance(
    "instance2",
    main_configs=["configs/kerberos_without_keytab.xml"],
    user_configs=["configs/users.xml"],
    with_kerberos_kdc=True,
    clickhouse_path_dir="clickhouse_path",
)


# Fixtures


@pytest.fixture(scope="module")
def kerberos_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


# Tests


def make_auth(instance):
    instance.exec_in_container(
        ["bash", "-c", "kinit -k -t /tmp/keytab/kuser.keytab kuser"]
    )
    return instance.exec_in_container(
        [
            "bash",
            "-c",
            "echo 'select currentUser()' | curl -vvv --negotiate -u : http://{}:8123/ --data-binary @-".format(
                instance.hostname
            ),
        ]
    )


def test_kerberos_auth_with_keytab(kerberos_cluster):
    assert make_auth(instance1) == "kuser\n"


def test_kerberos_auth_without_keytab(kerberos_cluster):
    assert (
        "DB::Exception: : Authentication failed: password is incorrect or there is no user with such name."
        in make_auth(instance2)
    )


if __name__ == "__main__":
    cluster.start()
    input("Cluster created, press any key to destroy...")
    cluster.shutdown()
