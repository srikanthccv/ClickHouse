#!/usr/bin/env python3
# Tags: no-parallel

import os
import sys

CURDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(CURDIR, "helpers"))

from client import client, prompt, end_of_block

log = None
# uncomment the line below for debugging
# log=sys.stdout

with client(name="client1>", log=log) as client1, client(
    name="client2>", log=log
) as client2:
    client1.expect(prompt)
    client2.expect(prompt)

    client1.send("SET allow_experimental_window_view = 1")
    client1.expect(prompt)
    client1.send("SET window_view_heartbeat_interval = 1")
    client1.expect(prompt)
    client2.send("SET allow_experimental_window_view = 1")
    client2.expect(prompt)

    client1.send("CREATE DATABASE IF NOT EXISTS 01078_window_view_alter_query_watch")
    client1.expect(prompt)
    client1.send("DROP TABLE IF EXISTS 01078_window_view_alter_query_watch.mt NO DELAY")
    client1.expect(prompt)
    client1.send("DROP TABLE IF EXISTS 01078_window_view_alter_query_watch.wv NO DELAY")
    client1.expect(prompt)

    client1.send(
        "CREATE TABLE 01078_window_view_alter_query_watch.mt(a Int32, timestamp DateTime('US/Samoa')) ENGINE=MergeTree ORDER BY tuple()"
    )
    client1.expect(prompt)
    client1.send(
        "CREATE WINDOW VIEW 01078_window_view_alter_query_watch.wv WATERMARK=ASCENDING AS SELECT count(a) AS count, tumbleEnd(wid) AS w_end FROM 01078_window_view_alter_query_watch.mt GROUP BY tumble(timestamp, INTERVAL '5' SECOND, 'US/Samoa') AS wid"
    )
    client1.expect(prompt)

    client1.send("WATCH 01078_window_view_alter_query_watch.wv")
    client1.expect("Query id" + end_of_block)
    client2.send(
        "INSERT INTO 01078_window_view_alter_query_watch.mt VALUES (1, '1990/01/01 12:00:00');"
    )
    client2.expect("Ok.")
    client2.send(
        "INSERT INTO 01078_window_view_alter_query_watch.mt VALUES (1, '1990/01/01 12:00:06');"
    )
    client2.expect("Ok.")
    client1.expect("1*" + end_of_block)
    client1.expect("Progress: 1.00 rows.*\)")
    client2.send(
        "ALTER TABLE 01078_window_view_alter_query_watch.wv MODIFY QUERY SELECT count(a) * 2 AS count, tumbleEnd(wid) AS w_end FROM 01078_window_view_alter_query_watch.mt GROUP BY tumble(timestamp, INTERVAL '5' SECOND, 'US/Samoa') AS wid"
    )
    client2.expect("Ok.")
    client1.expect(prompt)
    client1.send("WATCH 01078_window_view_alter_query_watch.wv")
    client1.expect("Query id" + end_of_block)
    client2.send(
        "INSERT INTO 01078_window_view_alter_query_watch.mt VALUES (1, '1990/01/01 12:00:11');"
    )
    client2.expect("Ok.")
    client2.send(
        "INSERT INTO 01078_window_view_alter_query_watch.mt VALUES (1, '1990/01/01 12:00:18');"
    )
    client2.expect("Ok.")
    client1.expect("2*" + end_of_block)
    client1.expect("Progress: 1.00 rows.*\)")

    # send Ctrl-C
    client1.send("\x03", eol="")
    match = client1.expect("(%s)|([#\$] )" % prompt)
    if match.groups()[1]:
        client1.send(client1.command)
        client1.expect(prompt)
    client1.send("DROP TABLE 01078_window_view_alter_query_watch.wv NO DELAY;")
    client1.expect(prompt)
    client1.send("DROP TABLE 01078_window_view_alter_query_watch.mt;")
    client1.expect(prompt)
    client1.send("DROP DATABASE IF EXISTS 01078_window_view_alter_query_watch;")
    client1.expect(prompt)
