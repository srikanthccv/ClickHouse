#!/usr/bin/env python3
import os
import sys

CURDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(CURDIR, 'helpers'))

from client import client, prompt, end_of_block
from httpclient import client as http_client

log = None
# uncomment the line below for debugging
#log=sys.stdout

with client(name='client1>', log=log) as client1:
    client1.expect(prompt)

    client1.send('SET allow_experimental_live_view = 1')
    client1.expect(prompt)

    client1.send('DROP TABLE IF EXISTS test.lv')
    client1.expect(prompt)
    client1.send(' DROP TABLE IF EXISTS test.mt')
    client1.expect(prompt)
    client1.send('CREATE TABLE test.mt (a Int32) Engine=MergeTree order by tuple()')
    client1.expect(prompt)
    client1.send('CREATE LIVE VIEW test.lv AS SELECT sum(a) FROM test.mt')
    client1.expect(prompt)

    with http_client({'method':'GET', 'url': '/?allow_experimental_live_view=1&live_view_heartbeat_interval=1&query=WATCH%20test.lv%20EVENTS%20FORMAT%20JSONEachRowWithProgress'}, name='client2>', log=log) as client2:
        client2.expect('{"progress":{"read_rows":"1","read_bytes":"8","written_rows":"0","written_bytes":"0","total_rows_to_read":"0"}}\n', escape=True)
        client2.expect('{"row":{"version":"1"}', escape=True)
        client2.expect('{"progress":{"read_rows":"1","read_bytes":"8","written_rows":"0","written_bytes":"0","total_rows_to_read":"0"}}', escape=True)
        # heartbeat is provided by progress message
        client2.expect('{"progress":{"read_rows":"1","read_bytes":"8","written_rows":"0","written_bytes":"0","total_rows_to_read":"0"}}', escape=True)

        client1.send('INSERT INTO test.mt VALUES (1),(2),(3)')
        client1.expect(prompt)

        client2.expect('{"row":{"version":"2"}}\n', escape=True)

    client1.send('DROP TABLE test.lv')
    client1.expect(prompt)
    client1.send('DROP TABLE test.mt')
    client1.expect(prompt)
