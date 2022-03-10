-- Tags: no-parallel

SET prefer_localhost_replica = 1;

DROP DATABASE IF EXISTS test_01457;

CREATE DATABASE test_01457;

CREATE TABLE tmp (n Int8) ENGINE=Memory;

CREATE TABLE test_01457.tf_remote AS remote('localhost', currentDatabase(), 'tmp');
SHOW CREATE TABLE test_01457.tf_remote;
CREATE TABLE test_01457.tf_remote_explicit_structure (n UInt64) AS remote('localhost', currentDatabase(), 'tmp');
SHOW CREATE TABLE test_01457.tf_remote_explicit_structure;
CREATE TABLE test_01457.tf_numbers (number String) AS numbers(1);
SHOW CREATE TABLE test_01457.tf_numbers;
CREATE TABLE test_01457.tf_merge AS merge(currentDatabase(), 'tmp');
SHOW CREATE TABLE test_01457.tf_merge;

DROP TABLE tmp;

DETACH DATABASE test_01457;
ATTACH DATABASE test_01457;

-- To suppress "Structure does not match (...), implicit conversion will be done." message
SET send_logs_level='error';

CREATE TABLE tmp (n Int8) ENGINE=Memory;
INSERT INTO test_01457.tf_remote_explicit_structure VALUES ('42');
SELECT * FROM tmp;
TRUNCATE TABLE tmp;
INSERT INTO test_01457.tf_remote VALUES (0);

SELECT (*,).1 AS c, toTypeName(c) FROM tmp;
SELECT (*,).1 AS c, toTypeName(c) FROM test_01457.tf_remote;
SELECT (*,).1 AS c, toTypeName(c) FROM test_01457.tf_remote_explicit_structure;
SELECT (*,).1 AS c, toTypeName(c) FROM test_01457.tf_numbers;
SELECT (*,).1 AS c, toTypeName(c) FROM test_01457.tf_merge;

DROP DATABASE test_01457;

DROP TABLE tmp;
