CREATE TABLE t1(x UInt32, y UInt32) ENGINE TinyLog;
CREATE TABLE t2(x UInt32, y UInt32 DEFAULT x + 1) ENGINE TinyLog;
CREATE TABLE t3(x UInt32, y UInt32 MATERIALIZED x + 1) ENGINE TinyLog;
CREATE TABLE t4(x UInt32, y UInt32 ALIAS x + 1) ENGINE TinyLog;

INSERT INTO t1 VALUES (1, 1);
INSERT INTO t2 VALUES (1, 1);
INSERT INTO t3 VALUES (1);
INSERT INTO t4 VALUES (1);

INSERT INTO FUNCTION remote('127.0.0.2', default, t1) VALUES (2, 2);
INSERT INTO FUNCTION remote('127.0.0.2', default, t2) VALUES (2, 2);
--TODO: INSERT into remote tables with MATERIALIZED columns.
--INSERT INTO FUNCTION remote('127.0.0.2', t3) VALUES (2);
INSERT INTO FUNCTION remote('127.0.0.2', default, t4) VALUES (2);

SELECT * FROM remote('127.0.0.2', default, t1) ORDER BY x;

SELECT '*** With a DEFAULT column ***';
SELECT * FROM remote('127.0.0.2', default, t2) ORDER BY x;

SELECT '*** With a MATERIALIZED column ***';
SELECT * FROM remote('127.0.0.2', default, t3) ORDER BY x;
SELECT x, y FROM remote('127.0.0.2', default, t3) ORDER BY x;

SELECT '*** With an ALIAS column ***';
SELECT * FROM remote('127.0.0.2', default, t4) ORDER BY x;
SELECT x, y FROM remote('127.0.0.2', default, t4) ORDER BY x;
