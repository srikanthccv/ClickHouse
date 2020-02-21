DROP TABLE IF EXISTS log;

CREATE TABLE log (x UInt8) ENGINE = StripeLog () SETTINGS disk = 'disk_memory';

SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (0);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (1);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (2);
SELECT * FROM log ORDER BY x;

TRUNCATE TABLE log;
DROP TABLE log;

CREATE TABLE log (x UInt8) ENGINE = TinyLog () SETTINGS disk = 'disk_memory';

SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (0);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (1);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (2);
SELECT * FROM log ORDER BY x;

TRUNCATE TABLE log;
DROP TABLE log;

CREATE TABLE log (x UInt8) ENGINE = Log () SETTINGS disk = 'disk_memory';

SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (0);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (1);
SELECT * FROM log ORDER BY x;
INSERT INTO log VALUES (2);
SELECT * FROM log ORDER BY x;

TRUNCATE TABLE log;
DROP TABLE log;
