DROP DATABASE IF EXISTS 01915_db;
CREATE DATABASE 01915_db ENGINE=Atomic;

DROP TABLE IF EXISTS 01915_db.table_1;
CREATE TABLE 01915_db.table_1 (id UInt64, value String) ENGINE=TinyLog;

DROP TABLE IF EXISTS 01915_db.table_2;
CREATE TABLE 01915_db.table_2 (id UInt64, value String) ENGINE=TinyLog;

INSERT INTO 01915_db.table_1 VALUES (1, 'Table1');
INSERT INTO 01915_db.table_2 VALUES (2, 'Table2');

DROP DICTIONARY IF EXISTS 01915_db.dictionary_1;
CREATE DICTIONARY 01915_db.dictionary_1 (id UInt64, value String)
PRIMARY KEY id
LAYOUT(DIRECT())
SOURCE(CLICKHOUSE(DB '01915_db' TABLE 'table_1'));

DROP DICTIONARY IF EXISTS 01915_db.dictionary_2;
CREATE DICTIONARY 01915_db.dictionary_2 (id UInt64, value String)
PRIMARY KEY id
LAYOUT(DIRECT())
SOURCE(CLICKHOUSE(DB '01915_db' TABLE 'table_2'));

SELECT * FROM 01915_db.dictionary_1;
SELECT * FROM 01915_db.dictionary_2;

EXCHANGE DICTIONARIES 01915_db.dictionary_1 AND 01915_db.dictionary_2;

SELECT * FROM 01915_db.dictionary_1;
SELECT * FROM 01915_db.dictionary_2;

DROP DICTIONARY 01915_db.dictionary_1;
DROP DICTIONARY 01915_db.dictionary_2;

DROP TABLE 01915_db.table_1;
DROP TABLE 01915_db.table_2;

DROP DATABASE 01915_db;
