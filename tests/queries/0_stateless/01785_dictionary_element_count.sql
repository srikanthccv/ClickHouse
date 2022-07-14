-- Tags: no-parallel

DROP DATABASE IF EXISTS _01785_db;
CREATE DATABASE _01785_db;

DROP TABLE IF EXISTS _01785_db.simple_key_source_table;
CREATE TABLE _01785_db.simple_key_source_table
(
    id UInt64,
    value String
) ENGINE = TinyLog();

INSERT INTO _01785_db.simple_key_source_table VALUES (1, 'First');
INSERT INTO _01785_db.simple_key_source_table VALUES (1, 'First');

DROP DICTIONARY IF EXISTS _01785_db.simple_key_flat_dictionary;
CREATE DICTIONARY _01785_db.simple_key_flat_dictionary
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() DB '_01785_db' TABLE 'simple_key_source_table'))
LAYOUT(FLAT())
LIFETIME(MIN 0 MAX 1000);

SELECT * FROM _01785_db.simple_key_flat_dictionary;
SELECT name, database, element_count FROM system.dictionaries WHERE database = '_01785_db' AND name = 'simple_key_flat_dictionary';

DROP DICTIONARY _01785_db.simple_key_flat_dictionary;

CREATE DICTIONARY _01785_db.simple_key_hashed_dictionary
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() DB '_01785_db' TABLE 'simple_key_source_table'))
LAYOUT(HASHED())
LIFETIME(MIN 0 MAX 1000);

SELECT * FROM _01785_db.simple_key_hashed_dictionary;
SELECT name, database, element_count FROM system.dictionaries WHERE database = '_01785_db' AND name = 'simple_key_hashed_dictionary';

DROP DICTIONARY _01785_db.simple_key_hashed_dictionary;

CREATE DICTIONARY _01785_db.simple_key_cache_dictionary
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() DB '_01785_db' TABLE 'simple_key_source_table'))
LAYOUT(CACHE(SIZE_IN_CELLS 100000))
LIFETIME(MIN 0 MAX 1000);

SELECT toUInt64(1) as key, dictGet('_01785_db.simple_key_cache_dictionary', 'value', key);
SELECT name, database, element_count FROM system.dictionaries WHERE database = '_01785_db' AND name = 'simple_key_cache_dictionary';

DROP DICTIONARY _01785_db.simple_key_cache_dictionary;

DROP TABLE _01785_db.simple_key_source_table;

DROP TABLE IF EXISTS _01785_db.complex_key_source_table;
CREATE TABLE _01785_db.complex_key_source_table
(
    id UInt64,
    id_key String,
    value String
) ENGINE = TinyLog();

INSERT INTO _01785_db.complex_key_source_table VALUES (1, 'FirstKey', 'First');
INSERT INTO _01785_db.complex_key_source_table VALUES (1, 'FirstKey', 'First');

CREATE DICTIONARY _01785_db.complex_key_hashed_dictionary
(
    id UInt64,
    id_key String,
    value String
)
PRIMARY KEY id, id_key
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() DB '_01785_db' TABLE 'complex_key_source_table'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(MIN 0 MAX 1000);

SELECT * FROM _01785_db.complex_key_hashed_dictionary;
SELECT name, database, element_count FROM system.dictionaries WHERE database = '_01785_db' AND name = 'complex_key_hashed_dictionary';

DROP DICTIONARY _01785_db.complex_key_hashed_dictionary;

DROP TABLE _01785_db.complex_key_source_table;

DROP DATABASE _01785_db;
