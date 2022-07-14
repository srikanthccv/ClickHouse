-- Tags: no-parallel

DROP DATABASE IF EXISTS _01681_database_for_flat_dictionary;
CREATE DATABASE _01681_database_for_flat_dictionary;

CREATE TABLE _01681_database_for_flat_dictionary.simple_key_simple_attributes_source_table
(
   id UInt64,
   value_first String,
   value_second String
)
ENGINE = TinyLog;

INSERT INTO _01681_database_for_flat_dictionary.simple_key_simple_attributes_source_table VALUES(0, 'value_0', 'value_second_0');
INSERT INTO _01681_database_for_flat_dictionary.simple_key_simple_attributes_source_table VALUES(1, 'value_1', 'value_second_1');
INSERT INTO _01681_database_for_flat_dictionary.simple_key_simple_attributes_source_table VALUES(2, 'value_2', 'value_second_2');

CREATE DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes
(
   id UInt64,
   value_first String DEFAULT 'value_first_default',
   value_second String DEFAULT 'value_second_default'
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() USER 'default' TABLE 'simple_key_simple_attributes_source_table'))
LIFETIME(MIN 1 MAX 1000)
LAYOUT(FLAT());

SELECT 'Dictionary flat_dictionary_simple_key_simple_attributes';
SELECT 'dictGet existing value';
SELECT dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_first', number) as value_first,
    dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_second', number) as value_second FROM system.numbers LIMIT 3;
SELECT 'dictGet with non existing value';
SELECT dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_first', number) as value_first,
    dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_second', number) as value_second FROM system.numbers LIMIT 4;
SELECT 'dictGetOrDefault existing value';
SELECT dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_first', number, toString('default')) as value_first,
    dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_second', number, toString('default')) as value_second FROM system.numbers LIMIT 3;
SELECT 'dictGetOrDefault non existing value';
SELECT dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_first', number, toString('default')) as value_first,
    dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', 'value_second', number, toString('default')) as value_second FROM system.numbers LIMIT 4;
SELECT 'dictHas';
SELECT dictHas('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes', number) FROM system.numbers LIMIT 4;

DROP DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_simple_attributes;
DROP TABLE _01681_database_for_flat_dictionary.simple_key_simple_attributes_source_table;

CREATE TABLE _01681_database_for_flat_dictionary.simple_key_complex_attributes_source_table
(
   id UInt64,
   value_first String,
   value_second Nullable(String)
)
ENGINE = TinyLog;

INSERT INTO _01681_database_for_flat_dictionary.simple_key_complex_attributes_source_table VALUES(0, 'value_0', 'value_second_0');
INSERT INTO _01681_database_for_flat_dictionary.simple_key_complex_attributes_source_table VALUES(1, 'value_1', NULL);
INSERT INTO _01681_database_for_flat_dictionary.simple_key_complex_attributes_source_table VALUES(2, 'value_2', 'value_second_2');

CREATE DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes
(
   id UInt64,
   value_first String DEFAULT 'value_first_default',
   value_second Nullable(String) DEFAULT 'value_second_default'
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() USER 'default' TABLE 'simple_key_complex_attributes_source_table'))
LIFETIME(MIN 1 MAX 1000)
LAYOUT(FLAT());

SELECT 'Dictionary flat_dictionary_simple_key_complex_attributes';
SELECT 'dictGet existing value';
SELECT dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_first', number) as value_first,
    dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_second', number) as value_second FROM system.numbers LIMIT 3;
SELECT 'dictGet with non existing value';
SELECT dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_first', number) as value_first,
    dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_second', number) as value_second FROM system.numbers LIMIT 4;
SELECT 'dictGetOrDefault existing value';
SELECT dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_first', number, toString('default')) as value_first,
    dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_second', number, toString('default')) as value_second FROM system.numbers LIMIT 3;
SELECT 'dictGetOrDefault non existing value';
SELECT dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_first', number, toString('default')) as value_first,
    dictGetOrDefault('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', 'value_second', number, toString('default')) as value_second FROM system.numbers LIMIT 4;
SELECT 'dictHas';
SELECT dictHas('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes', number) FROM system.numbers LIMIT 4;

DROP DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_complex_attributes;
DROP TABLE _01681_database_for_flat_dictionary.simple_key_complex_attributes_source_table;

CREATE TABLE _01681_database_for_flat_dictionary.simple_key_hierarchy_table
(
    id UInt64,
    parent_id UInt64
) ENGINE = TinyLog();

INSERT INTO _01681_database_for_flat_dictionary.simple_key_hierarchy_table VALUES (1, 0);
INSERT INTO _01681_database_for_flat_dictionary.simple_key_hierarchy_table VALUES (2, 1);
INSERT INTO _01681_database_for_flat_dictionary.simple_key_hierarchy_table VALUES (3, 1);
INSERT INTO _01681_database_for_flat_dictionary.simple_key_hierarchy_table VALUES (4, 2);

CREATE DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_hierarchy
(
   id UInt64,
   parent_id UInt64 HIERARCHICAL
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT tcpPort() USER 'default' TABLE 'simple_key_hierarchy_table'))
LIFETIME(MIN 1 MAX 1000)
LAYOUT(FLAT());

SELECT 'Dictionary flat_dictionary_simple_key_hierarchy';
SELECT 'dictGet';
SELECT dictGet('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_hierarchy', 'parent_id', number) FROM system.numbers LIMIT 5;
SELECT 'dictGetHierarchy';
SELECT dictGetHierarchy('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_hierarchy', toUInt64(1));
SELECT dictGetHierarchy('_01681_database_for_flat_dictionary.flat_dictionary_simple_key_hierarchy', toUInt64(4));

DROP DICTIONARY _01681_database_for_flat_dictionary.flat_dictionary_simple_key_hierarchy;
DROP TABLE _01681_database_for_flat_dictionary.simple_key_hierarchy_table;

DROP DATABASE _01681_database_for_flat_dictionary;
