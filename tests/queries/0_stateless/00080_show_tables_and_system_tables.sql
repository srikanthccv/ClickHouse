
DROP DATABASE IF EXISTS {CLICKHOUSE_DATABASE:Identifier};

CREATE DATABASE {CLICKHOUSE_DATABASE:Identifier};

CREATE TABLE {CLICKHOUSE_DATABASE:Identifier}.A (A UInt8) ENGINE = TinyLog;
CREATE TABLE {CLICKHOUSE_DATABASE:Identifier}.B (A UInt8) ENGINE = TinyLog;

SHOW TABLES from {CLICKHOUSE_DATABASE:Identifier};
SHOW TABLES in system where engine like '%System%' and name in ('numbers', 'one');

SELECT name, toUInt32(metadata_modification_time) > 0, engine_full, create_table_query FROM system.tables WHERE database = currentDatabase() ORDER BY name FORMAT TSVRaw;

CREATE TEMPORARY TABLE test_temporary_table (id UInt64);
SELECT name FROM system.tables WHERE is_temporary = 1 AND name = 'test_temporary_table';

CREATE TABLE {CLICKHOUSE_DATABASE:Identifier}.test_log(id UInt64) ENGINE = Log;
CREATE MATERIALIZED VIEW {CLICKHOUSE_DATABASE:Identifier}.test_materialized ENGINE = Log AS SELECT * FROM {CLICKHOUSE_DATABASE:Identifier}.test_log;
SELECT dependencies_database, dependencies_table FROM system.tables WHERE name = 'test_log';

DROP DATABASE {CLICKHOUSE_DATABASE:Identifier};


-- Check that create_table_query works for system tables and unusual Databases
DROP DATABASE IF EXISTS test_DatabaseMemory;
CREATE DATABASE test_DatabaseMemory ENGINE = Memory;
CREATE TABLE test_DatabaseMemory.A (A UInt8) ENGINE = Null;

SELECT sum(ignore(*, metadata_modification_time, engine_full, create_table_query)) FROM system.tables WHERE database = 'test_DatabaseMemory';

DROP DATABASE test_DatabaseMemory;
