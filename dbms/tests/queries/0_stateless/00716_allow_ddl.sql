SET send_logs_level = 'none';
SET allow_ddl = 0;

CREATE DATABASE some_db; -- { serverError 392 } 
CREATE TABLE test.some_table(a Int32) ENGINE = Memory; -- { serverError 392}
ALTER TABLE test.some_table DELETE WHERE 1; -- { serverError 392}
RENAME TABLE test.some_table TO test.some_table1; -- { serverError 392}
SET allow_ddl = 1; -- { serverError 392}
