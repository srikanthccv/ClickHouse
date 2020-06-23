DROP ROW POLICY IF EXISTS p1_01296, p2_01296, p3_01296, p4_01296, p5_01296 ON db_01296.table;
DROP ROW POLICY IF EXISTS p3_01296, p5_01296 ON db_01296.table2;
DROP DATABASE IF EXISTS db_01296;
DROP USER IF EXISTS u1_01296;

CREATE DATABASE db_01296;
USE db_01296;

SELECT '-- one policy';
CREATE POLICY p1_01296 ON table;
SHOW CREATE POLICY p1_01296 ON db_01296.table;
SHOW CREATE POLICY p1_01296 ON table;
ALTER POLICY p1_01296 ON table USING 1;
SHOW CREATE POLICY p1_01296 ON db_01296.table;
SHOW CREATE POLICY p1_01296 ON table;
DROP POLICY p1_01296 ON table;
DROP POLICY p1_01296 ON db_01296.table; -- { serverError 523 } -- Policy not found

SELECT '-- multiple policies';
CREATE ROW POLICY p1_01296, p2_01296 ON table USING 1;
CREATE USER u1_01296;
CREATE ROW POLICY p3_01296 ON table, table2 TO u1_01296;
CREATE ROW POLICY p4_01296 ON table, p5_01296 ON table2 USING a=b;
SHOW CREATE POLICY p1_01296 ON table;
SHOW CREATE POLICY p2_01296 ON table;
SHOW CREATE POLICY p3_01296 ON table;
SHOW CREATE POLICY p3_01296 ON table2;
SHOW CREATE POLICY p4_01296 ON table;
SHOW CREATE POLICY p5_01296 ON table2;
SHOW CREATE POLICY p1_01296 ON db_01296.table;
SHOW CREATE POLICY p2_01296 ON db_01296.table;
SHOW CREATE POLICY p3_01296 ON db_01296.table;
SHOW CREATE POLICY p3_01296 ON db_01296.table2;
SHOW CREATE POLICY p4_01296 ON db_01296.table;
SHOW CREATE POLICY p5_01296 ON db_01296.table2;
ALTER POLICY p1_01296, p2_01296 ON table TO ALL;
SHOW CREATE POLICY p1_01296 ON table;
SHOW CREATE POLICY p2_01296 ON table;
DROP POLICY p1_01296, p2_01296 ON table;
DROP POLICY p3_01296 ON table, table2;
DROP POLICY p4_01296 ON table, p5_01296 ON table2;
DROP POLICY p1_01296 ON db_01296.table; -- { serverError 523 } -- Policy not found
DROP POLICY p2_01296 ON db_01296.table; -- { serverError 523 } -- Policy not found
DROP POLICY p3_01296 ON db_01296.table; -- { serverError 523 } -- Policy not found
DROP POLICY p3_01296 ON db_01296.table2; -- { serverError 523 } -- Policy not found
DROP POLICY p4_01296 ON db_01296.table; -- { serverError 523 } -- Policy not found
DROP POLICY p5_01296 ON db_01296.table2; -- { serverError 523 } -- Policy not found

USE default;
DROP DATABASE db_01296;
DROP USER u1_01296;
