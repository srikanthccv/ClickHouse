DROP USER IF EXISTS test_user_01999;

CREATE USER test_user_01999;
SHOW CREATE USER test_user_01999;

SELECT 'A';
SHOW GRANTS FOR test_user_01999;

GRANT SELECT ON db1.* TO test_user_01999;
GRANT SHOW ON db2.table TO test_user_01999;

SELECT 'B';
SHOW GRANTS FOR test_user_01999;

REPLACE GRANT SELECT(col1) ON db3.table TO test_user_01999;

SELECT 'C';
SHOW GRANTS FOR test_user_01999;

REPLACE GRANT SELECT(col3) ON db3.table3, SELECT(col1, col2) ON db4.table4 TO test_user_01999;

SELECT 'D';
SHOW GRANTS FOR test_user_01999;

REPLACE GRANT SELECT(cola) ON db5.table, INSERT(colb) ON db6.tb61, SHOW ON db7.* TO test_user_01999;

SELECT 'E';
SHOW GRANTS FOR test_user_01999;

SELECT 'F';
REPLACE GRANT SELECT ON all.* TO test_user_01999;
SHOW GRANTS FOR test_user_01999;

SELECT 'G';
REPLACE GRANT USAGE ON *.* TO test_user_01999;
SHOW GRANTS FOR test_user_01999;

DROP USER test_user_01999;

