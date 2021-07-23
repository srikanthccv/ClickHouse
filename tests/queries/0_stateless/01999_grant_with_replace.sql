DROP USER IF EXISTS test_user_01999;

CREATE USER test_user_01999;
SHOW CREATE USER test_user_01999;

SELECT 'A';
SHOW GRANTS FOR test_user_01999;

GRANT SELECT ON db1.* TO test_user_01999;
GRANT SHOW ON db2.tb2 TO test_user_01999;

SELECT 'B';
SHOW GRANTS FOR test_user_01999;

GRANT SELECT(col1) ON db3.table TO test_user_01999 WITH REPLACE OPTION;

SELECT 'C';
SHOW GRANTS FOR test_user_01999;

GRANT SELECT(col3) ON db3.table3, SELECT(col1, col2) ON db4.table4 TO test_user_01999 WITH REPLACE OPTION;

SELECT 'D';
SHOW GRANTS FOR test_user_01999;

GRANT SELECT(cola) ON db5.table, INSERT(colb) ON db6.tb61, SHOW ON db7.* TO test_user_01999 WITH REPLACE OPTION;

SELECT 'E';
SHOW GRANTS FOR test_user_01999;

SELECT 'F';
GRANT SELECT ON all.* TO test_user_01999 WITH REPLACE OPTION;
SHOW GRANTS FOR test_user_01999;

SELECT 'G';
GRANT USAGE ON *.* TO test_user_01999 WITH REPLACE OPTION;
SHOW GRANTS FOR test_user_01999;

SELECT 'H';
DROP ROLE IF EXISTS test_role_01999;
CREATE role test_role_01999;
GRANT test_role_01999 to test_user_01999;
GRANT SELECT ON db1.tb1 TO test_user_01999;
SHOW GRANTS FOR test_user_01999;

SELECT 'I';
GRANT NONE ON *.* TO test_user_01999 WITH REPLACE OPTION;
SHOW GRANTS FOR test_user_01999;

SELECT 'J';
GRANT SHOW ON db8.* TO test_user_01999;
SHOW GRANTS FOR test_user_01999;

SELECT 'K';
GRANT NONE TO test_user_01999 WITH REPLACE OPTION;
SHOW GRANTS FOR test_user_01999;

DROP USER IF EXISTS test_user_01999;
DROP ROLE IF EXISTS test_role_01999;

SELECT 'L';
