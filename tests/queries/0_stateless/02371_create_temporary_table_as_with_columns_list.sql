CREATE TEMPORARY TABLE test_02327 (name String) AS SELECT * FROM VALUES(('Vasya'), ('Petya'));
SELECT * FROM test_02327;
DROP TABLE test_02327;
