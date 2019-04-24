DROP TABLE IF EXISTS test;

CREATE TABLE test(x Int32) ENGINE = Log;
INSERT INTO test VALUES (123);

SELECT a1 
FROM
(
    SELECT x AS a1, x AS a2 FROM test
    UNION ALL
    SELECT x, x FROM test
);

DROP TABLE test;
