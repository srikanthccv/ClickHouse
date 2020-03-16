DROP TABLE IF EXISTS test;
DROP TABLE IF EXISTS test_view;

CREATE TABLE test(date Date, id Int8, name String, value Int64) ENGINE = MergeTree(date, (id, date), 8192);
CREATE VIEW test_view AS SELECT * FROM test;

SET enable_debug_queries = 1;
SET enable_optimize_predicate_expression = 1;

-- Optimize predicate expression with view
ANALYZE SELECT * FROM test_view WHERE id = 1;
ANALYZE SELECT * FROM test_view WHERE id = 2;
ANALYZE SELECT id FROM test_view WHERE id  = 1;
ANALYZE SELECT s.id FROM test_view AS s WHERE s.id = 1;

SELECT * FROM (SELECT toUInt64(b), sum(id) AS b FROM test) WHERE `toUInt64(sum(id))` = 3; -- { serverError 47 }

DROP TABLE IF EXISTS test;
DROP TABLE IF EXISTS test_view;
