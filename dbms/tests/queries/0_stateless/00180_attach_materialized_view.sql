DROP TABLE IF EXISTS test.t;
DROP TABLE IF EXISTS test.mv;
DROP TABLE IF EXISTS test.`.inner.mv`;

CREATE TABLE test.t (x UInt8) ENGINE = Null;
CREATE MATERIALIZED VIEW test.mv ENGINE = Null AS SELECT * FROM test.t;

DETACH TABLE test.mv;
ATTACH MATERIALIZED VIEW test.mv ENGINE = Null AS SELECT * FROM test.t;

DROP TABLE test.t;
DROP TABLE test.mv;
