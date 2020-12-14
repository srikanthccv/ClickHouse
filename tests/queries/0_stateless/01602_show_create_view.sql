DROP DATABASE IF EXISTS test_1602;

CREATE DATABASE test_1602;

CREATE TABLE test_1602.tbl (`EventDate` DateTime, `CounterID` UInt32, `UserID` UInt32) ENGINE = MergeTree() PARTITION BY toYYYYMM(EventDate) ORDER BY (CounterID, EventDate, intHash32(UserID)) SETTINGS index_granularity = 8192;

CREATE VIEW test_1602.v AS SELECT * FROM test_1602.tbl;

CREATE MATERIALIZED VIEW test_1602.vv (`EventDate` DateTime, `CounterID` UInt32, `UserID` UInt32) ENGINE = MergeTree() PARTITION BY toYYYYMM(EventDate) ORDER BY (CounterID, EventDate, intHash32(UserID)) SETTINGS index_granularity = 8192 AS SELECT * FROM test_1602.tbl;


SET allow_experimental_live_view=1;

CREATE LIVE VIEW test_1602.vvv AS SELECT * FROM test_1602.tbl;

SHOW CREATE VIEW test_1602.v;

SHOW CREATE VIEW test_1602.vv;

SHOW CREATE VIEW test_1602.vvv;

SHOW CREATE VIEW test_1602.not_exist_view; -- { serverError 390 }

SHOW CREATE VIEW test_1602.tbl; -- { serverError 563 }

DROP DATABASE IF EXISTS test_1602;
