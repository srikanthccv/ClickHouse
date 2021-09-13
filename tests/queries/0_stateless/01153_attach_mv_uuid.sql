-- Tags: no-ordinary-database, no-parallel

DROP TABLE IF EXISTS src;
DROP TABLE IF EXISTS mv;
DROP TABLE IF EXISTS ".inner_id.e15f3ab5-6cae-4df3-b879-f40deafd82c2";

CREATE TABLE src (n UInt64) ENGINE=MergeTree ORDER BY n;
CREATE MATERIALIZED VIEW mv (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n AS SELECT n, n * n AS n2 FROM src;
INSERT INTO src VALUES (1), (2);
SELECT * FROM mv ORDER BY n;
DETACH TABLE mv;
ATTACH TABLE mv;
INSERT INTO src VALUES (3), (4);
SELECT * FROM mv ORDER BY n;
DROP TABLE mv SYNC;

SET show_table_uuid_in_table_create_query_if_not_nil=1;
CREATE TABLE ".inner_id.e15f3ab5-6cae-4df3-b879-f40deafd82c2" (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n;
ATTACH MATERIALIZED VIEW mv UUID 'e15f3ab5-6cae-4df3-b879-f40deafd82c2' (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n AS SELECT n, n * n AS n2 FROM src;
SHOW CREATE TABLE mv;
INSERT INTO src VALUES (1), (2);
SELECT * FROM mv ORDER BY n;
DETACH TABLE mv;
ATTACH TABLE mv;
SHOW CREATE TABLE mv;
INSERT INTO src VALUES (3), (4);
SELECT * FROM mv ORDER BY n;
DROP TABLE mv SYNC;

CREATE TABLE ".inner_id.e15f3ab5-6cae-4df3-b879-f40deafd82c2" UUID '3bd68e3c-2693-4352-ad66-a66eba9e345e' (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n;
ATTACH MATERIALIZED VIEW mv UUID 'e15f3ab5-6cae-4df3-b879-f40deafd82c2' TO INNER UUID '3bd68e3c-2693-4352-ad66-a66eba9e345e' (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n AS SELECT n, n * n AS n2 FROM src;
SHOW CREATE TABLE mv;
INSERT INTO src VALUES (1), (2);
SELECT * FROM mv ORDER BY n;
DETACH TABLE mv;
ATTACH TABLE mv;
SHOW CREATE TABLE mv;
INSERT INTO src VALUES (3), (4);
SELECT * FROM mv ORDER BY n;
DROP TABLE mv SYNC;

ATTACH MATERIALIZED VIEW mv UUID '3bd68e3c-2693-4352-ad66-a66eba9e345e' TO INNER UUID '3bd68e3c-2693-4352-ad66-a66eba9e345e' (n Int32, n2 Int64) ENGINE = MergeTree PARTITION BY n % 10 ORDER BY n AS SELECT n, n * n AS n2 FROM src; -- { serverError 36 }

DROP TABLE src;
