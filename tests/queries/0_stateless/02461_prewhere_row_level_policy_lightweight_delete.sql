DROP TABLE IF EXISTS url_na_log;

CREATE TABLE url_na_log(SiteId UInt32, DateVisit Date)
ENGINE = MergeTree()
ORDER BY (SiteId, DateVisit)
SETTINGS index_granularity = 8192, min_bytes_for_wide_part = 0;

-- Insert some data to have 110K rows in the range 2022-08-10 .. 2022-08-20 and some more rows before and after that range
insert into url_na_log select 209, '2022-08-09' from numbers(10000);
insert into url_na_log select 209, '2022-08-10' from numbers(10000);
insert into url_na_log select 209, '2022-08-11' from numbers(10000);
insert into url_na_log select 209, '2022-08-12' from numbers(10000);
insert into url_na_log select 209, '2022-08-13' from numbers(10000);
insert into url_na_log select 209, '2022-08-14' from numbers(10000);
insert into url_na_log select 209, '2022-08-15' from numbers(10000);
insert into url_na_log select 209, '2022-08-16' from numbers(10000);
insert into url_na_log select 209, '2022-08-17' from numbers(10000);
insert into url_na_log select 209, '2022-08-18' from numbers(10000);
insert into url_na_log select 209, '2022-08-19' from numbers(10000);
insert into url_na_log select 209, '2022-08-20' from numbers(10000);
insert into url_na_log select 209, '2022-08-21' from numbers(10000);


SET mutations_sync=2;
SET allow_experimental_lightweight_delete=1;

OPTIMIZE TABLE url_na_log FINAL;

-- { echoOn }

SELECT count() FROM url_na_log;
SELECT rows FROM system.parts WHERE database = currentDatabase() AND table = 'url_na_log' AND active;
SELECT count() FROM url_na_log PREWHERE DateVisit >= '2022-08-10' AND DateVisit <= '2022-08-20' WHERE SiteId = 209 SETTINGS max_block_size = 200000, max_threads = 1;


-- Delete more than a half rows (60K)  from the range 2022-08-10 .. 2022-08-20
-- There should be 50K rows remaining in this range
DELETE FROM url_na_log WHERE SiteId = 209 AND DateVisit >= '2022-08-13' AND DateVisit <= '2022-08-18';

SELECT count() FROM url_na_log;
SELECT rows FROM system.parts WHERE database = currentDatabase() AND table = 'url_na_log' AND active;
SELECT count() FROM url_na_log PREWHERE DateVisit >= '2022-08-10' AND DateVisit <= '2022-08-20' WHERE SiteId = 209 SETTINGS max_block_size = 200000, max_threads = 1;


-- Hide more than a half of remaining rows (30K) from the range 2022-08-10 .. 2022-08-20 using row policy
-- Now the this range should have 20K rows left
CREATE ROW POLICY url_na_log_policy0 ON url_na_log FOR SELECT USING DateVisit < '2022-08-11' or DateVisit > '2022-08-19' TO default;

SELECT count() FROM url_na_log;
SELECT rows FROM system.parts WHERE database = currentDatabase() AND table = 'url_na_log' AND active;
SELECT count() FROM url_na_log PREWHERE DateVisit >= '2022-08-10' AND DateVisit <= '2022-08-20' WHERE SiteId = 209 SETTINGS max_block_size = 200000, max_threads = 1;
