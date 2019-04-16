DROP TABLE IF EXISTS tab;
CREATE TABLE tab (date Date, value UInt64, s String, m FixedString(16)) ENGINE = MergeTree(date, (date, value), 8);
INSERT INTO tab SELECT today() as date, number as value, '' as s, toFixedString('', 16) as m from system.numbers limit 42;
SET preferred_max_column_in_block_size_bytes = 32;
SELECT blockSize(), * from tab format Null;
SELECT 0;

