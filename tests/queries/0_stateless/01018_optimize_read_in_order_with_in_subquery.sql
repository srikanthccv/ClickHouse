SET max_threads = 2;
SET optimize_read_in_order = 1;

DROP TABLE IF EXISTS TESTTABLE4;
CREATE TABLE TESTTABLE4 (_id UInt64, pt String, l String ) 
ENGINE = MergeTree() PARTITION BY (pt) ORDER BY (_id);
INSERT INTO TESTTABLE4 VALUES (0,'1','1'), (1,'0','1');

SELECT _id FROM TESTTABLE4 PREWHERE l IN (select '1') ORDER BY _id DESC LIMIT 10;

DROP TABLE TESTTABLE4;
