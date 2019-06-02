CREATE TABLE mt_00168 (EventDate Date, UTCEventTime DateTime, MoscowEventDate Date DEFAULT toDate(UTCEventTime)) ENGINE = MergeTree(EventDate, UTCEventTime, 8192);
CREATE TABLE mt_00168_buffer AS mt_00168 ENGINE = Buffer(default, mt_00168, 16, 10, 100, 10000, 1000000, 10000000, 100000000);
DESC TABLE mt_00168;
DESC TABLE mt_00168_buffer;
INSERT INTO mt_00168 (EventDate, UTCEventTime) VALUES ('2015-06-09', '2015-06-09 01:02:03');
SELECT * FROM mt_00168_buffer;
INSERT INTO mt_00168_buffer (EventDate, UTCEventTime) VALUES ('2015-06-09', '2015-06-09 01:02:03');
SELECT * FROM mt_00168_buffer;
