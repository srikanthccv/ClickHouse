SELECT '--TUMBLE--';
SELECT TUMBLE(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY);
SELECT TUMBLE_START(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY);
SELECT toDateTime(TUMBLE_START(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, 'US/Samoa'), 'US/Samoa');
SELECT toDateTime(TUMBLE_START(toDateTime('2020-01-09 12:00:01', 'US/Samoa'), INTERVAL '1' DAY), 'US/Samoa');
SELECT TUMBLE_START(TUMBLE(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY));
SELECT TUMBLE_END(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY);
SELECT toDateTime(TUMBLE_END(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, 'US/Samoa'), 'US/Samoa');
SELECT toDateTime(TUMBLE_END(toDateTime('2020-01-09 12:00:01', 'US/Samoa'), INTERVAL '1' DAY), 'US/Samoa');
SELECT TUMBLE_END(TUMBLE(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY));

SELECT '--HOP--';
SELECT HOP(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY);
SELECT HOP_START(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY);
SELECT toDateTime(HOP_START(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY, 'US/Samoa'), 'US/Samoa');
SELECT toDateTime(HOP_START(toDateTime('2020-01-09 12:00:01', 'US/Samoa'), INTERVAL '1' DAY, INTERVAL '3' DAY), 'US/Samoa');
SELECT HOP_START(HOP(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY));
SELECT HOP_END(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY);
SELECT toDateTime(HOP_END(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY, 'US/Samoa'), 'US/Samoa');
SELECT toDateTime(HOP_END(toDateTime('2020-01-09 12:00:01', 'US/Samoa'), INTERVAL '1' DAY, INTERVAL '3' DAY), 'US/Samoa');
SELECT HOP_END(HOP(toDateTime('2020-01-09 12:00:01'), INTERVAL '1' DAY, INTERVAL '3' DAY));