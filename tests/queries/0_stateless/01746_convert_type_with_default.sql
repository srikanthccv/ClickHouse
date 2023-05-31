select toUInt8OrDefault('1', cast(2 as UInt8));
select toUInt8OrDefault('1xx', cast(2 as UInt8));
select toInt8OrDefault('-1', cast(-2 as Int8));
select toInt8OrDefault('-1xx', cast(-2 as Int8));

select toUInt16OrDefault('1', cast(2 as UInt16));
select toUInt16OrDefault('1xx', cast(2 as UInt16));
select toInt16OrDefault('-1', cast(-2 as Int16));
select toInt16OrDefault('-1xx', cast(-2 as Int16));

select toUInt32OrDefault('1', cast(2 as UInt32));
select toUInt32OrDefault('1xx', cast(2 as UInt32));
select toInt32OrDefault('-1', cast(-2 as Int32));
select toInt32OrDefault('-1xx', cast(-2 as Int32));

select toUInt64OrDefault('1', cast(2 as UInt64));
select toUInt64OrDefault('1xx', cast(2 as UInt64));
select toInt64OrDefault('-1', cast(-2 as Int64));
select toInt64OrDefault('-1xx', cast(-2 as Int64));

select toInt128OrDefault('-1', cast(-2 as Int128));
select toInt128OrDefault('-1xx', cast(-2 as Int128));

select toUInt256OrDefault('1', cast(2 as UInt256));
select toUInt256OrDefault('1xx', cast(2 as UInt256));
select toInt256OrDefault('-1', cast(-2 as Int256));
select toInt256OrDefault('-1xx', cast(-2 as Int256));

SELECT toUUIDOrDefault('61f0c404-5cb3-11e7-907b-a6006ad3dba0', cast('59f0c404-5cb3-11e7-907b-a6006ad3dba0' as UUID));
SELECT toUUIDOrDefault('-----61f0c404-5cb3-11e7-907b-a6006ad3dba0', cast('59f0c404-5cb3-11e7-907b-a6006ad3dba0' as UUID));

select toDateOrDefault('1xxx');
select toDateOrDefault('2023-05-30');
select toDateOrDefault('2023-05-30', '2000-01-01'::Date);
select toDateOrDefault('1xx', '2023-05-30'::Date);
select toDateOrDefault(-1);

select toDateOrDefault(cast(19 as Int8));
select toDateOrDefault(cast(19 as UInt8));

select toDateOrDefault(cast(19 as Int16));
select toDateOrDefault(cast(19 as UInt16));

select toDateOrDefault(cast(19 as Int32));
select toDateOrDefault(cast(19 as UInt32));

select toDateOrDefault(cast(19 as Int64));
select toDateOrDefault(cast(19 as UInt64));

select toDateOrDefault(cast(19 as Int128));
select toDateOrDefault(cast(19 as UInt128));

select toDateOrDefault(cast(19 as Int256));
select toDateOrDefault(cast(19 as UInt256));

select toDateOrDefault(19507, '2000-01-01'::Date);
select toDateOrDefault(-1, '2000-01-01'::Date);

select toDateTimeOrDefault('2023-05-30 14:38:20');
select toDateTimeOrDefault('2023-05-30 14:38:20', 'UTC');
select toString(toDateTimeOrDefault('1xxx', 'Asia/Istanbul', '2023-05-30 14:38:20'::DateTime('Asia/Istanbul')), 'Asia/Istanbul');
select toDateTimeOrDefault(1685457500);

select toDateTimeOrDefault(cast(19 as Int8));
select toDateTimeOrDefault(cast(19 as UInt8));

select toDateTimeOrDefault(cast(19 as Int16));
select toDateTimeOrDefault(cast(19 as UInt16));

select toDateTimeOrDefault(cast(19 as Int32));
select toDateTimeOrDefault(cast(19 as UInt32));

select toDateTimeOrDefault(cast(19 as Int64));
select toDateTimeOrDefault(cast(19 as UInt64));

select toDateTimeOrDefault(cast(19 as Int128));
select toDateTimeOrDefault(cast(19 as UInt128));

select toDateTimeOrDefault(cast(19 as Int256));
select toDateTimeOrDefault(cast(19 as UInt256));