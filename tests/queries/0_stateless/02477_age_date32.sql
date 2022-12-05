-- { echo }

-- Date32 vs Date32
SELECT age('second', toDate32('1927-01-01', 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('minute', toDate32('1927-01-01', 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('hour', toDate32('1927-01-01', 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('day', toDate32('1927-01-01', 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('week', toDate32('1927-01-01', 'UTC'), toDate32('1927-01-08', 'UTC'), 'UTC');
SELECT age('month', toDate32('1927-01-01', 'UTC'), toDate32('1927-02-01', 'UTC'), 'UTC');
SELECT age('quarter', toDate32('1927-01-01', 'UTC'), toDate32('1927-04-01', 'UTC'), 'UTC');
SELECT age('year', toDate32('1927-01-01', 'UTC'), toDate32('1928-01-01', 'UTC'), 'UTC');

-- With DateTime64
-- Date32 vs DateTime64
SELECT age('second', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-01-02 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('minute', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-01-02 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('hour', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-01-02 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('day', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-01-02 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('week', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-01-08 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('month', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-02-01 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('quarter', toDate32('1927-01-01', 'UTC'), toDateTime64('1927-04-01 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('year', toDate32('1927-01-01', 'UTC'), toDateTime64('1928-01-01 00:00:00', 3, 'UTC'), 'UTC');

-- DateTime64 vs Date32
SELECT age('second', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('minute', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('hour', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('day', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('week', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-01-08', 'UTC'), 'UTC');
SELECT age('month', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-02-01', 'UTC'), 'UTC');
SELECT age('quarter', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1927-04-01', 'UTC'), 'UTC');
SELECT age('year', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), toDate32('1928-01-01', 'UTC'), 'UTC');

-- With DateTime
-- Date32 vs DateTime
SELECT age('second', toDate32('2015-08-18', 'UTC'), toDateTime('2015-08-19 00:00:00', 'UTC'), 'UTC');
SELECT age('minute', toDate32('2015-08-18', 'UTC'), toDateTime('2015-08-19 00:00:00', 'UTC'), 'UTC');
SELECT age('hour', toDate32('2015-08-18', 'UTC'), toDateTime('2015-08-19 00:00:00', 'UTC'), 'UTC');
SELECT age('day', toDate32('2015-08-18', 'UTC'), toDateTime('2015-08-19 00:00:00', 'UTC'), 'UTC');
SELECT age('week', toDate32('2015-08-18', 'UTC'), toDateTime('2015-08-25 00:00:00', 'UTC'), 'UTC');
SELECT age('month', toDate32('2015-08-18', 'UTC'), toDateTime('2015-09-18 00:00:00', 'UTC'), 'UTC');
SELECT age('quarter', toDate32('2015-08-18', 'UTC'), toDateTime('2015-11-18 00:00:00', 'UTC'), 'UTC');
SELECT age('year', toDate32('2015-08-18', 'UTC'), toDateTime('2016-08-18 00:00:00', 'UTC'), 'UTC');

-- DateTime vs Date32
SELECT age('second', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('minute', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('hour', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('day', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('week', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-08-25', 'UTC'), 'UTC');
SELECT age('month', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-09-18', 'UTC'), 'UTC');
SELECT age('quarter', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2015-11-18', 'UTC'), 'UTC');
SELECT age('year', toDateTime('2015-08-18 00:00:00', 'UTC'), toDate32('2016-08-18', 'UTC'), 'UTC');

-- With Date
-- Date32 vs Date
SELECT age('second', toDate32('2015-08-18', 'UTC'), toDate('2015-08-19', 'UTC'), 'UTC');
SELECT age('minute', toDate32('2015-08-18', 'UTC'), toDate('2015-08-19', 'UTC'), 'UTC');
SELECT age('hour', toDate32('2015-08-18', 'UTC'), toDate('2015-08-19', 'UTC'), 'UTC');
SELECT age('day', toDate32('2015-08-18', 'UTC'), toDate('2015-08-19', 'UTC'), 'UTC');
SELECT age('week', toDate32('2015-08-18', 'UTC'), toDate('2015-08-25', 'UTC'), 'UTC');
SELECT age('month', toDate32('2015-08-18', 'UTC'), toDate('2015-09-18', 'UTC'), 'UTC');
SELECT age('quarter', toDate32('2015-08-18', 'UTC'), toDate('2015-11-18', 'UTC'), 'UTC');
SELECT age('year', toDate32('2015-08-18', 'UTC'), toDate('2016-08-18', 'UTC'), 'UTC');

-- Date vs Date32
SELECT age('second', toDate('2015-08-18', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('minute', toDate('2015-08-18', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('hour', toDate('2015-08-18', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('day', toDate('2015-08-18', 'UTC'), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('week', toDate('2015-08-18', 'UTC'), toDate32('2015-08-25', 'UTC'), 'UTC');
SELECT age('month', toDate('2015-08-18', 'UTC'), toDate32('2015-09-18', 'UTC'), 'UTC');
SELECT age('quarter', toDate('2015-08-18', 'UTC'), toDate32('2015-11-18', 'UTC'), 'UTC');
SELECT age('year', toDate('2015-08-18', 'UTC'), toDate32('2016-08-18', 'UTC'), 'UTC');

-- Const vs non-const columns
SELECT age('day', toDate32('1927-01-01', 'UTC'), materialize(toDate32('1927-01-02', 'UTC')), 'UTC');
SELECT age('day', toDate32('1927-01-01', 'UTC'), materialize(toDateTime64('1927-01-02 00:00:00', 3, 'UTC')), 'UTC');
SELECT age('day', toDateTime64('1927-01-01 00:00:00', 3, 'UTC'), materialize(toDate32('1927-01-02', 'UTC')), 'UTC');
SELECT age('day', toDate32('2015-08-18', 'UTC'), materialize(toDateTime('2015-08-19 00:00:00', 'UTC')), 'UTC');
SELECT age('day', toDateTime('2015-08-18 00:00:00', 'UTC'), materialize(toDate32('2015-08-19', 'UTC')), 'UTC');
SELECT age('day', toDate32('2015-08-18', 'UTC'), materialize(toDate('2015-08-19', 'UTC')), 'UTC');
SELECT age('day', toDate('2015-08-18', 'UTC'), materialize(toDate32('2015-08-19', 'UTC')), 'UTC');

-- Non-const vs const columns
SELECT age('day', materialize(toDate32('1927-01-01', 'UTC')), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('day', materialize(toDate32('1927-01-01', 'UTC')), toDateTime64('1927-01-02 00:00:00', 3, 'UTC'), 'UTC');
SELECT age('day', materialize(toDateTime64('1927-01-01 00:00:00', 3, 'UTC')), toDate32('1927-01-02', 'UTC'), 'UTC');
SELECT age('day', materialize(toDate32('2015-08-18', 'UTC')), toDateTime('2015-08-19 00:00:00', 'UTC'), 'UTC');
SELECT age('day', materialize(toDateTime('2015-08-18 00:00:00', 'UTC')), toDate32('2015-08-19', 'UTC'), 'UTC');
SELECT age('day', materialize(toDate32('2015-08-18', 'UTC')), toDate('2015-08-19', 'UTC'), 'UTC');
SELECT age('day', materialize(toDate('2015-08-18', 'UTC')), toDate32('2015-08-19', 'UTC'), 'UTC');

-- Non-const vs non-const columns
SELECT age('day', materialize(toDate32('1927-01-01', 'UTC')), materialize(toDate32('1927-01-02', 'UTC')), 'UTC');
SELECT age('day', materialize(toDate32('1927-01-01', 'UTC')), materialize(toDateTime64('1927-01-02 00:00:00', 3, 'UTC')), 'UTC');
SELECT age('day', materialize(toDateTime64('1927-01-01 00:00:00', 3, 'UTC')), materialize(toDate32('1927-01-02', 'UTC')), 'UTC');
SELECT age('day', materialize(toDate32('2015-08-18', 'UTC')), materialize(toDateTime('2015-08-19 00:00:00', 'UTC')), 'UTC');
SELECT age('day', materialize(toDateTime('2015-08-18 00:00:00', 'UTC')), materialize(toDate32('2015-08-19', 'UTC')), 'UTC');
SELECT age('day', materialize(toDate32('2015-08-18', 'UTC')), materialize(toDate('2015-08-19', 'UTC')), 'UTC');
SELECT age('day', materialize(toDate('2015-08-18', 'UTC')), materialize(toDate32('2015-08-19', 'UTC')), 'UTC');
