SELECT parseDateTimeBestEffortOrNull('2010-01-01');
SELECT parseDateTimeBestEffortOrNull('2010-01-01 01:01:01');
SELECT parseDateTimeBestEffortOrNull('2020-01-01 11:01:01 am');
SELECT parseDateTimeBestEffortOrNull('2020-01-01 11:01:01 pm');
SELECT parseDateTimeBestEffortOrNull('2020-01-01 12:01:01 am');
SELECT parseDateTimeBestEffortOrNull('2020-01-01 12:01:01 pm');
SELECT parseDateTimeBestEffortOrNull('01:01:01');
SELECT parseDateTimeBestEffortOrNull('20100');
SELECT parseDateTimeBestEffortOrNull('0100:0100:0000');
SELECT parseDateTimeBestEffortOrNull('x');
SELECT parseDateTimeBestEffortOrNull('');
SELECT parseDateTimeBestEffortOrNull('       ');
