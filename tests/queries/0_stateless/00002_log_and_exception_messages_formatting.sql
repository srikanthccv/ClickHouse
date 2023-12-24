-- Tags: no-parallel, no-fasttest
-- no-parallel because we want to run this test when most of the other tests already passed

-- If this test fails, see the "Top patterns of log messages" diagnostics in the end of run.log

system flush logs;
drop table if exists logs;
create view logs as select * from system.text_log where now() - toIntervalMinute(120) < event_time;

-- Check that we don't have too many messages formatted with fmt::runtime or strings concatenation.
-- 0.001 threshold should be always enough, the value was about 0.00025
WITH 0.001 AS threshold
SELECT
    'runtime messages',
    greatest(coalesce(sum(length(message_format_string) = 0) / countOrNull(), 0) as v, threshold),
    v <= threshold ? [] :
        (SELECT groupArray((message, c)) FROM (
            SELECT message, count() as c FROM logs
            WHERE
                length(message_format_string) = 0
              AND message not like '% Received from %clickhouse-staging.com:9440%'
              AND source_file not like '%/AWSLogger.cpp%'
            GROUP BY message ORDER BY c LIMIT 10
        ))
FROM logs
WHERE
    message NOT LIKE '% Received from %clickhouse-staging.com:9440%'
  AND source_file not like '%/AWSLogger.cpp%';

-- Check the same for exceptions. The value was 0.03
WITH 0.05 AS threshold
SELECT
    'runtime exceptions',
    greatest(coalesce(sum(length(message_format_string) = 0) / countOrNull(), 0) as v, threshold),
    v <= threshold ? [] :
        (SELECT groupArray((message, c)) FROM (
            SELECT message, count() as c FROM logs
            WHERE
                length(message_format_string) = 0
              AND (message like '%DB::Exception%' or message like '%Coordination::Exception%')
              AND message not like '% Received from %clickhouse-staging.com:9440%'
            GROUP BY message ORDER BY c LIMIT 10
        ))
FROM logs
WHERE
    message NOT LIKE '% Received from %clickhouse-staging.com:9440%'
  AND (message like '%DB::Exception%' or message like '%Coordination::Exception%');

WITH 0.01 AS threshold
SELECT
    'unknown runtime exceptions',
    greatest(coalesce(sum(length(message_format_string) = 0) / countOrNull(), 0) as v, threshold),
    v <= threshold ? [] :
        (SELECT groupArray((message, c)) FROM (
            SELECT message, count() as c FROM logs
            WHERE
                length(message_format_string) = 0
              AND (message like '%DB::Exception%' or message like '%Coordination::Exception%')
              AND message not like '% Received from %' and message not like '%(SYNTAX_ERROR)%'
            GROUP BY message ORDER BY c LIMIT 10
        ))
FROM logs
WHERE
  (message like '%DB::Exception%' or message like '%Coordination::Exception%')
  AND message not like '% Received from %' and message not like '%(SYNTAX_ERROR)%';


-- FIXME some of the following messages are not informative and it has to be fixed
create temporary table known_short_messages (s String) as select * from (select
['', '{} ({})', '({}) Keys: {}', '({}) {}', 'Aggregating', 'Became leader', 'Cleaning queue',
'Creating set.', 'Cyclic aliases', 'Detaching {}', 'Executing {}', 'Fire events: {}',
'Found part {}', 'Loaded queue', 'No sharding key', 'No tables', 'Query: {}', 'Removed',
'Removed part {}', 'Removing parts.', 'Request URI: {}', 'Sending part {}',
'Sent handshake', 'Starting {}', 'Will mimic {}', 'Writing to {}', 'dropIfEmpty',
'loadAll {}', '{} ({}:{})', '{} -> {}', '{} {}', '{}: {}', '{}%', 'Read object: {}',
'New segment: {}', 'Convert overflow', 'Division by zero', 'Files set to {}',
'Bytes set to {}', 'Numeric overflow', 'Invalid mode: {}',
'Write file: {}', 'Unable to parse JSONPath', 'Host is empty in S3 URI.', 'Expected end of line',
'inflate failed: {}{}', 'Center is not valid', 'Column ''{}'' is ambiguous', 'Cannot parse object', 'Invalid date: {}',
'There is no cache by name: {}', 'No part {} in table', '`{}` should be a String', 'There are duplicate id {}',
'Invalid replica name: {}', 'Unexpected value {} in enum', 'Unknown BSON type: {}', 'Point is not valid',
'Invalid qualified name: {}', 'INTO OUTFILE is not allowed', 'Arguments must not be NaN', 'Cell is not valid',
'brotli decode error{}', 'Invalid H3 index: {}', 'Too large node state size', 'No additional keys found.',
'Attempt to read after EOF.', 'Replication was stopped', '{}	building file infos', 'Cannot parse uuid {}',
'Query was cancelled', 'Cancelled merging parts', 'Cancelled mutating parts', 'Log pulling is cancelled',
'Transaction was cancelled', 'Could not find table: {}', 'Table {} does not exist',
'Database {} does not exist', 'Dictionary ({}) not found', 'Unknown table function {}',
'Unknown format {}', 'Unknown explain kind ''{}''', 'Unknown setting {}', 'Unknown input format {}',
'Unknown identifier: ''{}''', 'User name is empty', 'Expected function, got: {}',
'Attempt to read after eof', 'String size is too big ({}), maximum: {}',
'Processed: {}%', 'Creating {}: {}', 'Table {}.{} doesn''t exist', 'Invalid cache key hex: {}',
'User has been dropped', 'Illegal type {} of argument of function {}. Should be DateTime or DateTime64',
'Unknown statistic column: {}',
'Bad SSH public key provided', 'Database {} does not exist', 'Substitution {} is not set', 'Invalid cache key hex: {}'
] as arr) array join arr;

-- Check that we don't have too many short meaningless message patterns.
WITH 1 AS max_messages
select 'messages shorter than 10',
    (uniqExact(message_format_string) as c) <= max_messages,
    c <= max_messages ? [] : groupUniqArray(message_format_string)
    from logs
    where length(message_format_string) < 10 and message_format_string not in known_short_messages;

-- Same as above. Feel free to update the threshold or remove this query if really necessary
WITH 3 AS max_messages
select 'messages shorter than 16',
    (uniqExact(message_format_string) as c) <= max_messages,
    c <= max_messages ? [] : groupUniqArray(message_format_string)
    from logs
    where length(message_format_string) < 16 and message_format_string not in known_short_messages;

-- Unlike above, here we look at length of the formatted message, not format string. Most short format strings are fine because they end up decorated with context from outer or inner exceptions, e.g.:
-- "Expected end of line" -> "Code: 117. DB::Exception: Expected end of line: (in file/uri /var/lib/clickhouse/user_files/data_02118): (at row 1)"
-- But we have to cut out the boilerplate, e.g.:
-- "Code: 60. DB::Exception: Table default.a does not exist. (UNKNOWN_TABLE), Stack trace" -> "Table default.a does not exist."
-- This table currently doesn't have enough information to do this reliably, so we just regex search for " (ERROR_NAME_IN_CAPS)" and hope that's good enough.
-- For the "Code: 123. DB::Exception: " part, we just subtract 26 instead of searching for it. Because sometimes it's not at the start, e.g.:
-- "Unexpected error, will try to restart main thread: Code: 341. DB::Exception: Unexpected error: Code: 57. DB::Exception:[...]"
WITH 3 AS max_messages
select 'exceptions shorter than 30',
    (uniqExact(message_format_string) as c) <= max_messages,
    c <= max_messages ? [] : groupUniqArray(message_format_string)
    from logs
    where message ilike '%DB::Exception%' and if(length(extract(message, '(.*)\\([A-Z0-9_]+\\)')) as pref > 0, pref, length(message)) < 30 + 26 and message_format_string not in known_short_messages;

-- Avoid too noisy messages: top 1 message frequency must be less than 30%. We should reduce the threshold
WITH 0.30 as threshold
select
    'noisy messages',
    greatest(coalesce(((select message_format_string, count() from logs group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
    r <= threshold ? '' : top_message.1;

-- Same as above, but excluding Test level (actually finds top 1 Trace message)
with 0.16 as threshold
select
    'noisy Trace messages',
    greatest(coalesce(((select message_format_string, count() from logs where level = 'Trace' and message_format_string not in ('Access granted: {}{}', '{} -> {}')
                        group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
    r <= threshold ? '' : top_message.1;

-- Same as above for Debug
WITH 0.09 as threshold
select 'noisy Debug messages',
       greatest(coalesce(((select message_format_string, count() from logs where level = 'Debug' group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
       r <= threshold ? '' : top_message.1;

-- Same as above for Info
WITH 0.05 as threshold
select 'noisy Info messages',
       greatest(coalesce(((select message_format_string, count() from logs where level = 'Information' group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
       r <= threshold ? '' : top_message.1;

-- Same as above for Warning
with 0.01 as threshold
select
    'noisy Warning messages',
    greatest(coalesce(((select message_format_string, count() from logs where level = 'Warning' and message_format_string not in ('Not enabled four letter command {}')
                       group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
    r <= threshold ? '' : top_message.1;

-- Same as above for Error
WITH 0.03 as threshold
select 'noisy Error messages',
    greatest(coalesce(((select message_format_string, count() from logs where level = 'Error' group by message_format_string order by count() desc limit 1) as top_message).2, 0) / (select count() from logs), threshold) as r,
    r <= threshold ? '' : top_message.1;

select 'no Fatal messages', count() from logs where level = 'Fatal';


-- Avoid too noisy messages: limit the number of messages with high frequency
select 'number of too noisy messages',
    greatest(count(), 3) from (select count() / (select count() from logs) as freq, message_format_string from logs group by message_format_string having freq > 0.10);
select 'number of noisy messages',
    greatest(count(), 10) from (select count() / (select count() from logs) as freq, message_format_string from logs group by message_format_string having freq > 0.05);

-- Each message matches its pattern (returns 0 rows)
-- Note: maybe we should make it stricter ('Code:%Exception: '||s||'%'), but it's not easy because of addMessage
select 'incorrect patterns', greatest(uniqExact(message_format_string), 15) from (
    select message_format_string, any(message) as any_message from logs
    where ((rand() % 8) = 0)
    and message not like (replaceRegexpAll(message_format_string, '{[:.0-9dfx]*}', '%') as s)
    and message not like (s || ' (skipped % similar messages)')
    and message not like ('%Exception: '||s||'%')
    and message not like ('%(skipped % similar messages)%')
    group by message_format_string
) where any_message not like '%Poco::Exception%';

drop table logs;
