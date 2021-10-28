SELECT arrayStringConcat(['Hello', 'World']);
SELECT arrayStringConcat(materialize(['Hello', 'World']));
SELECT arrayStringConcat(['Hello', 'World'], ', ');
SELECT arrayStringConcat(materialize(['Hello', 'World']), ', ');
SELECT arrayStringConcat(emptyArrayString());
SELECT arrayStringConcat(arrayMap(x -> toString(x), range(number))) FROM system.numbers LIMIT 10;
SELECT arrayStringConcat(arrayMap(x -> toString(x), range(number)), '') FROM system.numbers LIMIT 10;
SELECT arrayStringConcat(arrayMap(x -> toString(x), range(number)), ',') FROM system.numbers LIMIT 10;
SELECT arrayStringConcat(arrayMap(x -> transform(x, [0, 1, 2, 3, 4, 5, 6, 7, 8], ['yandex', 'google', 'test', '123', '', 'hello', 'world', 'goodbye', 'xyz'], ''), arrayMap(x -> x % 9, range(number))), ' ') FROM system.numbers LIMIT 20;
SELECT arrayStringConcat(arrayMap(x -> toString(x), range(number % 4))) FROM system.numbers LIMIT 10;
SELECT arrayStringConcat([Null, 'hello', Null, 'world', Null, 'xyz', 'def', Null], ';');
SELECT arrayStringConcat([Null, Null], ';');
SELECT arrayStringConcat([Null::Nullable(String), Null::Nullable(String)], ';');
SELECT arrayStringConcat(materialize([Null, 'hello', Null, 'world', Null, 'xyz', 'def', Null]), ';');
SELECT arrayStringConcat(materialize([Null, Null]), ';');
SELECT arrayStringConcat(materialize([Null::Nullable(String), Null::Nullable(String)]), ';');
