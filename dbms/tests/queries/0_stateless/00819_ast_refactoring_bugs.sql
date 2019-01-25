DROP TABLE IF EXISTS test.visits;
CREATE TABLE test.visits
(
    Sign Int8,
    Arr Array(Int8),
    `ParsedParams.Key1` Array(String),
    `ParsedParams.Key2` Array(String),
    CounterID UInt32
) ENGINE = Memory;

SELECT arrayMap(x -> x * Sign, Arr) FROM test.visits;

SELECT PP.Key2 AS `ym:s:pl2`
FROM test.visits
ARRAY JOIN
    `ParsedParams.Key2` AS `PP.Key2`,
    `ParsedParams.Key1` AS `PP.Key1`,
    arrayEnumerateUniq(`ParsedParams.Key2`, arrayMap(x_0 -> 1, `ParsedParams.Key1`)) AS `upp_==_yes_`,
    arrayEnumerateUniq(`ParsedParams.Key2`) AS _uniq_ParsedParams
WHERE CounterID = 100500;

DROP TABLE test.visits;
