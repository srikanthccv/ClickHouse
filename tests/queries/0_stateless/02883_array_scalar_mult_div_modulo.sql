SELECT materialize([2, 3, 5]) * materialize(7);
SELECT [2, 3, 5] * materialize(7);
SELECT materialize([2, 3, 5]) * 7;
SELECT [2, 3, 5] * 7;
SELECT [[[2, 3, 5, 5]]] * 7;
SELECT [[[2, 3, 5, 5]]] / 2;
SELECT [(1, 2), (2, 2)] * 7;
SELECT [(NULL, 2), (2, NULL)] * 7;
SELECT [(NULL, 2), (2, NULL)] / 1;
SELECT [(1., 100000000000000000000.), (NULL, 1048577)] * 7;
SELECT [CAST('2', 'UInt64'), number] * 7 from numbers(5);
CREATE TABLE my_table (values Array(Int32)) ENGINE = MergeTree() ORDER BY values;
INSERT INTO my_table (values) VALUES ([12, 3, 1]);
SELECT values * 5 FROM my_table WHERE arrayExists(x -> x > 5, values);
DROP TABLE my_table;
SELECT [6, 6, 3] % 2;
SELECT [6, 6, 3] / 2.5::Decimal(1, 1);
SELECT [1] / 'a'; -- { serverError 43 }
