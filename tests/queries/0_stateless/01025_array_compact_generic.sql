SELECT arrayCompact([]);
SELECT arrayCompact([1, 1, nan, nan, 2, 2, 2]);
SELECT arrayCompact([1, 1, nan, nan, -nan, 2, 2, 2]);
SELECT arrayCompact([1, 1, NULL, NULL, 2, 2, 2]);
SELECT arrayCompact([1, 1, NULL, NULL, nan, nan, 2, 2, 2]);
SELECT arrayCompact(['hello', '', '', '', 'world', 'world']);
SELECT arrayCompact([[[]], [[], []], [[], []], [[]]]);
SELECT arrayCompact(arrayMap(x -> toString(intDiv(x, 3)), range(number))) FROM numbers(10);
SELECT arrayCompact(x -> x.2, groupArray((number, intDiv(number, 3) % 3))) FROM numbers(10);
SELECT arrayCompact(x -> x.2, groupArray((toString(number), toString(intDiv(number, 3) % 3)))) FROM numbers(10);
SELECT arrayCompact(x -> x.2, groupArray((toString(number), intDiv(number, 3) % 3))) FROM numbers(10);
