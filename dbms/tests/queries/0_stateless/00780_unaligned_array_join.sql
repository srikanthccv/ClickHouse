SELECT number, arr1, arr2, x, y FROM (SELECT number, range(number % 2) AS arr1, range(number % 3) arr2 FROM system.numbers LIMIT 10) ARRAY JOIN arr1 AS x, arr2 AS y SETTINGS enable_unaligned_array_join = 1;
SELECT number, arr1, arr2, x, y FROM (SELECT number, range(number % 2) AS arr1, range(number % 3) arr2 FROM system.numbers LIMIT 10) LEFT ARRAY JOIN arr1 AS x, arr2 AS y SETTINGS enable_unaligned_array_join = 1;
