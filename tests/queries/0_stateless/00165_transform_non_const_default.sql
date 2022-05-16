SELECT transform(number, [3, 5, 7], [111, 222, 333], materialize(9999)) FROM system.numbers LIMIT 10;
SELECT transform(number, [3, 5, 7], ['hello', 'world', 'abc'], materialize('')) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], ['hello', 'world', 'abc'], materialize('')) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], ['hello', 'world', 'abc'], materialize('-')) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], [111, 222, 333], materialize(0)) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], [111, 222, 333], materialize(-1)) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], [111, 222, 333], materialize(-1.1)) FROM system.numbers LIMIT 10;
SELECT transform(toString(number), ['3', '5', '7'], [111, 222.2, 333], materialize(1)) FROM system.numbers LIMIT 10;
SELECT transform(1, [2, 3], ['Meta.ua', 'Google'], materialize('Остальные')) AS title;
SELECT transform(2, [2, 3], ['Meta.ua', 'Google'], materialize('Остальные')) AS title;
SELECT transform(3, [2, 3], ['Meta.ua', 'Google'], materialize('Остальные')) AS title;
SELECT transform(4, [2, 3], ['Meta.ua', 'Google'], materialize('Остальные')) AS title;
