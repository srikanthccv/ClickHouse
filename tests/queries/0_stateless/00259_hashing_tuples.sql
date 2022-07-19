-- Tags: no-fasttest

SELECT cityHash64(1, 2, '') AS x1, cityHash64((1, 2), '') AS x2, cityHash64(1, (2, '')) AS x3, cityHash64((1, 2, '')) AS x4;
SELECT cityHash64(materialize(1), 2, '') AS x1, cityHash64((materialize(1), 2), '') AS x2, cityHash64(materialize(1), (2, '')) AS x3, cityHash64((materialize(1), 2, '')) AS x4;
SELECT cityHash64(1, materialize(2), '') AS x1, cityHash64((1, materialize(2)), '') AS x2, cityHash64(1, (materialize(2), '')) AS x3, cityHash64((1, materialize(2), '')) AS x4;
SELECT cityHash64(1, 2, materialize('')) AS x1, cityHash64((1, 2), materialize('')) AS x2, cityHash64(1, (2, materialize(''))) AS x3, cityHash64((1, 2, materialize(''))) AS x4;
SELECT cityHash64(materialize(1), materialize(2), '') AS x1, cityHash64((materialize(1), materialize(2)), '') AS x2, cityHash64(materialize(1), (materialize(2), '')) AS x3, cityHash64((materialize(1), materialize(2), '')) AS x4;
SELECT cityHash64(1, materialize(2), materialize('')) AS x1, cityHash64((1, materialize(2)), materialize('')) AS x2, cityHash64(1, (materialize(2), materialize(''))) AS x3, cityHash64((1, materialize(2), materialize(''))) AS x4;
SELECT cityHash64(materialize(1), 2, materialize('')) AS x1, cityHash64((materialize(1), 2), materialize('')) AS x2, cityHash64(materialize(1), (2, materialize(''))) AS x3, cityHash64((materialize(1), 2, materialize(''))) AS x4;
SELECT cityHash64(materialize(1), materialize(2), materialize('')) AS x1, cityHash64((materialize(1), materialize(2)), materialize('')) AS x2, cityHash64(materialize(1), (materialize(2), materialize(''))) AS x3, cityHash64((materialize(1), materialize(2), materialize(''))) AS x4;
