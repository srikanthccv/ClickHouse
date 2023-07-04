-- Tags: no-fasttest, no-ubsan, no-cpu-aarch64, no-upgrade-check

SET allow_experimental_annoy_index = 1;
SET allow_experimental_analyzer = 0;

SELECT '--- Negative tests ---';

DROP TABLE IF EXISTS tab;

-- must have at most 2 arguments
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy('too', 'many', 'arguments')) ENGINE = MergeTree ORDER BY id; -- { serverError INCORRECT_QUERY }

-- first argument (distance_function) must be String
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy(3)) ENGINE = MergeTree ORDER BY id; -- { serverError INCORRECT_QUERY }

-- 2nd argument (number of trees) must be UInt64
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy('L2Distance', 'not an UInt64')) ENGINE = MergeTree ORDER BY id; -- { serverError INCORRECT_QUERY }

-- must be created on single column
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index (vector, id) TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError INCORRECT_NUMBER_OF_COLUMNS }

-- reject unsupported distance functions
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy('wormholeDistance')) ENGINE = MergeTree ORDER BY id; -- { serverError INCORRECT_DATA }

-- must be created on Array/Tuple(Float32) columns
SET allow_suspicious_low_cardinality_types = 1;
CREATE TABLE tab(id Int32, vector Float32, INDEX annoy_index vector TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError ILLEGAL_COLUMN }
CREATE TABLE tab(id Int32, vector Array(Float64), INDEX annoy_index vector TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError ILLEGAL_COLUMN }
CREATE TABLE tab(id Int32, vector Tuple(Float64), INDEX annoy_index vector TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError ILLEGAL_COLUMN }
CREATE TABLE tab(id Int32, vector LowCardinality(Float32), INDEX annoy_index vector TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError ILLEGAL_COLUMN }
CREATE TABLE tab(id Int32, vector Nullable(Float32), INDEX annoy_index vector TYPE annoy()) ENGINE = MergeTree ORDER BY id; -- { serverError ILLEGAL_COLUMN }

SELECT '--- Test default GRANULARITY (should be 100 mio. for annoy)---';

CREATE TABLE tab (id Int32, vector Array(Float32), INDEX annoy_index(vector) TYPE annoy) ENGINE=MergeTree ORDER BY id;
SHOW CREATE TABLE tab;
DROP TABLE tab;

CREATE TABLE tab (id Int32, vector Array(Float32)) ENGINE=MergeTree ORDER BY id;
ALTER TABLE tab ADD INDEX annoy_index(vector) TYPE annoy;
SHOW CREATE TABLE tab;

DROP TABLE tab;

SELECT '--- Test with Array, GRANULARITY = 1, index_granularity = 5 ---';

DROP TABLE IF EXISTS tab;
CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy() GRANULARITY 1) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 5;
INSERT INTO tab VALUES (1, [0.0, 0.0, 10.0]), (2, [0.0, 0.0, 10.5]), (3, [0.0, 0.0, 9.5]), (4, [0.0, 0.0, 9.7]), (5, [0.0, 0.0, 10.2]), (6, [10.0, 0.0, 0.0]), (7, [9.5, 0.0, 0.0]), (8, [9.7, 0.0, 0.0]), (9, [10.2, 0.0, 0.0]), (10, [10.5, 0.0, 0.0]), (11, [0.0, 10.0, 0.0]), (12, [0.0, 9.5, 0.0]), (13, [0.0, 9.7, 0.0]), (14, [0.0, 10.2, 0.0]), (15, [0.0, 10.5, 0.0]);

-- rows = 15, index_granularity = 5, GRANULARITY = 1 gives 3 annoy-indexed blocks (each comprising a single granule)
-- condition 'L2Distance(vector, reference_vector) < 1.0' ensures that only one annoy-indexed block produces results --> "Granules: 1/3"

-- See (*) why commented out
-- SELECT 'WHERE type, L2Distance';
-- SELECT *
-- FROM tab
-- WHERE L2Distance(vector, [0.0, 0.0, 10.0]) < 1.0
-- LIMIT 3;

SELECT 'WHERE type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
WHERE L2Distance(vector, [0.0, 0.0, 10.0]) < 1.0
LIMIT 3;

-- See (*) why commented out
-- SELECT 'ORDER BY type, L2Distance';
-- SELECT *
-- FROM tab
-- ORDER BY L2Distance(vector, [0.0, 0.0, 10.0])
-- LIMIT 3;

SELECT 'ORDER BY type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
ORDER BY L2Distance(vector, [0.0, 0.0, 10.0])
LIMIT 3;

-- Test special cases. Corresponding special case tests are omitted from later tests.

SELECT 'Reference ARRAYs with non-matching dimension are rejected';
SELECT *
FROM tab
ORDER BY L2Distance(vector, [0.0, 0.0])
LIMIT 3; -- { serverError INCORRECT_QUERY }

SELECT 'Special case: MaximumDistance is negative';
SELECT 'WHERE type, L2Distance';
SELECT *
FROM tab
WHERE L2Distance(vector, [0.0, 0.0, 10.0]) < -1.0
LIMIT 3; -- { serverError INCORRECT_QUERY }

SELECT 'Special case: setting annoy_index_search_k_nodes';
SELECT *
FROM tab
ORDER BY L2Distance(vector, [5.3, 7.3, 2.1])
LIMIT 3
SETTINGS annoy_index_search_k_nodes=0; -- searches zero nodes --> no results

SELECT 'Special case: setting max_limit_for_ann_queries';
EXPLAIN indexes=1
SELECT *
FROM tab
ORDER BY L2Distance(vector, [5.3, 7.3, 2.1])
LIMIT 3
SETTINGS max_limit_for_ann_queries=2; -- doesn't use the ann index

DROP TABLE tab;

-- Test Tuple embeddings. Triggers different logic than Array inside MergeTreeIndexAnnoy but the same logic as Array above MergeTreeIndexAnnoy.
-- Therefore test Tuple case just once.

SELECT '--- Test with Tuple, GRANULARITY = 1, index_granularity = 5 ---';

CREATE TABLE tab(id Int32, vector Tuple(Float32, Float32, Float32), INDEX annoy_index vector TYPE annoy() GRANULARITY 1) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 5;
INSERT INTO tab VALUES (1, (0.0, 0.0, 10.0)), (2, (0.0, 0.0, 10.5)), (3, (0.0, 0.0, 9.5)), (4, (0.0, 0.0, 9.7)), (5, (0.0, 0.0, 10.2)), (6, (10.0, 0.0, 0.0)), (7, (9.5, 0.0, 0.0)), (8, (9.7, 0.0, 0.0)), (9, (10.2, 0.0, 0.0)), (10, (10.5, 0.0, 0.0)), (11, (0.0, 10.0, 0.0)), (12, (0.0, 9.5, 0.0)), (13, (0.0, 9.7, 0.0)), (14, (0.0, 10.2, 0.0)), (15, (0.0, 10.5, 0.0));

-- See (*) why commented out
-- SELECT 'WHERE type, L2Distance';
-- SELECT *
-- FROM tab
-- WHERE L2Distance(vector, (0.0, 0.0, 10.0)) < 1.0
-- LIMIT 3;

SELECT 'WHERE type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
WHERE L2Distance(vector, (0.0, 0.0, 10.0)) < 1.0
LIMIT 3;

-- See (*) why commented out
-- SELECT 'ORDER BY type, L2Distance';
-- SELECT *
-- FROM tab
-- ORDER BY L2Distance(vector, (0.0, 0.0, 10.0))
-- LIMIT 3;

SELECT 'ORDER BY type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
ORDER BY L2Distance(vector, (0.0, 0.0, 10.0))
LIMIT 3;

DROP TABLE tab;

-- Not a systematic test, just to make sure no bad things happen
SELECT '--- Test non-default metric (cosine distance) + non-default NumTrees (200) ---';

CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy('cosineDistance', 200) GRANULARITY 1) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 5;
INSERT INTO tab VALUES (1, [0.0, 0.0, 10.0]), (2, [0.0, 0.0, 10.5]), (3, [0.0, 0.0, 9.5]), (4, [0.0, 0.0, 9.7]), (5, [0.0, 0.0, 10.2]), (6, [10.0, 0.0, 0.0]), (7, [9.5, 0.0, 0.0]), (8, [9.7, 0.0, 0.0]), (9, [10.2, 0.0, 0.0]), (10, [10.5, 0.0, 0.0]), (11, [0.0, 10.0, 0.0]), (12, [0.0, 9.5, 0.0]), (13, [0.0, 9.7, 0.0]), (14, [0.0, 10.2, 0.0]), (15, [0.0, 10.5, 0.0]);

-- See (*) why commented out
-- SELECT 'WHERE type, L2Distance';
-- SELECT *
-- FROM tab
-- WHERE L2Distance(vector, [0.0, 0.0, 10.0]) < 1.0
-- LIMIT 3;

-- See (*) why commented out
-- SELECT 'ORDER BY type, L2Distance';
-- SELECT *
-- FROM tab
-- ORDER BY L2Distance(vector, [0.0, 0.0, 10.0])
-- LIMIT 3;

DROP TABLE tab;

SELECT '--- Test with Array, GRANULARITY = 2, index_granularity = 4 ---';

CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy() GRANULARITY 2) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 4;
INSERT INTO tab VALUES (1, [0.0, 0.0, 10.0, 0.0]), (2, [0.0, 0.0, 10.5, 0.0]), (3, [0.0, 0.0, 9.5, 0.0]), (4, [0.0, 0.0, 9.7, 0.0]), (5, [10.0, 0.0, 0.0, 0.0]), (6, [9.5, 0.0, 0.0, 0.0]), (7, [9.7, 0.0, 0.0, 0.0]), (8, [10.2, 0.0, 0.0, 0.0]), (9, [0.0, 10.0, 0.0, 0.0]), (10, [0.0, 9.5, 0.0, 0.0]), (11, [0.0, 9.7, 0.0, 0.0]), (12, [0.0, 9.7, 0.0, 0.0]), (13, [0.0, 0.0, 0.0, 10.3]), (14, [0.0, 0.0, 0.0, 9.5]), (15, [0.0, 0.0, 0.0, 10.0]), (16, [0.0, 0.0, 0.0, 10.5]);

-- rows = 16, index_granularity = 4, GRANULARITY = 2 gives 2 annoy-indexed blocks (each comprising two granules)
-- condition 'L2Distance(vector, reference_vector) < 1.0' ensures that only one annoy-indexed block produces results --> "Granules: 2/4"

-- See (*) why commented out
-- SELECT 'WHERE type, L2Distance';
-- SELECT *
-- FROM tab
-- WHERE L2Distance(vector, [10.0, 0.0, 10.0, 0.0]) < 5.0
-- LIMIT 3;

SELECT 'WHERE type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
WHERE L2Distance(vector, [10.0, 0.0, 10.0, 0.0]) < 5.0
LIMIT 3;

-- See (*) why commented out
-- SELECT 'ORDER BY type, L2Distance';
-- SELECT *
-- FROM tab
-- ORDER BY L2Distance(vector, [10.0, 0.0, 10.0, 0.0])
-- LIMIT 3;

SELECT 'ORDER BY type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
ORDER BY L2Distance(vector, [10.0, 0.0, 10.0, 0.0])
LIMIT 3;

DROP TABLE tab;

SELECT '--- Test with Array, GRANULARITY = 4, index_granularity = 4 ---';

CREATE TABLE tab(id Int32, vector Array(Float32), INDEX annoy_index vector TYPE annoy() GRANULARITY 4) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 4;
INSERT INTO tab VALUES (1, [0.0, 0.0, 10.0, 0.0]), (2, [0.0, 0.0, 10.5, 0.0]), (3, [0.0, 0.0, 9.5, 0.0]), (4, [0.0, 0.0, 9.7, 0.0]), (5, [10.0, 0.0, 0.0, 0.0]), (6, [9.5, 0.0, 0.0, 0.0]), (7, [9.7, 0.0, 0.0, 0.0]), (8, [10.2, 0.0, 0.0, 0.0]), (9, [0.0, 10.0, 0.0, 0.0]), (10, [0.0, 9.5, 0.0, 0.0]), (11, [0.0, 9.7, 0.0, 0.0]), (12, [0.0, 9.7, 0.0, 0.0]), (13, [0.0, 0.0, 0.0, 10.3]), (14, [0.0, 0.0, 0.0, 9.5]), (15, [0.0, 0.0, 0.0, 10.0]), (16, [0.0, 0.0, 0.0, 10.5]);

-- rows = 16, index_granularity = 4, GRANULARITY = 4 gives a single annoy-indexed block (comprising all granules)
-- no two matches happen to be located in the same granule, so with LIMIT = 3, we'll get "Granules: 2/4"

-- See (*) why commented out
-- SELECT 'WHERE type, L2Distance';
-- SELECT *
-- FROM tab
-- WHERE L2Distance(vector, [10.0, 0.0, 10.0, 0.0]) < 5.0
-- LIMIT 3;

SELECT 'WHERE type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
WHERE L2Distance(vector, [10.0, 0.0, 10.0, 0.0]) < 5.0
LIMIT 3;

-- See (*) why commented out
-- SELECT 'ORDER BY type, L2Distance';
-- SELECT *
-- FROM tab
-- ORDER BY L2Distance(vector, [10.0, 0.0, 10.0, 0.0])
-- LIMIT 3;

SELECT 'ORDER BY type, L2Distance, check that index is used';
EXPLAIN indexes=1
SELECT *
FROM tab
ORDER BY L2Distance(vector, [10.0, 0.0, 10.0, 0.0])
LIMIT 3;

DROP TABLE tab;

-- (*) Storage and search in Annoy indexes is inherently random. Tests which check for exact row matches would be unstable. Therefore,
-- comment them out.
