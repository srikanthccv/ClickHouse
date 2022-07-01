DROP TABLE IF EXISTS alter_00147;

set allow_deprecated_syntax_for_merge_tree=1;
CREATE TABLE alter_00147 (d Date DEFAULT toDate('2015-01-01'), n Nested(x String)) ENGINE = MergeTree(d, d, 8192);

INSERT INTO alter_00147 (`n.x`) VALUES (['Hello', 'World']);

SELECT * FROM alter_00147;
SELECT * FROM alter_00147 ARRAY JOIN n;
SELECT * FROM alter_00147 ARRAY JOIN n WHERE n.x LIKE '%Hello%';

ALTER TABLE alter_00147 ADD COLUMN n.y Array(UInt64);

SELECT * FROM alter_00147;
SELECT * FROM alter_00147 ARRAY JOIN n;
SELECT * FROM alter_00147 ARRAY JOIN n WHERE n.x LIKE '%Hello%';

INSERT INTO alter_00147 (`n.x`) VALUES (['Hello2', 'World2']);

SELECT * FROM alter_00147 ORDER BY n.x;
SELECT * FROM alter_00147 ARRAY JOIN n ORDER BY n.x;
SELECT * FROM alter_00147 ARRAY JOIN n WHERE n.x LIKE '%Hello%' ORDER BY n.x;

OPTIMIZE TABLE alter_00147;

SELECT * FROM alter_00147 ORDER BY n.x;
SELECT * FROM alter_00147 ARRAY JOIN n ORDER BY n.x;
SELECT * FROM alter_00147 ARRAY JOIN n WHERE n.x LIKE '%Hello%' ORDER BY n.x;

DROP TABLE alter_00147;
