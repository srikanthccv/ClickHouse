DROP DICTIONARY IF EXISTS dict1;
CREATE DICTIONARY dict1
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 TABLE 'dict1'))
LAYOUT(DIRECT());

SELECT * FROM dict1; --{serverError 36}

DROP DICTIONARY dict1;

DROP DICTIONARY IF EXISTS dict2;
CREATE DICTIONARY 01780_db.dict2
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 DATABASE '01780_db' TABLE 'dict2'))
LAYOUT(DIRECT());

SELECT * FROM 01780_db.dict2; --{serverError 36}
DROP DICTIONARY 01780_db.dict2;

DROP TABLE IF EXISTS 01780_db.dict3_source;
CREATE TABLE 01780_db.dict3_source
(
    id UInt64,
    value String
) ENGINE = TinyLog;

INSERT INTO 01780_db.dict3_source VALUES (1, '1'), (2, '2'), (3, '3');

CREATE DICTIONARY 01780_db.dict3
(
    id UInt64,
    value String
)
PRIMARY KEY id
SOURCE(CLICKHOUSE(HOST 'localhost' PORT 9000 TABLE 'dict3_source' DATABASE '01780_db'))
LAYOUT(DIRECT());

SELECT * FROM 01780_db.dict3;

DROP DICTIONARY 01780_db.dict3;
