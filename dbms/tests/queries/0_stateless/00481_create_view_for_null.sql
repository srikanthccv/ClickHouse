DROP TABLE IF EXISTS null;
DROP TABLE IF EXISTS null_view;

CREATE TABLE null (x UInt8) ENGINE = Null;
CREATE VIEW null_view AS SELECT * FROM null;
INSERT INTO null VALUES (1);

SELECT * FROM null;
SELECT * FROM null_view;

DROP TABLE null;
DROP TABLE null_view;
