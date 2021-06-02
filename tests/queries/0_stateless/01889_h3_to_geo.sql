DROP TABLE IF EXISTS h3_indexes;

CREATE TABLE h3_indexes (h3_index UInt64) ENGINE = Memory;

INSERT INTO h3_indexes VALUES(644325524701193974);
INSERT INTO h3_indexes VALUES(644325524701193975);
INSERT INTO h3_indexes VALUES(644325524701193976);
INSERT INTO h3_indexes VALUES(644325524701193977);

select h3ToGeo(h3_index) from h3_indexes;

DROP TABLE h3_indexes;