SELECT pointInPolygonFranklin(tuple(2.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygonFranklin(tuple(1.0,2.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygonFranklin(tuple(4.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygon(tuple(2.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygon(tuple(1.0,2.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygon(tuple(4.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygonWinding(tuple(2.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygonWinding(tuple(1.0,2.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
SELECT pointInPolygonWinding(tuple(4.0,1.0), tuple(0.0,0.0), tuple(3.0,3.0), tuple(3.0,0.0), tuple(0.0,0.0));
