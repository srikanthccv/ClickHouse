select geoToH3(toFloat64(0),toFloat64(1),arrayJoin([1,2]));
select h3ToParent(641573946153969375, arrayJoin([1,2]));
SELECT h3HexAreaM2(arrayJoin([1,2]));
SELECT h3HexAreaKm2(arrayJoin([1,2]));
SELECT h3CellAreaM2(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3CellAreaRads2(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3GetResolution(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3EdgeAngle(arrayJoin([0,1,2]));
SELECT h3EdgeLengthM(arrayJoin([0,1,2]));
SELECT h3EdgeLengthKm(arrayJoin([0,1,2]));
SELECT h3ToGeo(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3ToGeoBoundary(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3kRing(arrayJoin([579205133326352383]), arrayJoin([toUInt16(1),toUInt16(2),toUInt16(3)]));
SELECT h3GetBaseCell(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3IndexesAreNeighbors(617420388351344639, arrayJoin([617420388352655359, 617420388351344639, 617420388352917503]));
SELECT h3ToChildren(599405990164561919, arrayJoin([6,5]));
SELECT h3ToParent(599405990164561919, arrayJoin([0,1]));
SELECT h3ToString(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT stringToH3(h3ToString(arrayJoin([579205133326352383,589753847883235327,594082350283882495])));
SELECT h3IsResClassIII(arrayJoin([579205133326352383,589753847883235327,594082350283882495]));
SELECT h3IsPentagon(arrayJoin([stringToH3('8f28308280f18f2'),stringToH3('821c07fffffffff'),stringToH3('0x8f28308280f18f2L'),stringToH3('0x821c07fffffffffL')]));
SELECT h3GetFaces(arrayJoin([stringToH3('8f28308280f18f2'),stringToH3('821c07fffffffff'),stringToH3('0x8f28308280f18f2L'),stringToH3('0x821c07fffffffffL')]));
SELECT h3ToCenterChild(577023702256844799, arrayJoin([1,2,3]));
SELECT h3ExactEdgeLengthM(arrayJoin([1298057039473278975,1370114633511206911,1442172227549134847,1514229821587062783]));
SELECT h3ExactEdgeLengthKm(arrayJoin([1298057039473278975,1370114633511206911,1442172227549134847,1514229821587062783]));
SELECT h3ExactEdgeLengthRads(arrayJoin([1298057039473278975,1370114633511206911,1442172227549134847,1514229821587062783]));
SELECT h3NumHexagons(arrayJoin([1,2,3]));
