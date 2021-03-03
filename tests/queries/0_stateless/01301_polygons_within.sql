select polygonsWithinCartesian([[[(0, 0),(0, 3),(1, 2.9),(2, 2.6),(2.6, 2),(2.9, 1),(3, 0),(0, 0)]]], [[[(1., 1.),(1., 4.),(4., 4.),(4., 1.),(1., 1.)]]]);
select polygonsWithinCartesian([[[(2., 2.), (2., 3.), (3., 3.), (3., 2.)]]], [[[(1., 1.),(1., 4.),(4., 4.),(4., 1.),(1., 1.)]]]);

select polygonsWithinSpherical([[[(4.3613577, 50.8651821), (4.349556, 50.8535879), (4.3602419, 50.8435626), (4.3830299, 50.8428851), (4.3904543, 50.8564867), (4.3613148, 50.8651279)]]], [[[(4.346693, 50.858306), (4.367945, 50.852455), (4.366227, 50.840809), (4.344961, 50.833264), (4.338074, 50.848677), (4.346693, 50.858306)]]]);
select polygonsWithinSpherical([[[(4.3501568, 50.8518269), (4.3444920, 50.8439961), (4.3565941, 50.8443213), (4.3501568, 50.8518269)]]], [[[(4.3679450, 50.8524550),(4.3466930, 50.8583060),(4.3380740, 50.8486770),(4.3449610, 50.8332640),(4.3662270, 50.8408090),(4.3679450, 50.8524550)]]]);

select polygonsWithinMercator([[[(4.3613577, 50.8651821), (4.349556, 50.8535879), (4.3602419, 50.8435626), (4.3830299, 50.8428851), (4.3904543, 50.8564867), (4.3613148, 50.8651279)]]], [[[(4.346693, 50.858306), (4.367945, 50.852455), (4.366227, 50.840809), (4.344961, 50.833264), (4.338074, 50.848677), (4.346693, 50.858306)]]]);
select polygonsWithinMercator([[[(4.3501568, 50.8518269), (4.3444920, 50.8439961), (4.3565941, 50.8443213), (4.3501568, 50.8518269)]]], [[[(4.3679450, 50.8524550),(4.3466930, 50.8583060),(4.3380740, 50.8486770),(4.3449610, 50.8332640),(4.3662270, 50.8408090),(4.3679450, 50.8524550)]]]);
