SELECT quantileExactWeighted(1)(number, 9223372036854775807) FROM numbers(6);
SELECT quantileExactWeighted((NULL, NULL, NULL, NULL))((NULL), number, (NULL, 0, 255), 9223372036854775807), quantilesExactWeighted(0, 1., 0.9998999834060669, 0.00009999999747378752, 0., 0., 0.9998999834060669, 0., 0.00009999999747378752, 0.5, 1., 0.9998999834060669, 0.00009999999747378752, 0., 1., 1)(number, 9223372036854775807) FROM (SELECT (NULL, NULL, NULL, 0.9998999834060669), number FROM system.numbers LIMIT 10);
