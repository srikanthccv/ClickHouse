SELECT countIf(1) FROM (SELECT NULL, exponentialTimeDecayedSum(100000002004087730000.)(value, time) OVER (Rows BETWEEN 255 FOLLOWING AND UNBOUNDED FOLLOWING), number = -2147483649, exponentialTimeDecayedSum(-1)(value, time) OVER (Rows BETWEEN UNBOUNDED PRECEDING AND 65537 PRECEDING), number AS value, number = NULL, exponentialTimeDecayedSum(65537)(value, time) OVER (Rows BETWEEN UNBOUNDED PRECEDING AND 1048577 FOLLOWING), number AS time FROM numbers(65535))
