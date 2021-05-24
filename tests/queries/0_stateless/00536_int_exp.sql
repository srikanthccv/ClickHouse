SELECT exp2(number) AS e2d, intExp2(number) AS e2i, toUInt64(e2d) = e2i AS e2eq, exp10(number) AS e10d, intExp10(number) AS e10i, toString(e10d) = toString(e10i) AS e10eq FROM system.numbers LIMIT 64;
