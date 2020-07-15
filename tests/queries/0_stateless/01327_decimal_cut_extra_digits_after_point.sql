SELECT CAST('1.1' AS Decimal(10, 5));
SELECT CAST('1.12345' AS Decimal(10, 5));
SELECT CAST('1.123451' AS Decimal(10, 5));
SELECT CAST('1.1234511111' AS Decimal(10, 5));
SELECT CAST('1.12345111111' AS Decimal(10, 5));
SELECT CAST('1.12345111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111' AS Decimal(10, 5));
SELECT CAST('12345.1' AS Decimal(10, 5));

-- Actually our decimal can contain more than 10 digits for free.
SELECT CAST('123456789123.1' AS Decimal(10, 5));
SELECT CAST('1234567891234.1' AS Decimal(10, 5));
SELECT CAST('1234567891234.12345111' AS Decimal(10, 5));
-- But it's just Decimal64, so there is the limit.
SELECT CAST('12345678912345.1' AS Decimal(10, 5)); -- { serverError 69 }

-- The rounding may work in unexpected way: this is just integer rounding.
-- We can improve it but here is the current behaviour:
SELECT CAST('1.123455' AS Decimal(10, 5));
SELECT CAST('1.123456' AS Decimal(10, 5));
SELECT CAST('1.123445' AS Decimal(10, 5)); -- Check if suddenly banker's rounding will be implemented.

CREATE TEMPORARY TABLE test (x Decimal(10, 5));
INSERT INTO test VALUES (1.12345111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111);
SELECT * FROM test;
