CREATE TABLE ps_simple_data_types
(
    i8    Int8,
    i16   Int16,
    i32   Int32,
    i64   Int64,
    i128  Int128,
    i256  Int256,
    ui8   UInt8,
    ui16  UInt16,
    ui32  UInt32,
    ui64  UInt64,
    ui128 UInt128,
    ui256 UInt256,
    f32   Float32,
    f64   Float64,
    b     Boolean
) ENGINE MergeTree ORDER BY i8;

INSERT INTO ps_simple_data_types
VALUES (127, 32767, 2147483647, 9223372036854775807, 170141183460469231731687303715884105727,
        57896044618658097711785492504343953926634992332820282019728792003956564819967,
        255, 65535, 4294967295, 18446744073709551615, 340282366920938463463374607431768211455,
        115792089237316195423570985008687907853269984665640564039457584007913129639935,
        1.234, 3.35245141223232, FALSE),
       (-128, -32768, -2147483648, -9223372036854775808, -170141183460469231731687303715884105728,
        -57896044618658097711785492504343953926634992332820282019728792003956564819968,
        120, 1234, 51234, 421342, 15324355, 41345135123432,
        -0.7968956, -0.113259, TRUE);

CREATE TABLE ps_string_types
(
    s String,
    sn Nullable(String),
    lc LowCardinality(String),
    nlc LowCardinality(Nullable(String))
) ENGINE MergeTree ORDER BY s;

INSERT INTO ps_string_types
VALUES ('foo', 'bar', 'qaz', 'qux'),
       ('42', NULL, 'test', NULL);

CREATE TABLE ps_decimal_types
(
    d32         Decimal(9, 2),
    d64         Decimal(18, 3),
    d128_native Decimal(30, 10),
    d128_text   Decimal(38, 31),
    d256        Decimal(76, 20)
) ENGINE MergeTree ORDER BY d32;

INSERT INTO ps_decimal_types
VALUES (1234567.89,
        123456789123456.789,
        12345678912345678912.1234567891,
        1234567.8912345678912345678911234567891,
        12345678912345678912345678911234567891234567891234567891.12345678911234567891),
       (-1.55, 6.03, 5, -1224124.23423, -54342.3);

CREATE TABLE ps_misc_types
(
    a Array(String),
    u UUID,
    t Tuple(Int32, String),
    m Map(String, Int32)
) ENGINE MergeTree ORDER BY u;

INSERT INTO ps_misc_types
VALUES (['foo', 'bar'], '5da5038d-788f-48c6-b510-babb41c538d3', (42, 'qaz'), {'qux': 144, 'text': 255}),
       ([], '9a0ccc06-2578-4861-8534-631c9d40f3f7', (0, ''), {});

CREATE TABLE ps_date_types
(
    d      Date,
    d32    Date32,
    dt     DateTime,
    dt64_3 DateTime64(3, 'UTC'),
    dt64_6 DateTime64(6, 'UTC'),
    dt64_9 DateTime64(9, 'UTC')
) ENGINE MergeTree ORDER BY d;

INSERT INTO ps_date_types
VALUES ('2149-06-06', '2178-04-16', '2106-02-07 06:28:15',
        '2106-02-07 06:28:15.123',
        '2106-02-07 06:28:15.123456',
        '2106-02-07 06:28:15.123456789');
