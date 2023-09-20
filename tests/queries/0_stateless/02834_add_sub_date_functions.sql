SET session_timezone = 'UTC';

SELECT ADDDATE(materialize('2022-05-07'::Date), INTERVAL 5 MINUTE);

SELECT addDate('2022-05-07'::Date, INTERVAL 5 MINUTE);
SELECT addDate('2022-05-07'::Date32, INTERVAL 5 MINUTE);
SELECT addDate('2022-05-07'::DateTime, INTERVAL 5 MINUTE);
SELECT addDate('2022-05-07'::DateTime64, INTERVAL 5 MINUTE);

SELECT addDate('2022-05-07'::Date); -- { serverError NUMBER_OF_ARGUMENTS_DOESNT_MATCH }
SELECT addDate('2022-05-07'::Date, INTERVAL 5 MINUTE, 5);  -- { serverError NUMBER_OF_ARGUMENTS_DOESNT_MATCH }
SELECT addDate('2022-05-07'::Date, 10); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT addDate('1234', INTERVAL 5 MINUTE);  -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }

SELECT '---';

SELECT SUBDATE(materialize('2022-05-07'::Date), INTERVAL 5 MINUTE);

SELECT subDate('2022-05-07'::Date, INTERVAL 5 MINUTE);
SELECT subDate('2022-05-07'::Date32, INTERVAL 5 MINUTE);
SELECT subDate('2022-05-07'::DateTime, INTERVAL 5 MINUTE);
SELECT subDate('2022-05-07'::DateTime64, INTERVAL 5 MINUTE);

SELECT subDate('2022-05-07'::Date); -- { serverError NUMBER_OF_ARGUMENTS_DOESNT_MATCH }
SELECT subDate('2022-05-07'::Date, INTERVAL 5 MINUTE, 5);  -- { serverError NUMBER_OF_ARGUMENTS_DOESNT_MATCH }
SELECT subDate('2022-05-07'::Date, 10); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT subDate('1234', INTERVAL 5 MINUTE);  -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
