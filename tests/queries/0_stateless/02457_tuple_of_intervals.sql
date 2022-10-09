EXPLAIN SYNTAX SELECT INTERVAL '-1 SECOND 2 MINUTE -3 MONTH 1 YEAR';

SELECT '-';
SELECT '2022-10-11'::Date + INTERVAL 1 DAY + INTERVAL 1 MONTH;
SELECT '2022-10-11'::Date + (INTERVAL 1 DAY, INTERVAL 1 MONTH);
SELECT '2022-10-11'::Date + INTERVAL '1 DAY 1 MONTH';

SELECT '-';
SELECT '2022-10-11'::Date + INTERVAL -1 SECOND + INTERVAL 2 MINUTE + INTERVAL -3 MONTH + INTERVAL 1 YEAR;
SELECT '2022-10-11'::Date + (INTERVAL -1 SECOND, INTERVAL 2 MINUTE, INTERVAL -3 MONTH, INTERVAL 1 YEAR);
SELECT '2022-10-11'::Date + INTERVAL '-1 SECOND 2 MINUTE -3 MONTH 1 YEAR';

SELECT '-';
SELECT '2022-10-11'::DateTime - INTERVAL 1 QUARTER - INTERVAL -3 WEEK - INTERVAL 1 YEAR - INTERVAL 1 HOUR;
SELECT '2022-10-11'::DateTime - (INTERVAL 1 QUARTER, INTERVAL -3 WEEK, INTERVAL 1 YEAR, INTERVAL 1 HOUR);
SELECT '2022-10-11'::DateTime - INTERVAL '1 QUARTER -3 WEEK 1 YEAR 1 HOUR';

SELECT '-';
SELECT '2022-10-11'::DateTime64 - INTERVAL 1 YEAR - INTERVAL 4 MONTH - INTERVAL 1 SECOND;
SELECT '2022-10-11'::DateTime64 - (INTERVAL 1 YEAR, INTERVAL 4 MONTH, INTERVAL 1 SECOND);
SELECT '2022-10-11'::DateTime64 - INTERVAL '1 YEAR 4 MONTH 1 SECOND';
