SELECT 1 % ( CASE WHEN 1 THEN (1 IS NOT NULL + *) ELSE NULL END );
SELECT CASE 1 WHEN FALSE THEN 1 ELSE CASE WHEN 1 THEN 1 - (CASE 1 WHEN 1 THEN 1 ELSE 1 END) END % 1 END;

SELECT 1 % if(1, dummy, NULL); -- { serverError ILLEGAL_DIVISION }
SELECT sum(multiIf(1, dummy, NULL));
SELECT sum(multiIf(1, dummy, NULL)) OVER ();
