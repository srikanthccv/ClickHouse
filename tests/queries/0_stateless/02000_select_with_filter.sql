SELECT argMax(number, number + 1) FILTER(WHERE number != 99) FROM numbers(100) ;
SELECT sum(number) FILTER(WHERE number % 2 == 0) FROM numbers(100);
SELECT sumIfOrNull(number, number % 2 == 1) FILTER(WHERE number % 2 == 0) FROM numbers(100); -- { clientError 184 }
SELECT sum(number) FILTER(WHERE) FROM numbers(100); -- { clientError 62 }
