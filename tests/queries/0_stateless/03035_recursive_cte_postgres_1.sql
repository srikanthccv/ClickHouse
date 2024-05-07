/**
  * Based on https://github.com/postgres/postgres/blob/master/src/test/regress/sql/with.sql, license:
  *
  * PostgreSQL Database Management System
  * (formerly known as Postgres, then as Postgres95)
  *
  * Portions Copyright (c) 1996-2024, PostgreSQL Global Development Group
  *
  * Portions Copyright (c) 1994, The Regents of the University of California
  *
  * Permission to use, copy, modify, and distribute this software and its
  * documentation for any purpose, without fee, and without a written agreement
  * is hereby granted, provided that the above copyright notice and this
  * paragraph and the following two paragraphs appear in all copies.
  *
  * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
  * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  * DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE
  * POSSIBILITY OF SUCH DAMAGE.
  *
  * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
  * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
  * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
  * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO
  *PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
  */

--
-- Tests for common table expressions (WITH query, ... SELECT ...)
--

-- { echoOn }

SET allow_experimental_analyzer = 1;

-- WITH RECURSIVE

-- sum of 1..100
WITH RECURSIVE t AS (
    SELECT 1 AS n
UNION ALL
    SELECT n+1 FROM t WHERE n < 100
)
SELECT sum(n) FROM t;

WITH RECURSIVE t AS (
    SELECT 1 AS n
UNION ALL
    SELECT n+1 FROM t WHERE n < 5
)
SELECT * FROM t;

-- This'd be an infinite loop, but outside query reads only as much as needed
WITH RECURSIVE t AS (
    SELECT 1 AS n
UNION ALL
    SELECT n+1 FROM t)
SELECT * FROM t LIMIT 10;

WITH RECURSIVE t AS (
    SELECT 'foo' AS n
UNION ALL
    SELECT n || ' bar' FROM t WHERE length(n) < 20
)
SELECT n, toTypeName(n) FROM t;

WITH RECURSIVE t AS (
    SELECT '7' AS n
UNION ALL
    SELECT n+1 FROM t WHERE n < 10
)
SELECT n, toTypeName(n) FROM t; -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }

-- Deeply nested WITH caused a list-munging problem in v13
-- Detection of cross-references and self-references
WITH RECURSIVE w1 AS
 (WITH w2 AS
  (WITH w3 AS
   (WITH w4 AS
    (WITH w5 AS
     (WITH RECURSIVE w6 AS
      (WITH w7 AS
       (WITH w8 AS
        (SELECT 1)
        SELECT * FROM w8)
       SELECT * FROM w7)
      SELECT * FROM w6)
     SELECT * FROM w5)
    SELECT * FROM w4)
   SELECT * FROM w3)
  SELECT * FROM w2)
SELECT * FROM w1;

-- { echoOff }
