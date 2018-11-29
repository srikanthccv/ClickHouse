CREATE DATABASE IF NOT EXISTS test;
DROP TABLE IF EXISTS test.unicode;

CREATE TABLE test.unicode(c1 String, c2 String) ENGINE = Memory;
INSERT INTO test.unicode VALUES ('Здравствуйте', 'Этот код можно отредактировать и запустить!'),
INSERT INTO test.unicode VALUES ('你好', '这段代码是可以编辑并且能够运行的！'),
INSERT INTO test.unicode VALUES ('Hola', '¡Este código es editable y ejecutable!'),
INSERT INTO test.unicode VALUES ('Bonjour', 'Ce code est modifiable et exécutable !'),
INSERT INTO test.unicode VALUES ('Ciao', 'Questo codice è modificabile ed eseguibile!'),
INSERT INTO test.unicode VALUES ('こんにちは', 'このコードは編集して実行出来ます！'),
INSERT INTO test.unicode VALUES ('안녕하세요', '여기에서 코드를 수정하고 실행할 수 있습니다!'),
INSERT INTO test.unicode VALUES ('Cześć', 'Ten kod można edytować oraz uruchomić!'),
INSERT INTO test.unicode VALUES ('Olá', 'Este código é editável e executável!'),
INSERT INTO test.unicode VALUES ('Chào bạn', 'Bạn có thể edit và run code trực tiếp!'),
INSERT INTO test.unicode VALUES ('Hallo', 'Dieser Code kann bearbeitet und ausgeführt werden!'),
INSERT INTO test.unicode VALUES ('Hej', 'Den här koden kan redigeras och köras!'),
INSERT INTO test.unicode VALUES ('Ahoj', 'Tento kód můžete upravit a spustit');
INSERT INTO test.unicode VALUES ('Tabs \t Tabs', 'Non-first \t Tabs');
INSERT INTO test.unicode VALUES ('Control characters \x1f\x1f\x1f\x1f with zero width', 'Invalid UTF-8 which eats pending characters \xf0, or invalid by itself \x80 with zero width');
INSERT INTO test.unicode VALUES ('Russian ё and ё ', 'Zero bytes \0 \0 in middle');
SELECT * FROM test.unicode SETTINGS max_threads = 1 FORMAT PrettyNoEscapes;
SELECT 'Tabs \t Tabs', 'Long\tTitle' FORMAT PrettyNoEscapes;

SELECT '你好', '世界' FORMAT Vertical;
SELECT 'Tabs \t Tabs', 'Non-first \t Tabs' FORMAT Vertical;
SELECT 'Control characters \x1f\x1f\x1f\x1f with zero width', 'Invalid UTF-8 which eats pending characters \xf0, and invalid by itself \x80 with zero width' FORMAT Vertical;
SELECT 'Russian ё and ё', 'Zero bytes \0 \0 in middle' FORMAT Vertical;

DROP TABLE IF EXISTS test.unicode;
