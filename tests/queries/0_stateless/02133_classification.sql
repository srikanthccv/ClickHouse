SELECT detectLanguage('Они сошлись. Волна и камень, Стихи и проза, лед и пламень, Не столь различны меж собой.');
SELECT detectLanguage('Sweet are the uses of adversity which, like the toad, ugly and venomous, wears yet a precious jewel in his head.');
SELECT detectLanguage('A vaincre sans peril, on triomphe sans gloire.');
SELECT detectLanguage('二兎を追う者は一兎をも得ず');
SELECT detectLanguage('有情饮水饱，无情食饭饥。');

SELECT detectCharset('Plain English');
SELECT detectLanguageUnknown('Plain English');

SELECT detectTonality('Милая кошка');
SELECT detectTonality('Злой человек');
SELECT detectTonality('Обычная прогулка по ближайшему парку');

SELECT detectProgrammingLanguage('#include <iostream>');
