SELECT
    extractURLParameters('http://yandex.ru/?a=b&c=d'),
    extractURLParameters('http://yandex.ru/?a=b&c=d#e=f'),
    extractURLParameters('http://yandex.ru/?a&c=d#e=f'),
    extractURLParameters('http://yandex.ru/?a=b&c=d#e=f&g=h'),
    extractURLParameters('http://yandex.ru/?a=b&c=d#e'),
    extractURLParameters('http://yandex.ru/?a=b&c=d#e&g=h'),
    extractURLParameters('http://yandex.ru/?a=b&c=d#test?e=f&g=h'),
    extractURLParameters('//yandex.ru/?a=b&c=d'),
    extractURLParameters('//yandex.ru/?a=b&c=d#e=f'),
    extractURLParameters('//yandex.ru/?a&c=d#e=f'),
    extractURLParameters('//yandex.ru/?a=b&c=d#e=f&g=h'),
    extractURLParameters('//yandex.ru/?a=b&c=d#e'),
    extractURLParameters('//yandex.ru/?a=b&c=d#e&g=h'),
    extractURLParameters('//yandex.ru/?a=b&c=d#test?e=f&g=h');

SELECT
    extractURLParameterNames('http://yandex.ru/?a=b&c=d'),
    extractURLParameterNames('http://yandex.ru/?a=b&c=d#e=f'),
    extractURLParameterNames('http://yandex.ru/?a&c=d#e=f'),
    extractURLParameterNames('http://yandex.ru/?a=b&c=d#e=f&g=h'),
    extractURLParameterNames('http://yandex.ru/?a=b&c=d#e'),
    extractURLParameterNames('http://yandex.ru/?a=b&c=d#e&g=h'),
    extractURLParameterNames('http://yandex.ru/?a=b&c=d#test?e=f&g=h'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d#e=f'),
    extractURLParameterNames('//yandex.ru/?a&c=d#e=f'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d#e=f&g=h'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d#e'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d#e&g=h'),
    extractURLParameterNames('//yandex.ru/?a=b&c=d#test?e=f&g=h');

SELECT
    extractURLParameter('http://yandex.ru/?a=b&c=d', 'a'),
    extractURLParameter('http://yandex.ru/?a=b&c=d', 'c'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e=f', 'e'),
    extractURLParameter('http://yandex.ru/?a&c=d#e=f', 'a'),
    extractURLParameter('http://yandex.ru/?a&c=d#e=f', 'c'),
    extractURLParameter('http://yandex.ru/?a&c=d#e=f', 'e'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e=f&g=h', 'g'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e', 'a'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e', 'c'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e', 'e'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'c'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'e'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'g'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'test'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'e'),
    extractURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'g'),
    extractURLParameter('//yandex.ru/?a=b&c=d', 'a'),
    extractURLParameter('//yandex.ru/?a=b&c=d', 'c'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e=f', 'e'),
    extractURLParameter('//yandex.ru/?a&c=d#e=f', 'a'),
    extractURLParameter('//yandex.ru/?a&c=d#e=f', 'c'),
    extractURLParameter('//yandex.ru/?a&c=d#e=f', 'e'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e=f&g=h', 'g'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e', 'a'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e', 'c'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e', 'e'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'c'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'e'),
    extractURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'g'),
    extractURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'test'),
    extractURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'e'),
    extractURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'g');

SELECT
    cutURLParameter('http://yandex.ru/?a=b&c=d', 'a'),
    cutURLParameter('http://yandex.ru/?a=b&c=d', 'c'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e=f', 'e'),
    cutURLParameter('http://yandex.ru/?a&c=d#e=f', 'a'),
    cutURLParameter('http://yandex.ru/?a&c=d#e=f', 'c'),
    cutURLParameter('http://yandex.ru/?a&c=d#e=f', 'e'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e=f&g=h', 'g'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e', 'a'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e', 'c'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e', 'e'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'c'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'e'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#e&g=h', 'g'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'test'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'e'),
    cutURLParameter('http://yandex.ru/?a=b&c=d#test?e=f&g=h', 'g'),
    cutURLParameter('//yandex.ru/?a=b&c=d', 'a'),
    cutURLParameter('//yandex.ru/?a=b&c=d', 'c'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e=f', 'e'),
    cutURLParameter('//yandex.ru/?a&c=d#e=f', 'a'),
    cutURLParameter('//yandex.ru/?a&c=d#e=f', 'c'),
    cutURLParameter('//yandex.ru/?a&c=d#e=f', 'e'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e=f&g=h', 'g'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e', 'a'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e', 'c'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e', 'e'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'c'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'e'),
    cutURLParameter('//yandex.ru/?a=b&c=d#e&g=h', 'g'),
    cutURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'test'),
    cutURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'e'),
    cutURLParameter('//yandex.ru/?a=b&c=d#test?e=f&g=h', 'g');


SELECT
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d')),
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d#e=f')),
    extractURLParameters(materialize('http://yandex.ru/?a&c=d#e=f')),
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d#e=f&g=h')),
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d#e')),
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d#e&g=h')),
    extractURLParameters(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d#e=f')),
    extractURLParameters(materialize('//yandex.ru/?a&c=d#e=f')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d#e=f&g=h')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d#e')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d#e&g=h')),
    extractURLParameters(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'));

SELECT
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d')),
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d#e=f')),
    extractURLParameterNames(materialize('http://yandex.ru/?a&c=d#e=f')),
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d#e=f&g=h')),
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d#e')),
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d#e&g=h')),
    extractURLParameterNames(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d#e=f')),
    extractURLParameterNames(materialize('//yandex.ru/?a&c=d#e=f')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d#e=f&g=h')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d#e')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d#e&g=h')),
    extractURLParameterNames(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'));

SELECT
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d'), 'a'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d'), 'c'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e=f'), 'e'),
    extractURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'a'),
    extractURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'c'),
    extractURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'e'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e=f&g=h'), 'g'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'a'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'c'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'e'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'c'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'e'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'g'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'test'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'e'),
    extractURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'g'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d'), 'a'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d'), 'c'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e=f'), 'e'),
    extractURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'a'),
    extractURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'c'),
    extractURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'e'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e=f&g=h'), 'g'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'a'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'c'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'e'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'c'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'e'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'g'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'test'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'e'),
    extractURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'g');

SELECT
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d'), 'a'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d'), 'c'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e=f'), 'e'),
    cutURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'a'),
    cutURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'c'),
    cutURLParameter(materialize('http://yandex.ru/?a&c=d#e=f'), 'e'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e=f&g=h'), 'g'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'a'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'c'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e'), 'e'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'c'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'e'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#e&g=h'), 'g'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'test'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'e'),
    cutURLParameter(materialize('http://yandex.ru/?a=b&c=d#test?e=f&g=h'), 'g'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d'), 'a'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d'), 'c'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e=f'), 'e'),
    cutURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'a'),
    cutURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'c'),
    cutURLParameter(materialize('//yandex.ru/?a&c=d#e=f'), 'e'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e=f&g=h'), 'g'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'a'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'c'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e'), 'e'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'c'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'e'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#e&g=h'), 'g'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'test'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'e'),
    cutURLParameter(materialize('//yandex.ru/?a=b&c=d#test?e=f&g=h'), 'g');
