-- { echoOn }

-- basic tests

-- expected output: {'age':'31','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar, age:31 team:psg,nationality:brazil') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- special (not control) characters in the middle of elements
-- expected output: {'age':'3!','name':'ney!mar','nationality':'br4z!l','t&am':'@psg'}
WITH
    extractKeyValuePairs('name:ney!mar, age:3! t&am:@psg,nationality:br4z!l') AS s_map,
        CAST(
            arrayMap(
                (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
            ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- non-standard escape characters (i.e not \n, \r, \t and etc), back-slash should be preserved
-- expected output: {'amount\\z':'$5\\h','currency':'\\$USD'}
WITH
    extractKeyValuePairs('currency:\$USD, amount\z:$5\h') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- standard escape sequences are covered by unit tests

-- simple quoting
-- expected output: {'age':'31','name':'neymar','team':'psg'}
WITH
    extractKeyValuePairs('name:"neymar", "age":31 "team":"psg"') AS s_map,
        CAST(
            arrayMap(
                (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
            ),
        'Map(String,String)'
    ) AS x
SELECT
    x;

-- empty values
-- expected output: {'age':'','name':'','nationality':''}
WITH
    extractKeyValuePairs('name:"", age: , nationality:') AS s_map,
    CAST(
        arrayMap(
            (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
        ),
        'Map(String,String)'
    ) AS x
SELECT
    x;

-- empty keys
-- empty keys are not allowed, thus empty output is expected
WITH
    extractKeyValuePairs('"":abc, :def') AS s_map,
    CAST(
        arrayMap(
            (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
        ),
        'Map(String,String)'
    ) AS x
SELECT
    x;

-- semi-colon as pair delimiter
-- expected output: {'age':'31','name':'neymar','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar;age:31;team:psg;invalid1:invalid1,invalid2:invalid2', ':', ';') AS s_map,
    CAST(
        arrayMap(
            (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
        ),
        'Map(String,String)'
    ) AS x
SELECT
    x;

-- both comma and semi-colon as pair delimiters
-- expected output: {'age':'31','last_key':'last_value','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar;age:31;team:psg;nationality:brazil,last_key:last_value', ':', ';,') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- single quote as quoting character
-- expected output: {'age':'31','last_key':'last_value','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:\'neymar\';\'age\':31;team:psg;nationality:brazil,last_key:last_value', ':', ';,', '\'') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- NO ESCAPING TESTS
-- expected output: {'age':'31','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar, age:31 team:psg,nationality:brazil', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- special (not control) characters in the middle of elements
-- expected output: {'age':'3!','name':'ney!mar','nationality':'br4z!l','t&am':'@psg'}
WITH
    extractKeyValuePairs('name:ney!mar, age:3! t&am:@psg,nationality:br4z!l', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- non-standard escape characters (i.e not \n, \r, \t and etc), it should accept everything
-- expected output: {'amount\\z':'$5\\h','currency':'\\$USD'}
WITH
    extractKeyValuePairs('currency:\$USD, amount\z:$5\h', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- standard escape sequences, it should return it as it is
-- expected output: {'key1':'header\nbody','key2':'start_of_text\tend_of_text'}
WITH
    extractKeyValuePairs('key1:header\nbody key2:start_of_text\tend_of_text', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- standard escape sequences are covered by unit tests

-- simple quoting
-- expected output: {'age':'31','name':'neymar','team':'psg'}
WITH
    extractKeyValuePairs('name:"neymar", "age":31 "team":"psg"', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- empty values
-- expected output: {'age':'','name':'','nationality':''}
WITH
    extractKeyValuePairs('name:"", age: , nationality:', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- empty keys
-- empty keys are not allowed, thus empty output is expected
WITH
    extractKeyValuePairs('"":abc, :def', ':', ', ', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- semi-colon as pair delimiter
-- expected output: {'age':'31','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar;age:31;team:psg;nationality:brazil', ':', ';', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- both comma and semi-colon as pair delimiters
-- expected output: {'age':'31','last_key':'last_value','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:neymar;age:31;team:psg;nationality:brazil,last_key:last_value', ':', ';,', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;

-- single quote as quoting character
-- expected output: {'age':'31','last_key':'last_value','name':'neymar','nationality':'brazil','team':'psg'}
WITH
    extractKeyValuePairs('name:\'neymar\';\'age\':31;team:psg;nationality:brazil,last_key:last_value', ':', ';,', '\'', '"', '0') AS s_map,
    CAST(
            arrayMap(
                    (x) -> (x, s_map[x]), arraySort(mapKeys(s_map))
                ),
            'Map(String,String)'
        ) AS x
SELECT
    x;
