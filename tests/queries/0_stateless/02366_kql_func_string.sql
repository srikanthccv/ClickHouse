DROP TABLE IF EXISTS Customers;
CREATE TABLE Customers
(    
    FirstName Nullable(String),
    LastName String, 
    Occupation String,
    Education String,
    Age Nullable(UInt8)
) ENGINE = Memory;

INSERT INTO Customers VALUES ('Theodore','Diaz','Skilled Manual','Bachelors',28), ('Stephanie','Cox','Management abcd defg','Bachelors',33),('Peter','Nara','Skilled Manual','Graduate Degree',26),('Latoya','Shen','Professional','Graduate Degree',25),('Apple','','Skilled Manual','Bachelors',28),(NULL,'why','Professional','Partial College',38);

set dialect='kusto';
print '-- test String Functions --';

print '-- Customers |where Education contains \'degree\'';
Customers |where Education contains 'degree' | order by LastName;
print '';
print '-- Customers |where Education !contains \'degree\'';
Customers |where Education !contains 'degree' | order by LastName;
print '';
print '-- Customers |where Education contains \'Degree\'';
Customers |where Education contains 'Degree' | order by LastName;
print '';
print '-- Customers |where Education !contains \'Degree\'';
Customers |where Education !contains 'Degree' | order by LastName;
print '';
print '-- Customers | where FirstName endswith \'RE\'';
Customers | where FirstName endswith 'RE' | order by LastName;
print '';
print '-- Customers | where ! FirstName endswith \'RE\'';
Customers | where FirstName ! endswith 'RE' | order by LastName;
print '';
print '--Customers | where FirstName endswith_cs \'re\'';
Customers | where FirstName endswith_cs 're' | order by LastName;
print '';
print '-- Customers | where FirstName !endswith_cs \'re\'';
Customers | where FirstName !endswith_cs 're' | order by LastName;
print '';
print '-- Customers | where Occupation == \'Skilled Manual\'';
Customers | where Occupation == 'Skilled Manual' | order by LastName;
print '';
print '-- Customers | where Occupation != \'Skilled Manual\'';
Customers | where Occupation != 'Skilled Manual' | order by LastName;
print '';
print '-- Customers | where Occupation has \'skilled\'';
Customers | where Occupation has 'skilled' | order by LastName;
print '';
print '-- Customers | where Occupation !has \'skilled\'';
Customers | where Occupation !has 'skilled' | order by LastName;
print '';
print '-- Customers | where Occupation has \'Skilled\'';
Customers | where Occupation has 'Skilled'| order by LastName;
print '';
print '-- Customers | where Occupation !has \'Skilled\'';
Customers | where Occupation !has 'Skilled'| order by LastName;
print '';
print '-- Customers | where Occupation hasprefix_cs \'Ab\'';
Customers | where Occupation hasprefix_cs 'Ab'| order by LastName;
print '';
print '-- Customers | where Occupation !hasprefix_cs \'Ab\'';
Customers | where Occupation !hasprefix_cs 'Ab'| order by LastName;
print '';
print '-- Customers | where Occupation hasprefix_cs \'ab\'';
Customers | where Occupation hasprefix_cs 'ab'| order by LastName;
print '';
print '-- Customers | where Occupation !hasprefix_cs \'ab\'';
Customers | where Occupation !hasprefix_cs 'ab'| order by LastName;
print '';
print '-- Customers | where Occupation hassuffix \'Ent\'';
Customers | where Occupation hassuffix 'Ent'| order by LastName;
print '';
print '-- Customers | where Occupation !hassuffix \'Ent\'';
Customers | where Occupation !hassuffix 'Ent'| order by LastName;
print '';
print '-- Customers | where Occupation hassuffix \'ent\'';
Customers | where Occupation hassuffix 'ent'| order by LastName;
print '';
print '-- Customers | where Occupation hassuffix \'ent\'';
Customers | where Occupation hassuffix 'ent'| order by LastName;
print '';
print '-- Customers |where Education in (\'Bachelors\',\'High School\')';
Customers |where Education in ('Bachelors','High School')| order by LastName;
print '';
print '-- Customers | where Education !in (\'Bachelors\',\'High School\')';
Customers | where Education !in ('Bachelors','High School')| order by LastName;
print '';
print '-- Customers | where FirstName matches regex \'P.*r\'';
Customers | where FirstName matches regex 'P.*r'| order by LastName;
print '';
print '-- Customers | where FirstName startswith \'pet\'';
Customers | where FirstName startswith 'pet'| order by LastName;
print '';
print '-- Customers | where FirstName !startswith \'pet\'';
Customers | where FirstName !startswith 'pet'| order by LastName;
print '';
print '-- Customers | where FirstName startswith_cs \'pet\'';
Customers | where FirstName startswith_cs 'pet'| order by LastName;
print '';
print '-- Customers | where FirstName !startswith_cs \'pet\'';
Customers | where FirstName !startswith_cs 'pet'| order by LastName;
print '';
print '-- Customers | project base64_encode_tostring(\'Kusto1\') | take 1';
Customers | project base64_encode_tostring('Kusto1') | take 1;
print '';
print '-- Customers | project base64_decode_tostring(\'S3VzdG8x\') | take 1';
Customers | project base64_decode_tostring('S3VzdG8x') | take 1;
print '';
print '-- Customers | where isempty(LastName)';
Customers | where isempty(LastName);
print '';
print '-- Customers | where isnotempty(LastName)';
Customers | where isnotempty(LastName);
print '';
print '-- Customers | where isnotnull(FirstName)';
Customers | where isnotnull(FirstName)| order by LastName;
print '';
print '-- Customers | where isnull(FirstName)';
Customers | where isnull(FirstName)| order by LastName;
print '';
print '-- Customers | project url_decode(\'https%3A%2F%2Fwww.test.com%2Fhello%20word\') | take 1';
Customers | project url_decode('https%3A%2F%2Fwww.test.com%2Fhello%20word') | take 1;
print '';
print '-- Customers | project url_encode(\'https://www.test.com/hello word\') | take 1';
Customers | project url_encode('https://www.test.com/hello word') | take 1;
print '';
print '-- Customers | project name_abbr = strcat(substring(FirstName,0,3), \' \', substring(LastName,2))';
Customers | project name_abbr = strcat(substring(FirstName,0,3), ' ', substring(LastName,2))| order by LastName;
print '';
print '-- Customers | project name = strcat(FirstName, \' \', LastName)';
Customers | project name = strcat(FirstName, ' ', LastName)| order by LastName;
print '';
print '-- Customers | project FirstName, strlen(FirstName)';
Customers | project FirstName, strlen(FirstName)| order by LastName;
print '';
print '-- Customers | project strrep(FirstName,2,\'_\')';
Customers | project strrep(FirstName,2,'_')| order by LastName;
print '';
print '-- Customers | project toupper(FirstName)';
Customers | project toupper(FirstName)| order by LastName;
print '';
print '-- Customers | project tolower(FirstName)';
Customers | project tolower(FirstName)| order by LastName;
print '';
print '-- support subquery for in orerator (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/in-cs-operator) (subquery need to be wraped with bracket inside bracket); TODO: case-insensitive not supported yet';
Customers | where Age in ((Customers|project Age|where Age < 30)) | order by LastName;
-- Customer | where LastName in~ ("diaz", "cox")
print '';
print '-- has_all (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/has-all-operator); TODO: subquery not supported yet';
Customers | where Occupation has_all ('manual', 'skilled') | order by LastName;
print '';
print '-- has_any (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/has-anyoperator); TODO: subquery not supported yet';
Customers|where Occupation has_any ('Skilled','abcd');
print '';
print '-- countof (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/countoffunction)';
Customers | project countof('The cat sat on the mat', 'at') | take 1;
Customers | project countof('The cat sat on the mat', 'at', 'normal') | take 1;
Customers | project countof('The cat sat on the mat', '\\s.he', 'regex') | take 1;
print '';
print '-- extract ( https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/extractfunction)';
Customers | project extract('(\\b[A-Z]+\\b).+(\\b\\d+)', 0, 'The price of PINEAPPLE ice cream is 20') | take 1;
Customers | project extract('(\\b[A-Z]+\\b).+(\\b\\d+)', 1, 'The price of PINEAPPLE ice cream is 20') | take 1;
Customers | project extract('(\\b[A-Z]+\\b).+(\\b\\d+)', 2, 'The price of PINEAPPLE ice cream is 20') | take 1;
Customers | project extract('(\\b[A-Z]+\\b).+(\\b\\d+)', 3, 'The price of PINEAPPLE ice cream is 20') | take 1;
Customers | project extract('(\\b[A-Z]+\\b).+(\\b\\d+)', 2, 'The price of PINEAPPLE ice cream is 20', typeof(real)) | take 1;
print '';
print '-- extract_all (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/extractallfunction); TODO: captureGroups not supported yet';
Customers | project extract_all('(\\w)(\\w+)(\\w)','The price of PINEAPPLE ice cream is 20') | take 1;
print '';
print '-- split (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/splitfunction)';
Customers | project split('aa_bb', '_') | take 1;
Customers | project split('aaa_bbb_ccc', '_', 1) | take 1;
Customers | project split('', '_') | take 1;
Customers | project split('a__b', '_') | take 1;
Customers | project split('aabbcc', 'bb') | take 1;
print '';
print '-- strcat_delim (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/strcat-delimfunction); TODO: only support string now.';
Customers | project strcat_delim('-', '1', '2', strcat('A','b')) | take 1;
-- Customers | project strcat_delim('-', '1', '2', 'A' , 1s);
print '';
print '-- indexof (https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/indexoffunction); TODO: length and occurrence not supported yet';
Customers | project indexof('abcdefg','cde') | take 1;
Customers | project indexof('abcdefg','cde',2) | take 1;
Customers | project indexof('abcdefg','cde',6) | take 1;
print '-- base64_encode_fromguid()';
print base64_encode_fromguid('ae3133f2-6e22-49ae-b06a-16e6a9b212eb');
print '-- base64_decode_toarray()';
print base64_decode_toarray('S3VzdG8=');
print '-- base64_decode_toguid()';
print base64_decode_toguid(base64_encode_fromguid('ae3133f2-6e22-49ae-b06a-16e6a9b212eb')) == 'ae3133f2-6e22-49ae-b06a-16e6a9b212eb';
print '-- parse_url()';
print parse_url('scheme://username:password@host:1234/this/is/a/path?k1=v1&k2=v2#fragment');
print '-- parse_urlquery()';
print parse_urlquery('k1=v1&k2=v2&k3=v3');
print '-- strcmp()';
print strcmp('ABC','ABC'), strcmp('abc','ABC'), strcmp('ABC','abc'), strcmp('abcde','abc');
print '-- translate()';
print translate('krasp', 'otsku', 'spark'), translate('abc', '', 'ab'), translate('abc', 'x', 'abc');
print '-- trim()';
print trim("--", "--https://www.ibm.com--");
print '-- trim_start()';
print trim_start("https://", "https://www.ibm.com");
print trim_start("[^\w]+", strcat("-  ","Te st", "1", "// $"));
print '-- trim_end()';
print trim_end("://www.ibm.com", "https://www.ibm.com");
print '-- replace_regex';
print replace_regex(strcat('Number is ', '1'), 'is (\d+)', 'was: \1');
print '-- has_any_index()';
print has_any_index('this is an example', dynamic(['this', 'example'])), has_any_index("this is an example", dynamic(['not', 'example'])), has_any_index("this is an example", dynamic(['not', 'found'])), has_any_index("this is an example", dynamic([]));
