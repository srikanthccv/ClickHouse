set output_format_write_statistics=0;
select formatRow('TSVWithNamesAndTypes', number, toDate(number)) from numbers(5);
select formatRow('CSVWithNamesAndTypes', number, toDate(number)) from numbers(5);
select formatRow('JSONCompactEachRowWithNamesAndTypes', number, toDate(number)) from numbers(5);
select formatRow('XML', number, toDate(number)) from numbers(5);

