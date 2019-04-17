import pytest
import os

from helpers.cluster import ClickHouseCluster
from dictionary import Field, Row, Dictionary, DictionaryStructure, Layout
from external_sources import SourceMySQL, SourceClickHouse, SourceFile, SourceExecutableCache, SourceExecutableHashed, SourceMongo
from external_sources import SourceHTTP, SourceHTTPS, SourceRedis

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

FIELDS = {
    "simple": [
        Field("KeyField", 'UInt64', is_key=True, default_value_for_get=9999999),
        Field("UInt8_", 'UInt8', default_value_for_get=55),
        Field("UInt16_", 'UInt16', default_value_for_get=66),
        Field("UInt32_", 'UInt32', default_value_for_get=77),
        Field("UInt64_", 'UInt64', default_value_for_get=88),
        Field("Int8_", 'Int8', default_value_for_get=-55),
        Field("Int16_", 'Int16', default_value_for_get=-66),
        Field("Int32_", 'Int32', default_value_for_get=-77),
        Field("Int64_", 'Int64', default_value_for_get=-88),
        Field("UUID_", 'UUID', default_value_for_get='550e8400-0000-0000-0000-000000000000'),
        Field("Date_", 'Date', default_value_for_get='2018-12-30'),
        Field("DateTime_", 'DateTime', default_value_for_get='2018-12-30 00:00:00'),
        Field("String_", 'String', default_value_for_get='hi'),
        Field("Float32_", 'Float32', default_value_for_get=555.11),
        Field("Float64_", 'Float64', default_value_for_get=777.11),
        Field("ParentKeyField", "UInt64", default_value_for_get=444, hierarchical=True)
    ],
    "complex": [
        Field("KeyField1", 'UInt64', is_key=True, default_value_for_get=9999999),
        Field("KeyField2", 'String', is_key=True, default_value_for_get='xxxxxxxxx'),
        Field("UInt8_", 'UInt8', default_value_for_get=55),
        Field("UInt16_", 'UInt16', default_value_for_get=66),
        Field("UInt32_", 'UInt32', default_value_for_get=77),
        Field("UInt64_", 'UInt64', default_value_for_get=88),
        Field("Int8_", 'Int8', default_value_for_get=-55),
        Field("Int16_", 'Int16', default_value_for_get=-66),
        Field("Int32_", 'Int32', default_value_for_get=-77),
        Field("Int64_", 'Int64', default_value_for_get=-88),
        Field("UUID_", 'UUID', default_value_for_get='550e8400-0000-0000-0000-000000000000'),
        Field("Date_", 'Date', default_value_for_get='2018-12-30'),
        Field("DateTime_", 'DateTime', default_value_for_get='2018-12-30 00:00:00'),
        Field("String_", 'String', default_value_for_get='hi'),
        Field("Float32_", 'Float32', default_value_for_get=555.11),
        Field("Float64_", 'Float64', default_value_for_get=777.11),
    ],
    "ranged": [
        Field("KeyField1", 'UInt64', is_key=True),
        Field("KeyField2", 'Date', is_range_key=True),
        Field("StartDate", 'Date', range_hash_type='min'),
        Field("EndDate", 'Date', range_hash_type='max'),
        Field("UInt8_", 'UInt8', default_value_for_get=55),
        Field("UInt16_", 'UInt16', default_value_for_get=66),
        Field("UInt32_", 'UInt32', default_value_for_get=77),
        Field("UInt64_", 'UInt64', default_value_for_get=88),
        Field("Int8_", 'Int8', default_value_for_get=-55),
        Field("Int16_", 'Int16', default_value_for_get=-66),
        Field("Int32_", 'Int32', default_value_for_get=-77),
        Field("Int64_", 'Int64', default_value_for_get=-88),
        Field("UUID_", 'UUID', default_value_for_get='550e8400-0000-0000-0000-000000000000'),
        Field("Date_", 'Date', default_value_for_get='2018-12-30'),
        Field("DateTime_", 'DateTime', default_value_for_get='2018-12-30 00:00:00'),
        Field("String_", 'String', default_value_for_get='hi'),
        Field("Float32_", 'Float32', default_value_for_get=555.11),
        Field("Float64_", 'Float64', default_value_for_get=777.11),
    ]

}

LAYOUTS = [
    Layout("hashed"),
    Layout("cache"),
    Layout("flat"),
    Layout("complex_key_hashed"),
    Layout("complex_key_cache"),
    Layout("range_hashed")
]

SOURCES = [
    SourceRedis("Redis", "localhost", "6380", "redis1", "6379", "", "", True),
    SourceMongo("MongoDB", "localhost", "27018", "mongo1", "27017", "root", "clickhouse", False),
    SourceMySQL("MySQL", "localhost", "3308", "mysql1", "3306", "root", "clickhouse", False),
    SourceClickHouse("RemoteClickHouse", "localhost", "9000", "clickhouse1", "9000", "default", "", False),
    SourceClickHouse("LocalClickHouse", "localhost", "9000", "node", "9000", "default", "", False),
    SourceFile("File", "localhost", "9000", "node", "9000", "", "", False),
    SourceExecutableHashed("ExecutableHashed", "localhost", "9000", "node", "9000", "", "", False),
    SourceExecutableCache("ExecutableCache", "localhost", "9000", "node", "9000", "", "", False),
    SourceHTTP("SourceHTTP", "localhost", "9000", "clickhouse1", "9000", "", "", False),
    SourceHTTPS("SourceHTTPS", "localhost", "9000", "clickhouse1", "9000", "", "", False),
]

DICTIONARIES = []

cluster = None
node = None

def setup_module(module):
    global DICTIONARIES
    global cluster
    global node

    dict_configs_path = os.path.join(SCRIPT_DIR, 'configs/dictionaries')
    for f in os.listdir(dict_configs_path):
        os.remove(os.path.join(dict_configs_path, f))

    for layout in LAYOUTS:
        for source in SOURCES:
            if source.compatible_with_layout(layout):
                structure = DictionaryStructure(layout, FIELDS[layout.layout_type], source.is_kv)
                dict_name = source.name + "_" + layout.name
                dict_path = os.path.join(dict_configs_path, dict_name + '.xml') # FIXME: single xml config for every column
                dictionary = Dictionary(dict_name, structure, source, dict_path, "table_" + dict_name)
                dictionary.generate_config()
                DICTIONARIES.append(dictionary)
            else:
                print "Source", source.name, "incompatible with layout", layout.name

    main_configs = []
    for fname in os.listdir(dict_configs_path):
        main_configs.append(os.path.join(dict_configs_path, fname))
    cluster = ClickHouseCluster(__file__, base_configs_dir=os.path.join(SCRIPT_DIR, 'configs'))
    node = cluster.add_instance('node', main_configs=main_configs, with_mysql=True, with_mongo=True, with_redis=True)
    cluster.add_instance('clickhouse1')

@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        for dictionary in DICTIONARIES:
            print "Preparing", dictionary.name
            dictionary.prepare_source(cluster)
            print "Prepared"

        yield cluster

    finally:
        cluster.shutdown()

def prepare_row(dct, fields, values):
    prepared_values = []
    for field, value in zip(fields, values):
        prepared_values.append(dct.source.prepare_value_for_type(field, value))
    return Row(fields, prepared_values)

def prepare_data(dct, fields, values_by_row):
    data = []
    for row in values_by_row:
        data.append(prepare_row(dct, fields, row))
    return data

def test_simple_dictionaries(started_cluster):
    fields = FIELDS["simple"]
    values_by_row = [
        [1, 22, 333, 4444, 55555, -6, -77,
         -888, -999, '550e8400-e29b-41d4-a716-446655440003',
         '1973-06-28', '1985-02-28 23:43:25', 'hello', 22.543, 3332154213.4, 0],
        [2, 3, 4, 5, 6, -7, -8,
         -9, -10, '550e8400-e29b-41d4-a716-446655440002',
         '1978-06-28', '1986-02-28 23:42:25', 'hello', 21.543, 3222154213.4, 1],
    ]

    simple_dicts = [d for d in DICTIONARIES if d.structure.layout.layout_type == "simple"]
    for dct in simple_dicts:
        data = prepare_data(dct, fields, values_by_row)
        dct.load_data(data)

    node.query("system reload dictionaries")

    queries_with_answers = []
    for dct in simple_dicts:
        data = prepare_data(dct, fields, values_by_row)
        for row in data:
            for field in fields:
                if not field.is_key:
                    for query in dct.get_select_get_queries(field, row):
                        queries_with_answers.append((query, row.get_value_by_name(field.name)))

                    for query in dct.get_select_has_queries(field, row):
                        queries_with_answers.append((query, 1))

                    for query in dct.get_select_get_or_default_queries(field, row):
                        queries_with_answers.append((query, field.default_value_for_get))
                if dct.is_kv:
                    break
        for query in dct.get_hierarchical_queries(data[0]):
            queries_with_answers.append((query, [1]))

        for query in dct.get_hierarchical_queries(data[1]):
            queries_with_answers.append((query, [2, 1]))

        for query in dct.get_is_in_queries(data[0], data[1]):
            queries_with_answers.append((query, 0))

        for query in dct.get_is_in_queries(data[1], data[0]):
            queries_with_answers.append((query, 1))

    for query, answer in queries_with_answers:
        print query
        if isinstance(answer, list):
            answer = str(answer).replace(' ', '')
        assert node.query(query) == str(answer) + '\n'

def test_complex_dictionaries(started_cluster):
    fields = FIELDS["complex"]
    values_by_row = [
        [1, 'world', 22, 333, 4444, 55555, -6,
         -77, -888, -999, '550e8400-e29b-41d4-a716-446655440003',
         '1973-06-28', '1985-02-28 23:43:25',
         'hello', 22.543, 3332154213.4],
        [2, 'qwerty2', 52, 2345, 6544, 9191991, -2,
         -717, -81818, -92929, '550e8400-e29b-41d4-a716-446655440007',
         '1975-09-28', '2000-02-28 23:33:24',
         'my', 255.543, 3332221.44],
    ]

    complex_dicts = [d for d in DICTIONARIES if d.structure.layout.layout_type == "complex" and not d.is_kv]
    for dct in complex_dicts:
        data = prepare_data(dct, fields, values_by_row)
        dct.load_data(data)

    node.query("system reload dictionaries")

    queries_with_answers = []
    for dct in complex_dicts:
        data = prepare_data(dct, fields, values_by_row)
        for row in data:
            for field in fields:
                if not field.is_key:
                    for query in dct.get_select_get_queries(field, row):
                        queries_with_answers.append((query, row.get_value_by_name(field.name)))

                    for query in dct.get_select_has_queries(field, row):
                        queries_with_answers.append((query, 1))

                    for query in dct.get_select_get_or_default_queries(field, row):
                        queries_with_answers.append((query, field.default_value_for_get))

    for query, answer in queries_with_answers:
        print query
        assert node.query(query) == str(answer) + '\n'

def test_ranged_dictionaries(started_cluster):
    fields = FIELDS["ranged"]
    values_by_row = [
        [1, '2019-02-10', '2019-02-01', '2019-02-28',
         22, 333, 4444, 55555, -6, -77, -888, -999,
         '550e8400-e29b-41d4-a716-446655440003',
         '1973-06-28', '1985-02-28 23:43:25', 'hello',
         22.543, 3332154213.4],
        [2, '2019-04-10', '2019-04-01', '2019-04-28',
         11, 3223, 41444, 52515, -65, -747, -8388, -9099,
         '550e8400-e29b-41d4-a716-446655440004',
         '1973-06-29', '2002-02-28 23:23:25', '!!!!',
         32.543, 3332543.4],
    ]

    ranged_dicts = [d for d in DICTIONARIES if d.structure.layout.layout_type == "ranged" and not d.is_kv]
    for dct in ranged_dicts:
        data = prepare_data(dct, fields, values_by_row)
        dct.load_data(data)

    node.query("system reload dictionaries")

    queries_with_answers = []
    for dct in ranged_dicts:
        data = prepare_data(dct, fields, values_by_row)
        for row in data:
            for field in fields:
                if not field.is_key and not field.is_range:
                    for query in dct.get_select_get_queries(field, row):
                        queries_with_answers.append((query, row.get_value_by_name(field.name)))
                if dct.is_kv:
                    break

    for query, answer in queries_with_answers:
        print query
        assert node.query(query) == str(answer) + '\n'
