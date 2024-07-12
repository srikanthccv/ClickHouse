#!/usr/bin/env bash

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

function test
{
  $CLICKHOUSE_LOCAL --allow_experimental_dynamic_type=1 --allow_experimental_variant_type=1 --output_format_binary_encode_types_in_binary_format=1 -q "select $1 as value format RowBinaryWithNamesAndTypes" | $CLICKHOUSE_LOCAL --input-format RowBinaryWithNamesAndTypes --input_format_binary_decode_types_in_binary_format=1 -q "select value, toTypeName(value) from table"
  $CLICKHOUSE_LOCAL --allow_experimental_dynamic_type=1 --allow_experimental_variant_type=1 --output_format_native_encode_types_in_binary_format=1 -q "select $1 as value format Native" | $CLICKHOUSE_LOCAL --input-format Native --input_format_native_decode_types_in_binary_format=1 -q "select value, toTypeName(value) from table"
}

test "materialize(42)::UInt8"
test "NULL"
test "materialize(42)::UInt8"
test "materialize(-42)::Int8"
test "materialize(42)::UInt16"
test "materialize(-42)::Int16"
test "materialize(42)::UInt32"
test "materialize(-42)::Int32"
test "materialize(42)::UInt64"
test "materialize(-42)::Int64"
test "materialize(42)::UInt128"
test "materialize(-42)::Int128"
test "materialize(42)::UInt256"
test "materialize(-42)::Int256"
test "materialize(42.42)::Float32"
test "materialize(42.42)::Float64"
test "materialize('2020-01-01')::Date"
test "materialize('2020-01-01')::Date32"
test "materialize('2020-01-01 00:00:00')::DateTime"
test "materialize('2020-01-01 00:00:00')::DateTime('EST')"
test "materialize('2020-01-01 00:00:00')::DateTime('CET')"
test "materialize('2020-01-01 00:00:00.000000')::DateTime64(6)"
test "materialize('2020-01-01 00:00:00.000000')::DateTime64(6, 'EST')"
test "materialize('2020-01-01 00:00:00.000000')::DateTime64(6, 'CET')"
test "materialize('Hello, World!')"
test "materialize('aaaaa')::FixedString(5)"
test "materialize('a')::Enum8('a' = 1, 'b' = 2, 'c' = -128)"
test "materialize('a')::Enum16('a' = 1, 'b' = 2, 'c' = -1280)"
test "materialize(42.42)::Decimal32(3)"
test "materialize(42.42)::Decimal64(3)"
test "materialize(42.42)::Decimal128(3)"
test "materialize(42.42)::Decimal256(3)"
test "materialize('984ac60f-4d08-4ef1-9c62-d82f343fbc90')::UUID"
test "materialize([1, 2, 3])::Array(UInt64)"
test "materialize([[[1], [2]], [[3, 4, 5]]])::Array(Array(Array(UInt64)))"
test "materialize(tuple(1, 'str', 42.42))::Tuple(UInt32, String, Float32)"
test "materialize(tuple(1, 'str', 42.42))::Tuple(a UInt32, b String, c Float32)"
test "materialize(tuple(1, tuple('str', tuple(42.42, -30))))::Tuple(UInt32, Tuple(String, Tuple(Float32, Int8)))"
test "materialize(tuple(1, tuple('str', tuple(42.42, -30))))::Tuple(a UInt32, b Tuple(c String, d Tuple(e Float32, f Int8)))"
test "quantileState(0.5)(42::UInt64)"
test "sumSimpleState(42::UInt64)"
test "toLowCardinality('Hello, World!')"
test "materialize(map(1, 'str1', 2, 'str2'))::Map(UInt64, String)"
test "materialize(map(1, map(1, map(1, 'str1')), 2, map(2, map(2, 'str2'))))::Map(UInt64, Map(UInt64, Map(UInt64, String)))"
test "materialize('127.0.0.0')::IPv4"
test "materialize('2001:db8:cafe:1:0:0:0:1')::IPv6"
test "materialize(true)::Bool"
test "materialize([tuple(1, 2), tuple(3, 4)])::Nested(a UInt32, b UInt32)"
test "materialize([(0, 0), (10, 0), (10, 10), (0, 10)])::Ring"
test "materialize((0, 0))::Point"
test "materialize([[(20, 20), (50, 20), (50, 50), (20, 50)], [(30, 30), (50, 50), (50, 30)]])::Polygon"
test "materialize([[[(0, 0), (10, 0), (10, 10), (0, 10)]], [[(20, 20), (50, 20), (50, 50), (20, 50)],[(30, 30), (50, 50), (50, 30)]]])::MultiPolygon"
test "materialize([map(42, tuple(1, [tuple(2, map(1, 2))]))])"
test "materialize(42::UInt32)::Variant(UInt32, String, Tuple(a UInt32, b Array(Map(String, String))))"
test "materialize([map(42, tuple(1, [tuple(2, map(1, 2))]))])::Dynamic"
test "materialize([map(42, tuple(1, [tuple(2, map(1, 2))]))])::Dynamic(max_types=10)"
test "materialize([map(42, tuple(1, [tuple(2, map(1, 2))]))])::Dynamic(max_types=255)"
