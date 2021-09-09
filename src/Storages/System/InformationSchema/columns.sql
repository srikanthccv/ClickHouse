ATTACH VIEW columns
(
    `table_catalog` String,
    `table_schema` String,
    `table_name` String,
    `column_name` String,
    `ordinal_position` UInt64,
    `column_default` String,
    `is_nullable` UInt8,
    `data_type` String,
    `character_maximum_length` Nullable(UInt64),
    `character_octet_length` Nullable(UInt64),
    `numeric_precision` Nullable(UInt64),
    `numeric_precision_radix` Nullable(UInt64),
    `numeric_scale` Nullable(UInt64),
    `datetime_precision` Nullable(UInt64),
    `character_set_catalog` Nullable(String),
    `character_set_schema` Nullable(String),
    `character_set_name` Nullable(String),
    `collation_catalog` Nullable(String),
    `collation_schema` Nullable(String),
    `collation_name` Nullable(String),
    `domain_catalog` Nullable(String),
    `domain_schema` Nullable(String),
    `domain_name` Nullable(String),
    `TABLE_CATALOG` String ALIAS table_catalog,
    `TABLE_SCHEMA` String ALIAS table_schema,
    `TABLE_NAME` String ALIAS table_name,
    `COLUMN_NAME` String ALIAS column_name,
    `ORDINAL_POSITION` UInt64 ALIAS ordinal_position,
    `COLUMN_DEFAULT` String ALIAS column_default,
    `IS_NULLABLE` UInt8 ALIAS is_nullable,
    `DATA_TYPE` String ALIAS data_type,
    `CHARACTER_MAXIMUM_LENGTH` Nullable(UInt64) ALIAS character_maximum_length,
    `CHARACTER_OCTET_LENGTH` Nullable(UInt64) ALIAS character_octet_length,
    `NUMERIC_PRECISION` Nullable(UInt64) ALIAS numeric_precision,
    `NUMERIC_PRECISION_RADIX` Nullable(UInt64) ALIAS numeric_precision_radix,
    `NUMERIC_SCALE` Nullable(UInt64) ALIAS numeric_scale,
    `DATETIME_PRECISION` Nullable(UInt64) ALIAS datetime_precision,
    `CHARACTER_SET_CATALOG` Nullable(String) ALIAS character_set_catalog,
    `CHARACTER_SET_SCHEMA` Nullable(String) ALIAS character_set_schema,
    `CHARACTER_SET_NAME` Nullable(String) ALIAS character_set_name,
    `COLLATION_CATALOG` Nullable(String) ALIAS collation_catalog,
    `COLLATION_SCHEMA` Nullable(String) ALIAS collation_schema,
    `COLLATION_NAME` Nullable(String) ALIAS collation_name,
    `DOMAIN_CATALOG` Nullable(String) ALIAS domain_catalog,
    `DOMAIN_SCHEMA` Nullable(String) ALIAS domain_schema,
    `DOMAIN_NAME` Nullable(String) ALIAS domain_name
) AS
SELECT
    database AS table_catalog,
    database AS table_schema,
    table AS table_name,
    name AS column_name,
    position AS ordinal_position,
    default_expression AS column_default,
    type LIKE 'Nullable(%)' AS is_nullable,
    type AS data_type,
    character_octet_length AS character_maximum_length,
    character_octet_length,
    numeric_precision,
    numeric_precision_radix,
    numeric_scale,
    datetime_precision,
    NULL AS character_set_catalog,
    NULL AS character_set_schema,
    NULL AS character_set_name,
    NULL AS collation_catalog,
    NULL AS collation_schema,
    NULL AS collation_name,
    NULL AS domain_catalog,
    NULL AS domain_schema,
    NULL AS domain_name
FROM system.columns
