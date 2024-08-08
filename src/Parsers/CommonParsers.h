#pragma once

#include <Parsers/IParserBase.h>

#include <cassert>
#include <string_view>

namespace DB
{

#define APPLY_FOR_PARSER_KEYWORDS(MR_MACROS) \
    MR_MACROS(ADD_COLUMN, "ADD COLUMN") \
    MR_MACROS(ADD_CONSTRAINT, "ADD CONSTRAINT") \
    MR_MACROS(ADD_INDEX, "ADD INDEX") \
    MR_MACROS(ADD_PROJECTION, "ADD PROJECTION") \
    MR_MACROS(ADD_STATISTICS, "ADD STATISTICS") \
    MR_MACROS(ADD, "ADD") \
    MR_MACROS(ADMIN_OPTION_FOR, "ADMIN OPTION FOR") \
    MR_MACROS(AFTER, "AFTER") \
    MR_MACROS(ALGORITHM, "ALGORITHM") \
    MR_MACROS(ALIAS, "ALIAS") \
    MR_MACROS(ALL, "ALL") \
    MR_MACROS(ALTER_COLUMN, "ALTER COLUMN") \
    MR_MACROS(ALTER_DATABASE, "ALTER DATABASE") \
    MR_MACROS(ALTER_LIVE_VIEW, "ALTER LIVE VIEW") \
    MR_MACROS(ALTER_POLICY, "ALTER POLICY") \
    MR_MACROS(ALTER_PROFILE, "ALTER PROFILE") \
    MR_MACROS(ALTER_QUOTA, "ALTER QUOTA") \
    MR_MACROS(ALTER_ROLE, "ALTER ROLE") \
    MR_MACROS(ALTER_ROW_POLICY, "ALTER ROW POLICY") \
    MR_MACROS(ALTER_SETTINGS_PROFILE, "ALTER SETTINGS PROFILE") \
    MR_MACROS(ALTER_TABLE, "ALTER TABLE") \
    MR_MACROS(ALTER_TEMPORARY_TABLE, "ALTER TEMPORARY TABLE") \
    MR_MACROS(ALTER_USER, "ALTER USER") \
    MR_MACROS(ALTER, "ALTER") \
    MR_MACROS(AND_STDOUT, "AND STDOUT") \
    MR_MACROS(AND, "AND") \
    MR_MACROS(ANTI, "ANTI") \
    MR_MACROS(ANY, "ANY") \
    MR_MACROS(APPEND, "APPEND") \
    MR_MACROS(APPLY_DELETED_MASK, "APPLY DELETED MASK") \
    MR_MACROS(APPLY, "APPLY") \
    MR_MACROS(ARRAY_JOIN, "ARRAY JOIN") \
    MR_MACROS(AS, "AS") \
    MR_MACROS(ASC, "ASC") \
    MR_MACROS(ASCENDING, "ASCENDING") \
    MR_MACROS(ASOF, "ASOF") \
    MR_MACROS(ASSUME, "ASSUME") \
    MR_MACROS(AST, "AST") \
    MR_MACROS(ASYNC, "ASYNC") \
    MR_MACROS(ATTACH_PART, "ATTACH PART") \
    MR_MACROS(ATTACH_PARTITION, "ATTACH PARTITION") \
    MR_MACROS(ATTACH_POLICY, "ATTACH POLICY") \
    MR_MACROS(ATTACH_PROFILE, "ATTACH PROFILE") \
    MR_MACROS(ATTACH_QUOTA, "ATTACH QUOTA") \
    MR_MACROS(ATTACH_ROLE, "ATTACH ROLE") \
    MR_MACROS(ATTACH_ROW_POLICY, "ATTACH ROW POLICY") \
    MR_MACROS(ATTACH_SETTINGS_PROFILE, "ATTACH SETTINGS PROFILE") \
    MR_MACROS(ATTACH_USER, "ATTACH USER") \
    MR_MACROS(ATTACH, "ATTACH") \
    MR_MACROS(AZURE, "AZURE") \
    MR_MACROS(BACKUP, "BACKUP") \
    MR_MACROS(BAGEXPANSION, "bagexpansion") \
    MR_MACROS(BEGIN_TRANSACTION, "BEGIN TRANSACTION") \
    MR_MACROS(BETWEEN, "BETWEEN") \
    MR_MACROS(BIDIRECTIONAL, "BIDIRECTIONAL") \
    MR_MACROS(BOTH, "BOTH") \
    MR_MACROS(BY, "BY") \
    MR_MACROS(CASCADE, "CASCADE") \
    MR_MACROS(CASE, "CASE") \
    MR_MACROS(CAST, "CAST") \
    MR_MACROS(CHANGE, "CHANGE") \
    MR_MACROS(CHANGED, "CHANGED") \
    MR_MACROS(CHAR_VARYING, "CHAR VARYING") \
    MR_MACROS(CHAR, "CHAR") \
    MR_MACROS(CHARACTER_LARGE_OBJECT, "CHARACTER LARGE OBJECT") \
    MR_MACROS(CHARACTER_VARYING, "CHARACTER VARYING") \
    MR_MACROS(CHARACTER, "CHARACTER") \
    MR_MACROS(CHECK_ALL_TABLES, "CHECK ALL TABLES") \
    MR_MACROS(CHECK_TABLE, "CHECK TABLE") \
    MR_MACROS(CHECK, "CHECK") \
    MR_MACROS(CLEANUP, "CLEANUP") \
    MR_MACROS(CLEAR_COLUMN, "CLEAR COLUMN") \
    MR_MACROS(CLEAR_INDEX, "CLEAR INDEX") \
    MR_MACROS(CLEAR_PROJECTION, "CLEAR PROJECTION") \
    MR_MACROS(CLEAR_STATISTICS, "CLEAR STATISTICS") \
    MR_MACROS(CLUSTER, "CLUSTER") \
    MR_MACROS(CLUSTERS, "CLUSTERS") \
    MR_MACROS(CN, "CN") \
    MR_MACROS(CODEC, "CODEC") \
    MR_MACROS(COLLATE, "COLLATE") \
    MR_MACROS(COLUMN, "COLUMN") \
    MR_MACROS(COLUMNS, "COLUMNS") \
    MR_MACROS(COMMENT_COLUMN, "COMMENT COLUMN") \
    MR_MACROS(COMMENT, "COMMENT") \
    MR_MACROS(COMMIT, "COMMIT") \
    MR_MACROS(COMPRESSION, "COMPRESSION") \
    MR_MACROS(CONST, "CONST") \
    MR_MACROS(CONSTRAINT, "CONSTRAINT") \
    MR_MACROS(CREATE_POLICY, "CREATE POLICY") \
    MR_MACROS(CREATE_PROFILE, "CREATE PROFILE") \
    MR_MACROS(CREATE_QUOTA, "CREATE QUOTA") \
    MR_MACROS(CREATE_ROLE, "CREATE ROLE") \
    MR_MACROS(CREATE_ROW_POLICY, "CREATE ROW POLICY") \
    MR_MACROS(CREATE_SETTINGS_PROFILE, "CREATE SETTINGS PROFILE") \
    MR_MACROS(CREATE_TABLE, "CREATE TABLE") \
    MR_MACROS(CREATE_TEMPORARY_TABLE, "CREATE TEMPORARY TABLE") \
    MR_MACROS(CREATE_USER, "CREATE USER") \
    MR_MACROS(CREATE, "CREATE") \
    MR_MACROS(CROSS, "CROSS") \
    MR_MACROS(CUBE, "CUBE") \
    MR_MACROS(CURRENT_GRANTS, "CURRENT GRANTS") \
    MR_MACROS(CURRENT_QUOTA, "CURRENT QUOTA") \
    MR_MACROS(CURRENT_ROLES, "CURRENT ROLES") \
    MR_MACROS(CURRENT_ROW, "CURRENT ROW") \
    MR_MACROS(CURRENT_TRANSACTION, "CURRENT TRANSACTION") \
    MR_MACROS(CURRENTUSER, "CURRENTUSER") \
    MR_MACROS(D, "D") \
    MR_MACROS(DATA, "DATA") \
    MR_MACROS(DATA_INNER_UUID, "DATA INNER UUID") \
    MR_MACROS(DATABASE, "DATABASE") \
    MR_MACROS(DATABASES, "DATABASES") \
    MR_MACROS(DATE, "DATE") \
    MR_MACROS(DAY, "DAY") \
    MR_MACROS(DAYS, "DAYS") \
    MR_MACROS(DD, "DD") \
    MR_MACROS(DEDUPLICATE, "DEDUPLICATE") \
    MR_MACROS(DEFAULT_DATABASE, "DEFAULT DATABASE") \
    MR_MACROS(DEFAULT_ROLE, "DEFAULT ROLE") \
    MR_MACROS(DEFAULT, "DEFAULT") \
    MR_MACROS(DEFINER, "DEFINER") \
    MR_MACROS(DELETE, "DELETE") \
    MR_MACROS(DEPENDS_ON, "DEPENDS ON") \
    MR_MACROS(DESC, "DESC") \
    MR_MACROS(DESCENDING, "DESCENDING") \
    MR_MACROS(DESCRIBE, "DESCRIBE") \
    MR_MACROS(DETACH_PART, "DETACH PART") \
    MR_MACROS(DETACH_PARTITION, "DETACH PARTITION") \
    MR_MACROS(DETACH, "DETACH") \
    MR_MACROS(DICTIONARIES, "DICTIONARIES") \
    MR_MACROS(DICTIONARY, "DICTIONARY") \
    MR_MACROS(DISK, "DISK") \
    MR_MACROS(DISTINCT_ON, "DISTINCT ON") \
    MR_MACROS(DISTINCT, "DISTINCT") \
    MR_MACROS(DIV, "DIV") \
    MR_MACROS(DROP_COLUMN, "DROP COLUMN") \
    MR_MACROS(DROP_CONSTRAINT, "DROP CONSTRAINT") \
    MR_MACROS(DROP_DEFAULT, "DROP DEFAULT") \
    MR_MACROS(DROP_DETACHED_PART, "DROP DETACHED PART") \
    MR_MACROS(DROP_DETACHED_PARTITION, "DROP DETACHED PARTITION") \
    MR_MACROS(DROP_INDEX, "DROP INDEX") \
    MR_MACROS(DROP_PART, "DROP PART") \
    MR_MACROS(DROP_PARTITION, "DROP PARTITION") \
    MR_MACROS(DROP_PROJECTION, "DROP PROJECTION") \
    MR_MACROS(DROP_STATISTICS, "DROP STATISTICS") \
    MR_MACROS(DROP_TABLE, "DROP TABLE") \
    MR_MACROS(DROP_TEMPORARY_TABLE, "DROP TEMPORARY TABLE") \
    MR_MACROS(DROP, "DROP") \
    MR_MACROS(ELSE, "ELSE") \
    MR_MACROS(EMPTY_AS, "EMPTY AS") \
    MR_MACROS(EMPTY, "EMPTY") \
    MR_MACROS(ENABLED_ROLES, "ENABLED ROLES") \
    MR_MACROS(END, "END") \
    MR_MACROS(ENFORCED, "ENFORCED") \
    MR_MACROS(ENGINE, "ENGINE") \
    MR_MACROS(EPHEMERAL_SEQUENTIAL, "EPHEMERAL SEQUENTIAL") \
    MR_MACROS(EPHEMERAL, "EPHEMERAL") \
    MR_MACROS(ESTIMATE, "ESTIMATE") \
    MR_MACROS(EVENT, "EVENT") \
    MR_MACROS(EVENTS, "EVENTS") \
    MR_MACROS(EVERY, "EVERY") \
    MR_MACROS(EXCEPT_DATABASE, "EXCEPT DATABASE") \
    MR_MACROS(EXCEPT_DATABASES, "EXCEPT DATABASES") \
    MR_MACROS(EXCEPT_TABLE, "EXCEPT TABLE") \
    MR_MACROS(EXCEPT_TABLES, "EXCEPT TABLES") \
    MR_MACROS(EXCEPT, "EXCEPT") \
    MR_MACROS(EXCHANGE_DICTIONARIES, "EXCHANGE DICTIONARIES") \
    MR_MACROS(EXCHANGE_TABLES, "EXCHANGE TABLES") \
    MR_MACROS(EXISTS, "EXISTS") \
    MR_MACROS(EXPLAIN, "EXPLAIN") \
    MR_MACROS(EXPRESSION, "EXPRESSION") \
    MR_MACROS(EXTENDED, "EXTENDED") \
    MR_MACROS(EXTERNAL_DDL_FROM, "EXTERNAL DDL FROM") \
    MR_MACROS(FALSE_KEYWORD, "FALSE") /*The name differs from the value*/ \
    MR_MACROS(FETCH_PART, "FETCH PART") \
    MR_MACROS(FETCH_PARTITION, "FETCH PARTITION") \
    MR_MACROS(FETCH, "FETCH") \
    MR_MACROS(FIELDS, "FIELDS") \
    MR_MACROS(FILE, "FILE") \
    MR_MACROS(FILESYSTEM_CACHE, "FILESYSTEM CACHE") \
    MR_MACROS(FILESYSTEM_CACHES, "FILESYSTEM CACHES") \
    MR_MACROS(FILTER, "FILTER") \
    MR_MACROS(FINAL, "FINAL") \
    MR_MACROS(FIRST, "FIRST") \
    MR_MACROS(FOLLOWING, "FOLLOWING") \
    MR_MACROS(FOR, "FOR") \
    MR_MACROS(FOREIGN_KEY, "FOREIGN KEY") \
    MR_MACROS(FOREIGN, "FOREIGN") \
    MR_MACROS(FORGET_PARTITION, "FORGET PARTITION") \
    MR_MACROS(FORMAT, "FORMAT") \
    MR_MACROS(FREEZE, "FREEZE") \
    MR_MACROS(FROM_INFILE, "FROM INFILE") \
    MR_MACROS(FROM_SHARD, "FROM SHARD") \
    MR_MACROS(FROM, "FROM") \
    MR_MACROS(FULL, "FULL") \
    MR_MACROS(FULLTEXT, "FULLTEXT") \
    MR_MACROS(FUNCTION, "FUNCTION") \
    MR_MACROS(GLOBAL_IN, "GLOBAL IN") \
    MR_MACROS(GLOBAL_NOT_IN, "GLOBAL NOT IN") \
    MR_MACROS(GLOBAL, "GLOBAL") \
    MR_MACROS(GRANT_OPTION_FOR, "GRANT OPTION FOR") \
    MR_MACROS(GRANT, "GRANT") \
    MR_MACROS(GRANTEES, "GRANTEES") \
    MR_MACROS(GRANULARITY, "GRANULARITY") \
    MR_MACROS(GROUP_BY, "GROUP BY") \
    MR_MACROS(GROUPING_SETS, "GROUPING SETS") \
    MR_MACROS(GROUPS, "GROUPS") \
    MR_MACROS(H, "H") \
    MR_MACROS(HASH, "HASH") \
    MR_MACROS(HAVING, "HAVING") \
    MR_MACROS(HDFS, "HDFS") \
    MR_MACROS(HH, "HH") \
    MR_MACROS(HIERARCHICAL, "HIERARCHICAL") \
    MR_MACROS(HOST, "HOST") \
    MR_MACROS(HOUR, "HOUR") \
    MR_MACROS(HOURS, "HOURS") \
    MR_MACROS(HTTP, "HTTP") \
    MR_MACROS(ID, "ID") \
    MR_MACROS(IDENTIFIED, "IDENTIFIED") \
    MR_MACROS(IF_EMPTY, "IF EMPTY") \
    MR_MACROS(IF_EXISTS, "IF EXISTS") \
    MR_MACROS(IF_NOT_EXISTS, "IF NOT EXISTS") \
    MR_MACROS(IGNORE_NULLS, "IGNORE NULLS") \
    MR_MACROS(ILIKE, "ILIKE") \
    MR_MACROS(IN_PARTITION, "IN PARTITION") \
    MR_MACROS(IN, "IN") \
    MR_MACROS(INDEX, "INDEX") \
    MR_MACROS(INDEXES, "INDEXES") \
    MR_MACROS(INDICES, "INDICES") \
    MR_MACROS(INHERIT, "INHERIT") \
    MR_MACROS(INJECTIVE, "INJECTIVE") \
    MR_MACROS(INNER, "INNER") \
    MR_MACROS(INSERT_INTO, "INSERT INTO") \
    MR_MACROS(INTERPOLATE, "INTERPOLATE") \
    MR_MACROS(INTERSECT, "INTERSECT") \
    MR_MACROS(INTERVAL, "INTERVAL") \
    MR_MACROS(INTO_OUTFILE, "INTO OUTFILE") \
    MR_MACROS(INVISIBLE, "INVISIBLE") \
    MR_MACROS(INVOKER, "INVOKER") \
    MR_MACROS(IP, "IP") \
    MR_MACROS(IS_NOT_DISTINCT_FROM, "IS NOT DISTINCT FROM") \
    MR_MACROS(IS_NOT_NULL, "IS NOT NULL") \
    MR_MACROS(IS_NULL, "IS NULL") \
    MR_MACROS(JOIN, "JOIN") \
    MR_MACROS(JWT, "JWT") \
    MR_MACROS(KERBEROS, "KERBEROS") \
    MR_MACROS(KEY_BY, "KEY BY") \
    MR_MACROS(KEY, "KEY") \
    MR_MACROS(KEYED_BY, "KEYED BY") \
    MR_MACROS(KEYS, "KEYS") \
    MR_MACROS(KILL, "KILL") \
    MR_MACROS(KIND, "KIND") \
    MR_MACROS(LARGE_OBJECT, "LARGE OBJECT") \
    MR_MACROS(LAST, "LAST") \
    MR_MACROS(LAYOUT, "LAYOUT") \
    MR_MACROS(LDAP, "LDAP") \
    MR_MACROS(LEADING, "LEADING") \
    MR_MACROS(LEFT_ARRAY_JOIN, "LEFT ARRAY JOIN") \
    MR_MACROS(LEFT, "LEFT") \
    MR_MACROS(LESS_THAN, "LESS THAN") \
    MR_MACROS(LEVEL, "LEVEL") \
    MR_MACROS(LIFETIME, "LIFETIME") \
    MR_MACROS(LIGHTWEIGHT, "LIGHTWEIGHT") \
    MR_MACROS(LIKE, "LIKE") \
    MR_MACROS(LIMIT, "LIMIT") \
    MR_MACROS(LINEAR, "LINEAR") \
    MR_MACROS(LIST, "LIST") \
    MR_MACROS(LIVE, "LIVE") \
    MR_MACROS(LOCAL, "LOCAL") \
    MR_MACROS(M, "M") \
    MR_MACROS(MATCH, "MATCH") \
    MR_MACROS(MATERIALIZE_COLUMN, "MATERIALIZE COLUMN") \
    MR_MACROS(MATERIALIZE_INDEX, "MATERIALIZE INDEX") \
    MR_MACROS(MATERIALIZE_PROJECTION, "MATERIALIZE PROJECTION") \
    MR_MACROS(MATERIALIZE_STATISTICS, "MATERIALIZE STATISTICS") \
    MR_MACROS(MATERIALIZE_TTL, "MATERIALIZE TTL") \
    MR_MACROS(MATERIALIZE, "MATERIALIZE") \
    MR_MACROS(MATERIALIZED, "MATERIALIZED") \
    MR_MACROS(MAX, "MAX") \
    MR_MACROS(MCS, "MCS") \
    MR_MACROS(MEMORY, "MEMORY") \
    MR_MACROS(MERGES, "MERGES") \
    MR_MACROS(METRICS, "METRICS") \
    MR_MACROS(METRICS_INNER_UUID, "METRICS INNER UUID") \
    MR_MACROS(MI, "MI") \
    MR_MACROS(MICROSECOND, "MICROSECOND") \
    MR_MACROS(MICROSECONDS, "MICROSECONDS") \
    MR_MACROS(MILLISECOND, "MILLISECOND") \
    MR_MACROS(MILLISECONDS, "MILLISECONDS") \
    MR_MACROS(MIN, "MIN") \
    MR_MACROS(MINUTE, "MINUTE") \
    MR_MACROS(MINUTES, "MINUTES") \
    MR_MACROS(MM, "MM") \
    MR_MACROS(MOD, "MOD") \
    MR_MACROS(MODIFY_COLUMN, "MODIFY COLUMN") \
    MR_MACROS(MODIFY_COMMENT, "MODIFY COMMENT") \
    MR_MACROS(MODIFY_DEFINER, "MODIFY DEFINER") \
    MR_MACROS(MODIFY_ORDER_BY, "MODIFY ORDER BY") \
    MR_MACROS(MODIFY_QUERY, "MODIFY QUERY") \
    MR_MACROS(MODIFY_REFRESH, "MODIFY REFRESH") \
    MR_MACROS(MODIFY_SAMPLE_BY, "MODIFY SAMPLE BY") \
    MR_MACROS(MODIFY_STATISTICS, "MODIFY STATISTICS") \
    MR_MACROS(MODIFY_SETTING, "MODIFY SETTING") \
    MR_MACROS(MODIFY_SQL_SECURITY, "MODIFY SQL SECURITY") \
    MR_MACROS(MODIFY_TTL, "MODIFY TTL") \
    MR_MACROS(MODIFY, "MODIFY") \
    MR_MACROS(MONTH, "MONTH") \
    MR_MACROS(MONTHS, "MONTHS") \
    MR_MACROS(MOVE_PART, "MOVE PART") \
    MR_MACROS(MOVE_PARTITION, "MOVE PARTITION") \
    MR_MACROS(MOVE, "MOVE") \
    MR_MACROS(MS, "MS") \
    MR_MACROS(MUTATION, "MUTATION") \
    MR_MACROS(N, "N") \
    MR_MACROS(NAME, "NAME") \
    MR_MACROS(NAMED_COLLECTION, "NAMED COLLECTION") \
    MR_MACROS(NANOSECOND, "NANOSECOND") \
    MR_MACROS(NANOSECONDS, "NANOSECONDS") \
    MR_MACROS(NEXT, "NEXT") \
    MR_MACROS(NO_ACTION, "NO ACTION") \
    MR_MACROS(NO_DELAY, "NO DELAY") \
    MR_MACROS(NO_LIMITS, "NO LIMITS") \
    MR_MACROS(NONE, "NONE") \
    MR_MACROS(NOT_BETWEEN, "NOT BETWEEN") \
    MR_MACROS(NOT_IDENTIFIED, "NOT IDENTIFIED") \
    MR_MACROS(NOT_ILIKE, "NOT ILIKE") \
    MR_MACROS(NOT_IN, "NOT IN") \
    MR_MACROS(NOT_KEYED, "NOT KEYED") \
    MR_MACROS(NOT_LIKE, "NOT LIKE") \
    MR_MACROS(NOT_OVERRIDABLE, "NOT OVERRIDABLE") \
    MR_MACROS(NOT, "NOT") \
    MR_MACROS(NS, "NS") \
    MR_MACROS(NULL_KEYWORD, "NULL") \
    MR_MACROS(NULLS, "NULLS") \
    MR_MACROS(OFFSET, "OFFSET") \
    MR_MACROS(ON_DELETE, "ON DELETE") \
    MR_MACROS(ON_UPDATE, "ON UPDATE") \
    MR_MACROS(ON_VOLUME, "ON VOLUME") \
    MR_MACROS(ON, "ON") \
    MR_MACROS(ONLY, "ONLY") \
    MR_MACROS(OPTIMIZE_TABLE, "OPTIMIZE TABLE") \
    MR_MACROS(OR_REPLACE, "OR REPLACE") \
    MR_MACROS(OR, "OR") \
    MR_MACROS(ORDER_BY, "ORDER BY") \
    MR_MACROS(OUTER, "OUTER") \
    MR_MACROS(OVER, "OVER") \
    MR_MACROS(OVERRIDABLE, "OVERRIDABLE") \
    MR_MACROS(PART, "PART") \
    MR_MACROS(PARTIAL, "PARTIAL") \
    MR_MACROS(PARTITION_BY, "PARTITION BY") \
    MR_MACROS(PARTITION, "PARTITION") \
    MR_MACROS(PARTITIONS, "PARTITIONS") \
    MR_MACROS(PASTE, "PASTE") \
    MR_MACROS(PERIODIC_REFRESH, "PERIODIC REFRESH") \
    MR_MACROS(PERMANENTLY, "PERMANENTLY") \
    MR_MACROS(PERMISSIVE, "PERMISSIVE") \
    MR_MACROS(PERSISTENT_SEQUENTIAL, "PERSISTENT SEQUENTIAL") \
    MR_MACROS(PERSISTENT, "PERSISTENT") \
    MR_MACROS(PIPELINE, "PIPELINE") \
    MR_MACROS(PLAN, "PLAN") \
    MR_MACROS(POPULATE, "POPULATE") \
    MR_MACROS(PRECEDING, "PRECEDING") \
    MR_MACROS(PRECISION, "PRECISION") \
    MR_MACROS(PREWHERE, "PREWHERE") \
    MR_MACROS(PRIMARY_KEY, "PRIMARY KEY") \
    MR_MACROS(PRIMARY, "PRIMARY") \
    MR_MACROS(PROFILE, "PROFILE") \
    MR_MACROS(PROJECTION, "PROJECTION") \
    MR_MACROS(PROTOBUF, "Protobuf") \
    MR_MACROS(PULL, "PULL") \
    MR_MACROS(Q, "Q") \
    MR_MACROS(QQ, "QQ") \
    MR_MACROS(QUARTER, "QUARTER") \
    MR_MACROS(QUARTERS, "QUARTERS") \
    MR_MACROS(QUERY_TREE, "QUERY TREE") \
    MR_MACROS(QUERY, "QUERY") \
    MR_MACROS(QUOTA, "QUOTA") \
    MR_MACROS(RANDOMIZE_FOR, "RANDOMIZE FOR") \
    MR_MACROS(RANDOMIZED, "RANDOMIZED") \
    MR_MACROS(RANGE, "RANGE") \
    MR_MACROS(READONLY, "READONLY") \
    MR_MACROS(REALM, "REALM") \
    MR_MACROS(RECOMPRESS, "RECOMPRESS") \
    MR_MACROS(REFERENCES, "REFERENCES") \
    MR_MACROS(REFRESH, "REFRESH") \
    MR_MACROS(REGEXP, "REGEXP") \
    MR_MACROS(REMOVE_SAMPLE_BY, "REMOVE SAMPLE BY") \
    MR_MACROS(REMOVE_TTL, "REMOVE TTL") \
    MR_MACROS(REMOVE, "REMOVE") \
    MR_MACROS(RENAME_COLUMN, "RENAME COLUMN") \
    MR_MACROS(RENAME_DATABASE, "RENAME DATABASE") \
    MR_MACROS(RENAME_DICTIONARY, "RENAME DICTIONARY") \
    MR_MACROS(RENAME_TABLE, "RENAME TABLE") \
    MR_MACROS(RENAME_TO, "RENAME TO") \
    MR_MACROS(RENAME, "RENAME") \
    MR_MACROS(REPLACE_PARTITION, "REPLACE PARTITION") \
    MR_MACROS(REPLACE, "REPLACE") \
    MR_MACROS(RESET_SETTING, "RESET SETTING") \
    MR_MACROS(RESPECT_NULLS, "RESPECT NULLS") \
    MR_MACROS(RESTORE, "RESTORE") \
    MR_MACROS(RESTRICT, "RESTRICT") \
    MR_MACROS(RESTRICTIVE, "RESTRICTIVE") \
    MR_MACROS(RESUME, "RESUME") \
    MR_MACROS(REVOKE, "REVOKE") \
    MR_MACROS(RIGHT, "RIGHT") \
    MR_MACROS(ROLLBACK, "ROLLBACK") \
    MR_MACROS(ROLLUP, "ROLLUP") \
    MR_MACROS(ROW, "ROW") \
    MR_MACROS(ROWS, "ROWS") \
    MR_MACROS(S, "S") \
    MR_MACROS(S3, "S3") \
    MR_MACROS(SALT, "SALT") \
    MR_MACROS(SAMPLE_BY, "SAMPLE BY") \
    MR_MACROS(SAMPLE, "SAMPLE") \
    MR_MACROS(SAN, "SAN") \
    MR_MACROS(SCHEME, "SCHEME") \
    MR_MACROS(SECOND, "SECOND") \
    MR_MACROS(SECONDS, "SECONDS") \
    MR_MACROS(SELECT, "SELECT") \
    MR_MACROS(SEMI, "SEMI") \
    MR_MACROS(SERVER, "SERVER") \
    MR_MACROS(SET_DEFAULT_ROLE, "SET DEFAULT ROLE") \
    MR_MACROS(SET_DEFAULT, "SET DEFAULT") \
    MR_MACROS(SET_FAKE_TIME, "SET FAKE TIME") \
    MR_MACROS(SET_NULL, "SET NULL") \
    MR_MACROS(SET_ROLE_DEFAULT, "SET ROLE DEFAULT") \
    MR_MACROS(SET_ROLE, "SET ROLE") \
    MR_MACROS(SET_TRANSACTION_SNAPSHOT, "SET TRANSACTION SNAPSHOT") \
    MR_MACROS(SET, "SET") \
    MR_MACROS(SETTINGS, "SETTINGS") \
    MR_MACROS(SHOW_ACCESS, "SHOW ACCESS") \
    MR_MACROS(SHOW_CREATE, "SHOW CREATE") \
    MR_MACROS(SHOW_ENGINES, "SHOW ENGINES") \
    MR_MACROS(SHOW_FUNCTIONS, "SHOW FUNCTIONS") \
    MR_MACROS(SHOW_GRANTS, "SHOW GRANTS") \
    MR_MACROS(SHOW_PRIVILEGES, "SHOW PRIVILEGES") \
    MR_MACROS(SHOW_PROCESSLIST, "SHOW PROCESSLIST") \
    MR_MACROS(SHOW_SETTING, "SHOW SETTING") \
    MR_MACROS(SHOW, "SHOW") \
    MR_MACROS(SIGNED, "SIGNED") \
    MR_MACROS(SIMPLE, "SIMPLE") \
    MR_MACROS(SOURCE, "SOURCE") \
    MR_MACROS(SPATIAL, "SPATIAL") \
    MR_MACROS(SQL_SECURITY, "SQL SECURITY") \
    MR_MACROS(SS, "SS") \
    MR_MACROS(START_TRANSACTION, "START TRANSACTION") \
    MR_MACROS(STATISTICS, "STATISTICS") \
    MR_MACROS(STEP, "STEP") \
    MR_MACROS(STORAGE, "STORAGE") \
    MR_MACROS(STRICT, "STRICT") \
    MR_MACROS(SUBPARTITION_BY, "SUBPARTITION BY") \
    MR_MACROS(SUBPARTITION, "SUBPARTITION") \
    MR_MACROS(SUBPARTITIONS, "SUBPARTITIONS") \
    MR_MACROS(SUSPEND, "SUSPEND") \
    MR_MACROS(SYNC, "SYNC") \
    MR_MACROS(SYNTAX, "SYNTAX") \
    MR_MACROS(SYSTEM, "SYSTEM") \
    MR_MACROS(TABLE_OVERRIDE, "TABLE OVERRIDE") \
    MR_MACROS(TABLE, "TABLE") \
    MR_MACROS(TABLES, "TABLES") \
    MR_MACROS(TAGS, "TAGS") \
    MR_MACROS(TAGS_INNER_UUID, "TAGS INNER UUID") \
    MR_MACROS(TEMPORARY_TABLE, "TEMPORARY TABLE") \
    MR_MACROS(TEMPORARY, "TEMPORARY") \
    MR_MACROS(TEST, "TEST") \
    MR_MACROS(THEN, "THEN") \
    MR_MACROS(TIMESTAMP, "TIMESTAMP") \
    MR_MACROS(TO_DISK, "TO DISK") \
    MR_MACROS(TO_INNER_UUID, "TO INNER UUID") \
    MR_MACROS(TO_SHARD, "TO SHARD") \
    MR_MACROS(TO_TABLE, "TO TABLE") \
    MR_MACROS(TO_VOLUME, "TO VOLUME") \
    MR_MACROS(TO, "TO") \
    MR_MACROS(TOP, "TOP") \
    MR_MACROS(TOTALS, "TOTALS") \
    MR_MACROS(TRACKING_ONLY, "TRACKING ONLY") \
    MR_MACROS(TRAILING, "TRAILING") \
    MR_MACROS(TRANSACTION, "TRANSACTION") \
    MR_MACROS(TRIGGER, "TRIGGER") \
    MR_MACROS(TRUE_KEYWORD, "TRUE") /*The name differs from the value*/ \
    MR_MACROS(TRUNCATE, "TRUNCATE") \
    MR_MACROS(TTL, "TTL") \
    MR_MACROS(TYPE, "TYPE") \
    MR_MACROS(TYPEOF, "TYPEOF") \
    MR_MACROS(UNBOUNDED, "UNBOUNDED") \
    MR_MACROS(UNDROP, "UNDROP") \
    MR_MACROS(UNFREEZE, "UNFREEZE") \
    MR_MACROS(UNION, "UNION") \
    MR_MACROS(UNIQUE, "UNIQUE") \
    MR_MACROS(UNSET_FAKE_TIME, "UNSET FAKE TIME") \
    MR_MACROS(UNSIGNED, "UNSIGNED") \
    MR_MACROS(UPDATE, "UPDATE") \
    MR_MACROS(URL, "URL") \
    MR_MACROS(USE, "USE") \
    MR_MACROS(USING, "USING") \
    MR_MACROS(UUID, "UUID") \
    MR_MACROS(VALID_UNTIL, "VALID UNTIL") \
    MR_MACROS(VALUES, "VALUES") \
    MR_MACROS(VARYING, "VARYING") \
    MR_MACROS(VIEW, "VIEW") \
    MR_MACROS(VISIBLE, "VISIBLE") \
    MR_MACROS(WATCH, "WATCH") \
    MR_MACROS(WATERMARK, "WATERMARK") \
    MR_MACROS(WEEK, "WEEK") \
    MR_MACROS(WEEKS, "WEEKS") \
    MR_MACROS(WHEN, "WHEN") \
    MR_MACROS(WHERE, "WHERE") \
    MR_MACROS(WINDOW, "WINDOW") \
    MR_MACROS(QUALIFY, "QUALIFY") \
    MR_MACROS(WITH_ADMIN_OPTION, "WITH ADMIN OPTION") \
    MR_MACROS(WITH_CHECK, "WITH CHECK") \
    MR_MACROS(WITH_FILL, "WITH FILL") \
    MR_MACROS(WITH_GRANT_OPTION, "WITH GRANT OPTION") \
    MR_MACROS(WITH_NAME, "WITH NAME") \
    MR_MACROS(WITH_REPLACE_OPTION, "WITH REPLACE OPTION") \
    MR_MACROS(WITH_TIES, "WITH TIES") \
    MR_MACROS(WITH, "WITH") \
    MR_MACROS(RECURSIVE, "RECURSIVE") \
    MR_MACROS(WK, "WK") \
    MR_MACROS(WRITABLE, "WRITABLE") \
    MR_MACROS(WW, "WW") \
    MR_MACROS(YEAR, "YEAR") \
    MR_MACROS(YEARS, "YEARS") \
    MR_MACROS(YY, "YY") \
    MR_MACROS(YYYY, "YYYY") \
    MR_MACROS(ZKPATH, "ZKPATH") \

/// The list of keywords where underscore is intentional
#define APPLY_FOR_PARSER_KEYWORDS_WITH_UNDERSCORES(MR_MACROS) \
    MR_MACROS(ALLOWED_LATENESS, "ALLOWED_LATENESS") \
    MR_MACROS(AUTO_INCREMENT, "AUTO_INCREMENT") \
    MR_MACROS(BASE_BACKUP, "base_backup") \
    MR_MACROS(BCRYPT_HASH, "BCRYPT_HASH") \
    MR_MACROS(BCRYPT_PASSWORD, "BCRYPT_PASSWORD") \
    MR_MACROS(CHANGEABLE_IN_READONLY, "CHANGEABLE_IN_READONLY") \
    MR_MACROS(CLUSTER_HOST_IDS, "cluster_host_ids") \
    MR_MACROS(CURRENT_USER, "CURRENT_USER") \
    MR_MACROS(DOUBLE_SHA1_HASH, "DOUBLE_SHA1_HASH") \
    MR_MACROS(DOUBLE_SHA1_PASSWORD, "DOUBLE_SHA1_PASSWORD") \
    MR_MACROS(IS_OBJECT_ID, "IS_OBJECT_ID") \
    MR_MACROS(NO_PASSWORD, "NO_PASSWORD") \
    MR_MACROS(PART_MOVE_TO_SHARD, "PART_MOVE_TO_SHARD") \
    MR_MACROS(PLAINTEXT_PASSWORD, "PLAINTEXT_PASSWORD") \
    MR_MACROS(SHA256_HASH, "SHA256_HASH") \
    MR_MACROS(SHA256_PASSWORD, "SHA256_PASSWORD") \
    MR_MACROS(SQL_TSI_DAY, "SQL_TSI_DAY") \
    MR_MACROS(SQL_TSI_HOUR, "SQL_TSI_HOUR") \
    MR_MACROS(SQL_TSI_MICROSECOND, "SQL_TSI_MICROSECOND") \
    MR_MACROS(SQL_TSI_MILLISECOND, "SQL_TSI_MILLISECOND") \
    MR_MACROS(SQL_TSI_MINUTE, "SQL_TSI_MINUTE") \
    MR_MACROS(SQL_TSI_MONTH, "SQL_TSI_MONTH") \
    MR_MACROS(SQL_TSI_NANOSECOND, "SQL_TSI_NANOSECOND") \
    MR_MACROS(SQL_TSI_QUARTER, "SQL_TSI_QUARTER") \
    MR_MACROS(SQL_TSI_SECOND, "SQL_TSI_SECOND") \
    MR_MACROS(SQL_TSI_WEEK, "SQL_TSI_WEEK") \
    MR_MACROS(SQL_TSI_YEAR, "SQL_TSI_YEAR") \
    MR_MACROS(SSH_KEY, "SSH_KEY") \
    MR_MACROS(SSL_CERTIFICATE, "SSL_CERTIFICATE") \
    MR_MACROS(STRICTLY_ASCENDING, "STRICTLY_ASCENDING") \
    MR_MACROS(WITH_ITEMINDEX, "WITH_ITEMINDEX") \

enum class Keyword : size_t
{
#define DECLARE_PARSER_KEYWORD_ENUM(identifier, name) \
    identifier,

    APPLY_FOR_PARSER_KEYWORDS(DECLARE_PARSER_KEYWORD_ENUM)
    APPLY_FOR_PARSER_KEYWORDS_WITH_UNDERSCORES(DECLARE_PARSER_KEYWORD_ENUM)
#undef DECLARE_PARSER_KEYWORD_ENUM
};


std::string_view toStringView(Keyword type);

const std::vector<String> & getAllKeyWords();


/** Parse specified keyword such as SELECT or compound keyword such as ORDER BY.
  * All case insensitive. Requires word boundary.
  * For compound keywords, any whitespace characters and comments could be in the middle.
  */
/// Example: ORDER/* Hello */BY
class ParserKeyword : public IParserBase
{
private:
    std::string_view s;

    explicit ParserKeyword(std::string_view s_): s(s_) { assert(!s.empty()); }

public:
    static ParserKeyword createDeprecated(std::string_view s_)
    {
        return ParserKeyword(s_);
    }

    static std::shared_ptr<ParserKeyword> createDeprecatedPtr(std::string_view s_)
    {
        return std::shared_ptr<ParserKeyword>(new ParserKeyword(s_));
    }

    explicit ParserKeyword(Keyword keyword);

    constexpr const char * getName() const override { return s.data(); }

    Highlight highlight() const override { return Highlight::keyword; }

protected:
    bool parseImpl(Pos & pos, ASTPtr & node, Expected & expected) override;
};


class ParserToken : public IParserBase
{
private:
    TokenType token_type;
public:
    ParserToken(TokenType token_type_) : token_type(token_type_) {} /// NOLINT

protected:
    const char * getName() const override { return "token"; }

    bool parseImpl(Pos & pos, ASTPtr & /*node*/, Expected & expected) override
    {
        if (pos->type != token_type)
        {
            expected.add(pos, getTokenName(token_type));
            return false;
        }
        ++pos;
        return true;
    }
};


// Parser always returns true and do nothing.
class ParserNothing : public IParserBase
{
public:
    const char * getName() const override { return "nothing"; }

    bool parseImpl(Pos & /*pos*/, ASTPtr & /*node*/, Expected & /*expected*/) override { return true; }
};

}
