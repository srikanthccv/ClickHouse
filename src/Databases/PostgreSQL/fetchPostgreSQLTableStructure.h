#pragma once

#if !defined(ARCADIA_BUILD)
#include "config_core.h"
#endif

#if USE_LIBPQXX
#include <Core/PostgreSQL/PostgreSQLConnection.h>
#include <Core/NamesAndTypes.h>


namespace DB
{

struct PostgreSQLTableStructure
{
    std::shared_ptr<NamesAndTypesList> columns;
    std::shared_ptr<NamesAndTypesList> primary_key_columns;
    std::shared_ptr<NamesAndTypesList> replica_identity_columns;
};

using PostgreSQLTableStructurePtr = std::unique_ptr<PostgreSQLTableStructure>;

std::unordered_set<std::string> fetchPostgreSQLTablesList(pqxx::connection & connection);

PostgreSQLTableStructure fetchPostgreSQLTableStructure(
    pqxx::connection & connection, const String & postgres_table_name, bool use_nulls);

template<typename T>
PostgreSQLTableStructure fetchPostgreSQLTableStructure(
    T & tx, const String & postgres_table_name, bool use_nulls,
    bool with_primary_key = false, bool with_replica_identity_index = false);

template<typename T>
std::unordered_set<std::string> fetchPostgreSQLTablesList(T & tx);

}

#endif
