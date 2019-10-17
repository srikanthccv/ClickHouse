#pragma once

#include <Parsers/ASTIdentifier.h>
#include <Interpreters/DatabaseAndTableWithAlias.h>

namespace DB
{

struct IdentifierSemanticImpl
{
    bool special = false;       /// for now it's 'not a column': tables, subselects and some special stuff like FORMAT
    bool need_long_name = false;/// if column presents in multiple tables we need qualified names
    bool can_be_alias = true;   /// if it's a cropped name it could not be an alias
    size_t membership = 0;      /// table position in join (starting from 1) detected by qualifier or 0 if not detected.
};

/// Static calss to manipulate IdentifierSemanticImpl via ASTIdentifier
struct IdentifierSemantic
{
    enum class ColumnMatch
    {
        NoMatch,
        ColumnName,       /// table has column with same name
        TableName,        /// column qualified with table name
        DatabaseAndTable, /// column qualified with database and table name
        TableAlias,       /// column qualified with table alias
        Ambiguous,
    };

    /// @returns name for column identifiers
    static std::optional<String> getColumnName(const ASTIdentifier & node);
    static std::optional<String> getColumnName(const ASTPtr & ast);

    /// @returns name for 'not a column' identifiers
    static std::optional<String> getTableName(const ASTIdentifier & node);
    static std::optional<String> getTableName(const ASTPtr & ast);
    static std::pair<String, String> extractDatabaseAndTable(const ASTIdentifier & identifier);

    static ColumnMatch canReferColumnToTable(const ASTIdentifier & identifier, const DatabaseAndTableWithAlias & db_and_table);
    static String columnNormalName(const ASTIdentifier & identifier, const DatabaseAndTableWithAlias & db_and_table);
    static String columnLongName(const ASTIdentifier & identifier, const DatabaseAndTableWithAlias & db_and_table);
    static void setColumnNormalName(ASTIdentifier & identifier, const DatabaseAndTableWithAlias & db_and_table);
    static void setColumnLongName(ASTIdentifier & identifier, const DatabaseAndTableWithAlias & db_and_table);
    static void setNeedLongName(ASTIdentifier & identifier, bool); /// if set setColumnNormalName makes qualified name
    static bool canBeAlias(const ASTIdentifier & identifier);
    static void setMembership(ASTIdentifier & identifier, size_t table_no);
    static size_t getMembership(const ASTIdentifier & identifier);
    static bool chooseTable(const ASTIdentifier &, const std::vector<DatabaseAndTableWithAlias> & tables, size_t & best_table_pos,
                            bool ambiguous = false);
    static bool chooseTable(const ASTIdentifier &, const std::vector<TableWithColumnNames> & tables, size_t & best_table_pos,
                            bool ambiguous = false);

private:
    static bool doesIdentifierBelongTo(const ASTIdentifier & identifier, const String & database, const String & table);
    static bool doesIdentifierBelongTo(const ASTIdentifier & identifier, const String & table);
    static void setColumnShortName(ASTIdentifier & identifier, size_t match);
};

inline UInt32 value(IdentifierSemantic::ColumnMatch match)
{
    return static_cast<UInt32>(match);
}

}
