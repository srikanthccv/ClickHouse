#pragma once

#include <Interpreters/Context.h>

namespace DB
{

class IAST;
using ASTPtr = std::shared_ptr<IAST>;

class ASTIdentifier;
struct ASTTableExpression;


struct DatabaseAndTableWithAlias
{
    String database;
    String table;
    String alias;

    /// "alias." or "database.table." if alias is empty
    String getQualifiedNamePrefix() const;

    /// If ast is ASTIdentifier, prepend getQualifiedNamePrefix() to it's name.
    void makeQualifiedName(const ASTPtr & ast) const;
};

void stripIdentifier(DB::ASTPtr & ast, size_t num_qualifiers_to_strip);

DatabaseAndTableWithAlias getTableNameWithAliasFromTableExpression(const ASTTableExpression & table_expression,
                                                                          const Context & context);

size_t getNumComponentsToStripInOrderToTranslateQualifiedName(const ASTIdentifier & identifier,
                                                              const DatabaseAndTableWithAlias & names);

std::pair<String, String> getDatabaseAndTableNameFromIdentifier(const ASTIdentifier & identifier);

}
