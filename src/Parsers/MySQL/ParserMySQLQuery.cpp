#include <Parsers/MySQL/ParserMySQLQuery.h>

#include <Common/setThreadName.h>
#include <Parsers/ParserDropQuery.h>
#include <Parsers/ParserRenameQuery.h>
#include <Parsers/MySQL/ASTCreateQuery.h>

#include <Parsers/parseQuery.h>
#include <Databases/MySQL/MaterializeMySQLSyncThread.h>

namespace DB
{

namespace MySQLParser
{

bool ParserMySQLQuery::parseImpl(IParser::Pos & pos, ASTPtr & node, Expected & expected)
{
    if (getThreadName() != "MySQLDBSync")
        return false;

    ParserDropQuery p_drop_query;
    ParserRenameQuery p_rename_query;
    ParserCreateQuery p_create_query;
    /// TODO: alter table

    return p_create_query.parse(pos, node, expected) || p_drop_query.parse(pos, node, expected)
        || p_rename_query.parse(pos, node, expected);
}

}

}
