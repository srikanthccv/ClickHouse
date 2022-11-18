#include <Parsers/MySQL/ASTDropQuery.h>

#include <Parsers/ASTIdentifier.h>
#include <Parsers/CommonParsers.h>
#include <Parsers/ExpressionElementParsers.h>
#include <Parsers/parseDatabaseAndTableName.h>
#include <Parsers/ExpressionListParsers.h>

namespace DB
{

namespace MySQLParser
{

ASTPtr ASTDropQuery::clone() const
{
    auto res = std::make_shared<ASTDropQuery>(*this);
    res->children.clear();
    if (database)
        res->database = database;
    if (index)
        res->index = index;
    res->is_temporary = is_temporary;
    res->is_drop = is_drop;
    res->if_exists = if_exists;
    return res;
}

bool ParserDropQuery::parseImpl(IParser::Pos & pos, ASTPtr & node, Expected & expected)
{
    ParserKeyword s_drop("DROP");
    ParserKeyword s_truncate("TRUNCATE");
    ParserKeyword s_table("TABLE");
    ParserKeyword s_temporary("TEMPORARY");
    ParserKeyword s_database("DATABASE");
    ParserKeyword s_if_exists("IF EXISTS");
    ParserKeyword s_view("VIEW");
    ParserKeyword on("ON");
    ParserIdentifier name_p(true);

    ParserKeyword s_event("EVENT");
    ParserKeyword s_function("FUNCTION");
    ParserKeyword s_index("INDEX");
    ParserKeyword s_server("SERVER");
    ParserKeyword s_trigger("TRIGGER");

    auto query = std::make_shared<ASTDropQuery>();
    node = query;

    ASTPtr database;
    ASTPtr index;
    ASTDropQuery::QualifiedNames names;
    bool if_exists = false;
    bool is_drop = true;
    bool is_temporary = false;

    if (s_truncate.ignore(pos, expected) && s_table.ignore(pos, expected))
    {   
        is_drop = false;
        query->kind = ASTDropQuery::Kind::Table;
        ASTDropQuery::QualifiedName name;
        if (parseDatabaseAndTableName(pos, expected, name.schema, name.shortName))
            names.push_back(name);
        else
            return false;
    } 
    else if (s_drop.ignore(pos, expected))
    {
        if (s_database.ignore(pos, expected))
        {
            if (s_if_exists.ignore(pos, expected))
                if_exists = true;

            if (!name_p.parse(pos, database, expected))
                return false;
        }
        else
        {
            if (s_view.ignore(pos, expected))
                query->kind = ASTDropQuery::Kind::View;
            else if (s_temporary.ignore(pos, expected) && s_table.ignore(pos, expected))
            {
                is_temporary = true;
                query->kind = ASTDropQuery::Kind::Table;
            }
            else if (s_table.ignore(pos, expected))
                query->kind = ASTDropQuery::Kind::Table;
            else if (s_index.ignore(pos, expected))
            {
                query->kind = ASTDropQuery::Kind::Index;
                if (!(name_p.parse(pos, index, expected) && on.ignore(pos, expected)))
                    return false;
            }
            else if (s_event.ignore(pos, expected) || s_function.ignore(pos, expected) || s_server.ignore(pos, expected) 
                || s_trigger.ignore(pos, expected))
            {
                query->kind = ASTDropQuery::Kind::Other;
            }
            else
                return false;

            if (s_if_exists.ignore(pos, expected))
                if_exists = true;
            //parse name
            auto parse_element = [&]
            {
                ASTDropQuery::QualifiedName element;
                if (parseDatabaseAndTableName(pos, expected, element.schema, element.shortName))
                {
                    names.emplace_back(std::move(element));
                    return true;
                }
                return false;
            };

            if (!ParserList::parseUtil(pos, expected, parse_element, false))
                return false;
        }
    } 
    else 
        return false;

    query->if_exists = if_exists;
    query->database = database;
    query->index = index;
    query->names = names;
    query->is_drop = is_drop;
    query->is_temporary = is_temporary;

    return true;    
    
}

}

}
