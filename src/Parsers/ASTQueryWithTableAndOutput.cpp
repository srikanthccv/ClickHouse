#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTQueryWithTableAndOutput.h>
#include <Common/quoteString.h>
#include <IO/Operators.h>


namespace DB
{

String ASTQueryWithTableAndOutput::getDatabase() const
{
    String name;
    tryGetIdentifierNameInto(database, name);
    return name;
}

String ASTQueryWithTableAndOutput::getTable() const
{
    String name;
    tryGetIdentifierNameInto(table, name);
    return name;
}

void ASTQueryWithTableAndOutput::setDatabase(const String & name)
{
    if (name.empty() && !database)
        return;

    assert(!name.empty());

    if (database)
    {
        if (auto * database_ptr = database->as<ASTIdentifier>())
            database_ptr->setShortName(name);
    }
    else
    {
        database = std::make_shared<ASTIdentifier>(name);
        children.push_back(database);
    }
}

void ASTQueryWithTableAndOutput::setTable(const String & name)
{
    if (name.empty() && !table)
        return;

    assert(!name.empty());

    if (table)
    {
        if (auto * table_ptr = table->as<ASTIdentifier>())
            table_ptr->setShortName(name);
    }
    else
    {
        table = std::make_shared<ASTIdentifier>(name);
        children.push_back(table);
    }
}

void ASTQueryWithTableAndOutput::cloneTableOptions(ASTQueryWithTableAndOutput & cloned) const
{
    if (database)
    {
        cloned.database = database->clone();
        cloned.children.push_back(cloned.database);
    }
    if (table)
    {
        cloned.table = table->clone();
        cloned.children.push_back(cloned.table);
    }
}
void ASTQueryWithTableAndOutput::formatHelper(const FormatSettings & settings, const char * name) const
{
    settings.ostr << (settings.hilite ? hilite_keyword : "") << name << " " << (settings.hilite ? hilite_none : "");
    settings.ostr << (!getDatabase().empty() ? backQuoteIfNeed(getDatabase()) + "." : "") << backQuoteIfNeed(getTable());
}

}

