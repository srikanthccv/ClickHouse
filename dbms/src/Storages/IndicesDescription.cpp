#include <Storages/IndicesDescription.h>

#include <Parsers/formatAST.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/parseQuery.h>


namespace DB
{

IndicesDescription::IndicesDescription(const IndicesDescription & indices_)
{
    for (const auto & index : indices_.indices)
        indices.push_back(
                std::dynamic_pointer_cast<ASTIndexDeclaration>(index->clone()));
}

IndicesDescription & IndicesDescription::operator=(const IndicesDescription & indices_)
{
    for (const auto & index : indices_.indices)
        indices.push_back(
                std::dynamic_pointer_cast<ASTIndexDeclaration>(index->clone()));
    return *this;
}

String IndicesDescription::toString() const
{
    if (indices.empty())
        return {};

    ASTExpressionList list;
    for (const auto & index : indices)
        list.children.push_back(index);

    return serializeAST(list, true);
}

IndicesDescription IndicesDescription::parse(const String & str)
{
    if (str.empty())
        return {};

    IndicesDescription res;
    ParserIndexDeclarationList parser;
    ASTPtr list = parseQuery(parser, str, 0);

    for (const auto & index : list->children)
        res.indices.push_back(std::dynamic_pointer_cast<ASTIndexDeclaration>(index));

    return res;
}

}
