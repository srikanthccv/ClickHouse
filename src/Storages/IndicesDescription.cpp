#include <Interpreters/ExpressionAnalyzer.h>
#include <Interpreters/SyntaxAnalyzer.h>
#include <Storages/IndicesDescription.h>

#include <Parsers/ASTIndexDeclaration.h>
#include <Parsers/formatAST.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/parseQuery.h>
#include <Storages/extractKeyExpressionList.h>

#include <Core/Defines.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int INCORRECT_QUERY;
};


StorageMetadataSkipIndexField StorageMetadataSkipIndexField::getSkipIndexFromAST(const ASTPtr & definition_ast, const ColumnsDescription & columns, const Context & context)
{
    const auto * index_definition = definition_ast->as<ASTIndexDeclaration>();
    if (!index_definition)
        throw Exception("Cannot create skip index from non ASTIndexDeclaration AST", ErrorCodes::LOGICAL_ERROR);

    if (index_definition->name.empty())
        throw Exception("Skip index must have name in definition.", ErrorCodes::INCORRECT_QUERY);

    if (!index_definition->type)
        throw Exception("TYPE is required for index", ErrorCodes::INCORRECT_QUERY);

    if (index_definition->type->parameters && !index_definition->type->parameters->children.empty())
        throw Exception("Index type cannot have parameters", ErrorCodes::INCORRECT_QUERY);

    StorageMetadataSkipIndexField result;
    result.definition_ast = index_definition->clone();
    result.name = index_definition->name;
    result.type = Poco::toLower(index_definition->type->name);
    result.granularity = index_definition->granularity;

    ASTPtr expr_list = extractKeyExpressionList(index_definition->expr->clone());
    result.expression_list_ast = expr_list->clone();

    auto syntax = SyntaxAnalyzer(context).analyze(expr_list, columns.getAllPhysical());
    result.expression = ExpressionAnalyzer(expr_list, syntax, context).getActions(true);
    result.sample_block = result.expression->getSampleBlock();

    for (size_t i = 0; i < result.sample_block.columns(); ++i)
    {
        const auto & column = result.sample_block.getByPosition(i);
        result.column_names.emplace_back(column.name);
        result.data_types.emplace_back(column.type);
    }

    const auto & definition_arguments = index_definition->type->arguments;
    if (definition_arguments)
    {
        for (size_t i = 0; i < definition_arguments->children.size(); ++i)
        {
            const auto * argument = definition_arguments->children[i]->as<ASTLiteral>();
            if (!argument)
                throw Exception("Only literals can be skip index arguments", ErrorCodes::INCORRECT_QUERY);
            result.arguments.emplace_back(argument->value);
        }
    }

    return result;
}


bool IndicesDescription::has(const String & name) const
{
    for (const auto & index : *this)
        if (index.name == name)
            return true;
    return false;
}

String IndicesDescription::toString() const
{
    if (empty())
        return {};

    ASTExpressionList list;
    for (const auto & index : *this)
        list.children.push_back(index.definition_ast);

    return serializeAST(list, true);
}


IndicesDescription IndicesDescription::parse(const String & str, const ColumnsDescription & columns, const Context & context)
{
    IndicesDescription result;
    if (str.empty())
        return result;

    ParserIndexDeclarationList parser;
    ASTPtr list = parseQuery(parser, str, 0, DBMS_DEFAULT_MAX_PARSER_DEPTH);

    for (const auto & index : list->children)
        result.emplace_back(StorageMetadataSkipIndexField::getSkipIndexFromAST(index, columns, context));

    return result;
}

}
