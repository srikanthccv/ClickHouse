#include <Interpreters/ExtractExpressionInfoVisitor.h>
#include <Functions/FunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <Interpreters/IdentifierSemantic.h>
#include <Parsers/ASTSubquery.h>


namespace DB
{

void ExpressionInfoMatcher::visit(const ASTPtr & ast, Data & data)
{
    if (const auto * function = ast->as<ASTFunction>())
        visit(*function, ast, data);
    else if (const auto * identifier = ast->as<ASTIdentifier>())
        visit(*identifier, ast, data);
}

void ExpressionInfoMatcher::visit(const ASTFunction & ast_function, const ASTPtr &, Data & data)
{
    if (ast_function.name == "arrayJoin")
        data.is_array_join = true;
    else if (AggregateFunctionFactory::instance().isAggregateFunctionName(ast_function.name))
        data.is_aggregate_function = true;
    else
    {
        const auto & function = FunctionFactory::instance().tryGet(ast_function.name, data.context);

        /// Skip lambda, tuple and other special functions
        if (function && function->isStateful())
            data.is_stateful_function = true;
    }
}

void ExpressionInfoMatcher::visit(const ASTIdentifier & identifier, const ASTPtr &, Data & data)
{
    if (!identifier.compound())
    {
        for (size_t index = 0; index < data.tables.size(); ++index)
        {
            const auto & columns = data.tables[index].columns;

            // TODO: make sure no collision ever happens
            if (std::find(columns.begin(), columns.end(), identifier.name) != columns.end())
            {
                data.unique_reference_tables_pos.emplace(index);
                break;
            }
        }
    }
    else
    {
        size_t best_table_pos = 0;
        if (IdentifierSemantic::chooseTable(identifier, data.tables, best_table_pos))
            data.unique_reference_tables_pos.emplace(best_table_pos);
    }
}

bool ExpressionInfoMatcher::needChildVisit(const ASTPtr & node, const ASTPtr &)
{
    if (const auto & function = node->as<ASTFunction>(); function && function->name == "lambda")
        return false;

    return !node->as<ASTSubquery>();
}

bool hasStatefulFunction(const ASTPtr & node, const Context & context)
{
    for (const auto & select_expression : node->children)
    {
        ExpressionInfoVisitor::Data expression_info{.context = context, .tables = {}};
        ExpressionInfoVisitor(expression_info).visit(select_expression);

        if (expression_info.is_stateful_function)
            return true;
    }

    return false;
}

}

