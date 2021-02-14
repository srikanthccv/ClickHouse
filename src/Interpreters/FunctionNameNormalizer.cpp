#include <Interpreters/FunctionNameNormalizer.h>

#include <Parsers/ASTColumnDeclaration.h>
#include <Parsers/ASTCreateQuery.h>

namespace DB
{

const String & getFunctionCanonicalNameIfAny(const String & name);
const String & getAggregateFunctionCanonicalNameIfAny(const String & name);

void FunctionNameNormalizer::visit(IAST * ast)
{
    if (!ast)
        return;

    if (auto * node_storage = ast->as<ASTStorage>())
    {
        visit(node_storage->partition_by);
        visit(node_storage->primary_key);
        visit(node_storage->order_by);
        visit(node_storage->sample_by);
        visit(node_storage->ttl_table);
        return;
    }

    if (auto * node_decl = ast->as<ASTColumnDeclaration>())
    {
        visit(node_decl->default_expression.get());
        visit(node_decl->ttl.get());
        return;
    }

    if (auto * node_func = ast->as<ASTFunction>())
        node_func->name = getAggregateFunctionCanonicalNameIfAny(getFunctionCanonicalNameIfAny(node_func->name));

    for (auto & child : ast->children)
        visit(child.get());
}

}
