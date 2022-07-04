#include <Storages/MergeTree/CommonANNIndexes.h>
#include <Storages/MergeTree/KeyCondition.h>

#include <Parsers/ASTFunction.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTOrderByElement.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/ASTSetQuery.h>

#include <Interpreters/Context.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int INCORRECT_QUERY;
}

namespace ApproximateNearestNeighbour
{

ANNCondition::ANNCondition(const SelectQueryInfo & query_info,
                                 ContextPtr context) :
    block_with_constants{KeyCondition::getBlockWithConstants(query_info.query, query_info.syntax_analyzer_result, context)},
    ann_index_params{context->getSettings().get("ann_index_params").get<String>()},
    index_is_useful{checkQueryStructure(query_info)} {}

bool ANNCondition::alwaysUnknownOrTrue(String metric_name) const
{
    if (!index_is_useful)
    {
        return true; // Query isn't supported
    }
    // If query is supported, check metrics for match
    return !(metric_name == query_information->metric_name);
}

///TODO: check for all getters?
float ANNCondition::getComparisonDistanceForWhereQuery() const
{
    ///TODO: query_information->???
    if (query_information->query_type == ANNQueryInformation::Type::WhereQuery)
    {
        return query_information->distance;
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Not supported method for this query type");
}

UInt64 ANNCondition::getLimitCount() const
{
    if (index_is_useful)
    {
        return query_information->limit;
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "No LIMIT section in query, not supported");
}

bool ANNCondition::checkQueryStructure(const SelectQueryInfo & query)
{
    // RPN-s for different sections of the query
    RPN rpn_prewhere_clause;
    RPN rpn_where_clause;
    RPN rpn_order_by_clause;
    RPNElement rpn_limit;

    ANNQueryInformation prewhere_info;
    ANNQueryInformation where_info;
    ANNQueryInformation order_by_info;

    // Build rpns for query sections
    const auto & select = query.query->as<ASTSelectQuery &>();

    if (select.prewhere()) // If query has PREWHERE clause
    {
        traverseAST(select.prewhere(), rpn_prewhere_clause);
    }

    if (select.where()) // If query has WHERE clause
    {
        traverseAST(select.where(), rpn_where_clause);
    }

    if (select.limitLength()) // If query has LIMIT clause
    {
        traverseAtomAST(select.limitLength(), rpn_limit);
    }

    if (select.orderBy()) // If query has ORDERBY clause
    {
        traverseOrderByAST(select.orderBy(), rpn_order_by_clause);
    }

    // Reverse RPNs for conveniences during parsing
    std::reverse(rpn_prewhere_clause.begin(), rpn_prewhere_clause.end());
    std::reverse(rpn_where_clause.begin(), rpn_where_clause.end());
    std::reverse(rpn_order_by_clause.begin(), rpn_order_by_clause.end());

    // Match rpns with supported types and extract information
    const bool prewhere_is_valid = matchRPNWhere(rpn_prewhere_clause, prewhere_info);
    const bool where_is_valid = matchRPNWhere(rpn_where_clause, where_info);
    const bool limit_is_valid = matchRPNLimit(rpn_limit, query_information->limit);
    const bool order_by_is_valid = matchRPNOrderBy(rpn_order_by_clause, order_by_info);

    // Query without LIMIT clause is not supported
    if (!limit_is_valid)
    {
        return false;
    }

    // Search type query in both sections isn't supported
    if (prewhere_is_valid && where_is_valid)
    {
        return false;
    }

    // Search type should be in WHERE or PREWHERE clause
    if (prewhere_is_valid || where_is_valid)
    {
        query_information = std::move(where_is_valid ? where_info : prewhere_info);
    }

    if (order_by_is_valid)
    {
        // Query with valid where and order by type is not supported
        if (query_information.has_value())
        {
            return false;
        }

        query_information = std::move(order_by_info);
    }

    return query_information.has_value();
}

void ANNCondition::traverseAST(const ASTPtr & node, RPN & rpn)
{
    // If the node is ASTFunction, it may have children nodes
    if (const auto * func = node->as<ASTFunction>())
    {
        const ASTs & children = func->arguments->children;
        // Traverse children nodes
        for (const auto& child : children)
        {
            traverseAST(child, rpn);
        }
    }

    RPNElement element;
    // Get the data behind node
    if (!traverseAtomAST(node, element))
    {
        element.function = RPNElement::FUNCTION_UNKNOWN;
    }

    rpn.emplace_back(std::move(element));
}

bool ANNCondition::traverseAtomAST(const ASTPtr & node, RPNElement & out)
{
    // Match Functions
    if (const auto * function = node->as<ASTFunction>())
    {
        // Set the name
        out.func_name = function->name;

        if (function->name == "L1Distance" ||
            function->name == "L2Distance" ||
            function->name == "LinfDistance" ||
            function->name == "cosineDistance" ||
            function->name == "dotProduct" ||
            function->name == "LpDistance")
        {
            out.function = RPNElement::FUNCTION_DISTANCE;
        }
        else if (function->name == "tuple")
        {
            out.function = RPNElement::FUNCTION_TUPLE;
        }
        else if (function->name == "less" ||
                 function->name == "greater" ||
                 function->name == "lessOrEquals" ||
                 function->name == "greaterOrEquals")
        {
            out.function = RPNElement::FUNCTION_COMPARISON;
        }
        else
        {
            return false;
        }

        return true;
    }
    // Match identifier
    else if (const auto * identifier = node->as<ASTIdentifier>())
    {
        out.function = RPNElement::FUNCTION_IDENTIFIER;
        out.identifier.emplace(identifier->name());
        out.func_name = "column identifier";

        return true;
    }

    // Check if we have constants behind the node
    return tryCastToConstType(node, out);
}

bool ANNCondition::tryCastToConstType(const ASTPtr & node, RPNElement & out)
{
    Field const_value;
    DataTypePtr const_type;

    if (KeyCondition::getConstant(node, block_with_constants, const_value, const_type))
    {
        /// Check for constant types
        if (const_value.getType() == Field::Types::Float64)
        {
            out.function = RPNElement::FUNCTION_FLOAT_LITERAL;
            out.float_literal.emplace(const_value.get<Float32>());
            out.func_name = "Float literal";
            return true;
        }

        /// TODO: Uint?
        if (const_value.getType() == Field::Types::UInt64)
        {
            out.function = RPNElement::FUNCTION_INT_LITERAL;
            out.int_literal.emplace(const_value.get<UInt64>());
            out.func_name = "Int literal";
            return true;
        }

        if (const_value.getType() == Field::Types::Int64)
        {
            out.function = RPNElement::FUNCTION_INT_LITERAL;
            out.int_literal.emplace(const_value.get<Int64>());
            out.func_name = "Int literal";
            return true;
        }

        if (const_value.getType() == Field::Types::Tuple)
        {
            out.function = RPNElement::FUNCTION_LITERAL_TUPLE;
            out.tuple_literal = const_value.get<Tuple>();
            out.func_name = "Tuple literal";
            return true;
        }
    }

    return false;
}

void ANNCondition::traverseOrderByAST(const ASTPtr & node, RPN & rpn)
{
    if (const auto * expr_list = node->as<ASTExpressionList>())
    {
        if (const auto * order_by_element = expr_list->children.front()->as<ASTOrderByElement>())
        {
            traverseAST(order_by_element->children.front(), rpn);
        }
    }
}

// Returns true and stores ANNQueryInformation if the query has valid WHERE clause
bool ANNCondition::matchRPNWhere(RPN & rpn, ANNQueryInformation & expr)
{
    // WHERE section must have at least 5 expressions
    // Operator->Distance(float)->DistanceFunc->Column->TupleFunc(TargetVector(floats))
    if (rpn.size() < 5)
    {
        return false;
    }

    auto iter = rpn.begin();
    bool identifier_found = false;

    // Query starts from operator less
    if (iter->function != RPNElement::FUNCTION_COMPARISON)
    {
        return false;
    }

    const bool greater_case = iter->func_name == "greater" || iter->func_name == "greaterOrEquals";
    const bool less_case = iter->func_name == "less" || iter->func_name == "lessOrEquals";

    ++iter;

    if (less_case)
    {
        if (iter->function != RPNElement::FUNCTION_FLOAT_LITERAL)
        {
            return false;
        }

        expr.distance = getFloatOrIntLiteralOrPanic(iter);
        ++iter;

    }
    else if (!greater_case)
    {
        return false;
    }

    auto end = rpn.end();
    if (!matchMainParts(iter, end, expr, identifier_found))
    {
        return false;
    }

    // Final checks of correctness
    if (!identifier_found || expr.target.empty())
    {
        return false;
    }

    if (greater_case)
    {
        if (expr.target.size() < 2)
        {
            return false;
        }
        expr.distance = expr.target.back();
        expr.target.pop_back();
    }

    // query is ok
    return true;
}

// Returns true and stores ANNExpr if the query has valid ORDERBY clause
bool ANNCondition::matchRPNOrderBy(RPN & rpn, ANNQueryInformation & expr)
{
    // ORDER BY clause must have at least 3 expressions
    if (rpn.size() < 3)
    {
        return false;
    }

    auto iter = rpn.begin();
    auto end = rpn.end();
    bool identifier_found = false;

    return ANNCondition::matchMainParts(iter, end, expr, identifier_found);
}

// Returns true and stores Length if we have valid LIMIT clause in query
bool ANNCondition::matchRPNLimit(RPNElement & rpn, UInt64 & limit)
{
    if (rpn.function == RPNElement::FUNCTION_INT_LITERAL)
    {
        limit = rpn.int_literal.value();
        return true;
    }

    return false;
}

/* Matches dist function, target vector, column name */
bool ANNCondition::matchMainParts(RPN::iterator & iter, RPN::iterator & end, ANNQueryInformation & expr, bool & identifier_found)
{
    // Matches DistanceFunc->[Column]->[TupleFunc]->TargetVector(floats)->[Column]
    if (iter->function != RPNElement::FUNCTION_DISTANCE)
    {
        return false;
    }

    expr.metric_name = iter->func_name;
    ++iter;

    if (expr.metric_name == "LpDistance")
    {
        if (iter->function != RPNElement::FUNCTION_FLOAT_LITERAL &&
            iter->function != RPNElement::FUNCTION_INT_LITERAL)
        {
            return false;
        }
        expr.p_for_lp_dist = getFloatOrIntLiteralOrPanic(iter);
        ++iter;
    }


    if (iter->function == RPNElement::FUNCTION_IDENTIFIER)
    {
        identifier_found = true;
        expr.column_name = std::move(iter->identifier.value());
        ++iter;
    }

    if (iter->function == RPNElement::FUNCTION_TUPLE)
    {
        ++iter;
    }

    if (iter->function == RPNElement::FUNCTION_LITERAL_TUPLE)
    {
        for (const auto & value : iter->tuple_literal.value())
        {
            expr.target.emplace_back(value.get<float>());
        }
        ++iter;
    }


    while (iter != end)
    {
        if (iter->function == RPNElement::FUNCTION_FLOAT_LITERAL ||
            iter->function == RPNElement::FUNCTION_INT_LITERAL)
        {
            expr.target.emplace_back(getFloatOrIntLiteralOrPanic(iter));
        }
        else if (iter->function == RPNElement::FUNCTION_IDENTIFIER)
        {
            if (identifier_found)
            {
                return false;
            }
            expr.column_name = std::move(iter->identifier.value());
            identifier_found = true;
        }
        else
        {
            return false;
        }

        ++iter;
    }

    return true;
}

// Gets float or int from AST node
float ANNCondition::getFloatOrIntLiteralOrPanic(RPN::iterator& iter)
{
    if (iter->float_literal.has_value())
    {
        return iter->float_literal.value();
    }
    if (iter->int_literal.has_value())
    {
        return static_cast<float>(iter->int_literal.value());
    }
    throw Exception("Wrong parsed AST in buildRPN\n", ErrorCodes::INCORRECT_QUERY);
}

}

}
