#include <Analyzer/QueryNode.h>

#include <Common/SipHash.h>
#include <Common/FieldVisitorToString.h>

#include <Core/NamesAndTypes.h>

#include <IO/WriteBuffer.h>
#include <IO/WriteHelpers.h>
#include <IO/Operators.h>

#include <Parsers/ASTExpressionList.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Parsers/ASTSubquery.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTFunction.h>

#include <Analyzer/Utils.h>

namespace DB
{

QueryNode::QueryNode()
{
    children.resize(children_size);
    children[with_child_index] = std::make_shared<ListNode>();
    children[projection_child_index] = std::make_shared<ListNode>();
    children[group_by_child_index] = std::make_shared<ListNode>();
    children[order_by_child_index] = std::make_shared<ListNode>();
}

String QueryNode::getName() const
{
    WriteBufferFromOwnString buffer;

    if (hasWith())
    {
        buffer << getWith().getName();
        buffer << ' ';
    }

    buffer << "SELECT ";
    buffer << getProjection().getName();

    if (getJoinTree())
    {
        buffer << " FROM ";
        buffer << getJoinTree()->getName();
    }

    if (getPrewhere())
    {
        buffer << " PREWHERE ";
        buffer << getPrewhere()->getName();
    }

    if (getWhere())
    {
        buffer << " WHERE ";
        buffer << getWhere()->getName();
    }

    if (hasGroupBy())
        buffer << getGroupBy().getName();

    return buffer.str();
}

void QueryNode::dumpTreeImpl(WriteBuffer & buffer, FormatState & format_state, size_t indent) const
{
    buffer << std::string(indent, ' ') << "QUERY id: " << format_state.getNodeId(this);

    if (hasAlias())
        buffer << ", alias: " << getAlias();

    buffer << ", is_subquery: " << is_subquery;
    buffer << ", is_cte: " << is_cte;
    buffer << ", is_distinct: " << is_distinct;
    buffer << ", is_limit_with_ties: " << is_limit_with_ties;

    if (!cte_name.empty())
        buffer << ", cte_name: " << cte_name;

    if (constant_value)
    {
        buffer << ", constant_value: " << constant_value->getValue().dump();
        buffer << ", constant_value_type: " << constant_value->getType()->getName();
    }

    if (hasWith())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "WITH\n";
        getWith().dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (!projection_columns.empty())
    {
        buffer << '\n';
        buffer << std::string(indent + 2, ' ') << "PROJECTION COLUMNS\n";

        size_t projection_columns_size = projection_columns.size();
        for (size_t i = 0; i < projection_columns_size; ++i)
        {
            const auto & projection_column = projection_columns[i];
            buffer << std::string(indent + 4, ' ') << projection_column.name << " " << projection_column.type->getName();
            if (i + 1 != projection_columns_size)
                buffer << '\n';
        }
    }

    buffer << '\n';
    buffer << std::string(indent + 2, ' ') << "PROJECTION\n";
    getProjection().dumpTreeImpl(buffer, format_state, indent + 4);

    if (getJoinTree())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "JOIN TREE\n";
        getJoinTree()->dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (getPrewhere())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "PREWHERE\n";
        getPrewhere()->dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (getWhere())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "WHERE\n";
        getWhere()->dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (hasGroupBy())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "GROUP BY\n";
        getGroupBy().dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (hasOrderBy())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "ORDER BY\n";
        getOrderBy().dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (hasLimit())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "LIMIT\n";
        getLimit()->dumpTreeImpl(buffer, format_state, indent + 4);
    }

    if (hasOffset())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "OFFSET\n";
        getOffset()->dumpTreeImpl(buffer, format_state, indent + 4);
    }
}

bool QueryNode::isEqualImpl(const IQueryTreeNode & rhs) const
{
    const auto & rhs_typed = assert_cast<const QueryNode &>(rhs);
    if (constant_value && rhs_typed.constant_value && *constant_value != *rhs_typed.constant_value)
        return false;
    else if (constant_value && !rhs_typed.constant_value)
        return false;
    else if (!constant_value && rhs_typed.constant_value)
        return false;

    return is_subquery == rhs_typed.is_subquery &&
        is_cte == rhs_typed.is_cte &&
        cte_name == rhs_typed.cte_name &&
        is_distinct == rhs_typed.is_distinct &&
        is_limit_with_ties == rhs_typed.is_limit_with_ties;
}

void QueryNode::updateTreeHashImpl(HashState & state) const
{
    state.update(is_subquery);
    state.update(is_cte);

    state.update(cte_name.size());
    state.update(cte_name);

    state.update(is_distinct);
    state.update(is_limit_with_ties);

    if (constant_value)
    {
        auto constant_dump = applyVisitor(FieldVisitorToString(), constant_value->getValue());
        state.update(constant_dump.size());
        state.update(constant_dump);

        auto constant_value_type_name = constant_value->getType()->getName();
        state.update(constant_value_type_name.size());
        state.update(constant_value_type_name);
    }
}

ASTPtr QueryNode::toASTImpl() const
{
    auto select_query = std::make_shared<ASTSelectQuery>();
    select_query->distinct = is_distinct;

    if (hasWith())
        select_query->setExpression(ASTSelectQuery::Expression::WITH, getWith().toAST());

    select_query->setExpression(ASTSelectQuery::Expression::SELECT, children[projection_child_index]->toAST());

    ASTPtr tables_in_select_query_ast = std::make_shared<ASTTablesInSelectQuery>();
    addTableExpressionOrJoinIntoTablesInSelectQuery(tables_in_select_query_ast, getJoinTree());
    select_query->setExpression(ASTSelectQuery::Expression::TABLES, std::move(tables_in_select_query_ast));

    if (getPrewhere())
        select_query->setExpression(ASTSelectQuery::Expression::PREWHERE, getPrewhere()->toAST());

    if (getWhere())
        select_query->setExpression(ASTSelectQuery::Expression::WHERE, getWhere()->toAST());

    if (hasGroupBy())
        select_query->setExpression(ASTSelectQuery::Expression::GROUP_BY, getGroupBy().toAST());

    if (hasOrderBy())
        select_query->setExpression(ASTSelectQuery::Expression::ORDER_BY, getOrderBy().toAST());

    if (hasLimit())
        select_query->setExpression(ASTSelectQuery::Expression::LIMIT_LENGTH, getLimit()->toAST());

    if (hasOffset())
        select_query->setExpression(ASTSelectQuery::Expression::LIMIT_OFFSET, getOffset()->toAST());

    auto result_select_query = std::make_shared<ASTSelectWithUnionQuery>();
    result_select_query->union_mode = SelectUnionMode::Unspecified;

    auto list_of_selects = std::make_shared<ASTExpressionList>();
    list_of_selects->children.push_back(std::move(select_query));

    result_select_query->children.push_back(std::move(list_of_selects));
    result_select_query->list_of_selects = result_select_query->children.back();

    if (is_subquery)
    {
        auto subquery = std::make_shared<ASTSubquery>();

        subquery->cte_name = cte_name;
        subquery->children.push_back(std::move(result_select_query));

        return subquery;
    }

    return result_select_query;
}

QueryTreeNodePtr QueryNode::cloneImpl() const
{
    auto result_query_node = std::make_shared<QueryNode>();

    result_query_node->is_subquery = is_subquery;
    result_query_node->is_cte = is_cte;
    result_query_node->is_distinct = is_distinct;
    result_query_node->is_limit_with_ties = is_limit_with_ties;
    result_query_node->cte_name = cte_name;
    result_query_node->projection_columns = projection_columns;
    result_query_node->constant_value = constant_value;

    return result_query_node;
}

}
