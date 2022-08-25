#include <Planner/Utils.h>

#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTSubquery.h>

#include <Columns/getLeastSuperColumn.h>

#include <IO/WriteBufferFromString.h>

#include <Analyzer/QueryNode.h>
#include <Analyzer/ConstantNode.h>
#include <Analyzer/FunctionNode.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TYPE_MISMATCH;
}

String dumpQueryPlan(QueryPlan & query_plan)
{
    WriteBufferFromOwnString query_plan_buffer;
    query_plan.explainPlan(query_plan_buffer, QueryPlan::ExplainPlanOptions{true, true, true, true});
    return query_plan_buffer.str();
}

String dumpQueryPipeline(QueryPlan & query_plan)
{
    QueryPlan::ExplainPipelineOptions explain_pipeline;
    WriteBufferFromOwnString query_pipeline_buffer;
    query_plan.explainPipeline(query_pipeline_buffer, explain_pipeline);
    return query_pipeline_buffer.str();
}

Block buildCommonHeaderForUnion(const Blocks & queries_headers)
{
    size_t num_selects = queries_headers.size();
    Block common_header = queries_headers.front();
    size_t num_columns = common_header.columns();

    for (size_t query_num = 1; query_num < num_selects; ++query_num)
    {
        if (queries_headers.at(query_num).columns() != num_columns)
            throw Exception(ErrorCodes::TYPE_MISMATCH,
                            "Different number of columns in UNION elements: {} and {}",
                            common_header.dumpNames(),
                            queries_headers[query_num].dumpNames());
    }

    std::vector<const ColumnWithTypeAndName *> columns(num_selects);

    for (size_t column_num = 0; column_num < num_columns; ++column_num)
    {
        for (size_t i = 0; i < num_selects; ++i)
            columns[i] = &queries_headers[i].getByPosition(column_num);

        ColumnWithTypeAndName & result_elem = common_header.getByPosition(column_num);
        result_elem = getLeastSuperColumn(columns);
    }

    return common_header;
}

ASTPtr queryNodeToSelectQuery(const QueryTreeNodePtr & query_node)
{
    auto & query_node_typed = query_node->as<QueryNode &>();
    auto result_ast = query_node_typed.toAST();

    if (auto * select_with_union = result_ast->as<ASTSelectWithUnionQuery>())
        result_ast = select_with_union->list_of_selects->children.at(0);

    if (auto * subquery = result_ast->as<ASTSubquery>())
        result_ast = subquery->children.at(0);

    return result_ast;
}

}
