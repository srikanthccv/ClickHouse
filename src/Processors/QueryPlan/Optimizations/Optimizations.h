#pragma once
#include <Processors/QueryPlan/QueryPlan.h>
#include <array>

namespace DB
{

namespace QueryPlanOptimizations
{

/// This is the main function which optimizes the whole QueryPlan tree.
void optimizeTree(QueryPlan::Node & root, QueryPlan::Nodes & nodes);

/// Optimization is a function applied to QueryPlan::Node.
/// It can read and update subtree of specified node.
/// It return the number of updated layers of subtree if some change happened.
/// It must guarantee that the structure of tree is correct.
///
/// New nodes should be added to QueryPlan::Nodes list.
/// It is not needed to remove old nodes from the list.
struct Optimization
{
    using Function = size_t (*)(QueryPlan::Node *, QueryPlan::Nodes &);
    const Function apply = nullptr;
    const char * name;
};

/// Move ARRAY JOIN up if possible.
size_t tryLiftUpArrayJoin(QueryPlan::Node * parent_node, QueryPlan::Nodes & nodes);

/// Move LimitStep down if possible.
size_t tryPushDownLimit(QueryPlan::Node * parent_node, QueryPlan::Nodes &);

/// Split FilterStep into chain `ExpressionStep -> FilterStep`, where FilterStep contains minimal number of nodes.
size_t trySplitFilter(QueryPlan::Node * node, QueryPlan::Nodes & nodes);

/// Replace chain `ExpressionStep -> ExpressionStep` to single ExpressionStep
/// Replace chain `FilterStep -> ExpressionStep` to single FilterStep
size_t tryMergeExpressions(QueryPlan::Node * parent_node, QueryPlan::Nodes &);

/// Move FilterStep down if possible.
/// May split FilterStep and push down only part of it.
size_t tryPushDownLimit(QueryPlan::Node * parent_node, QueryPlan::Nodes & nodes);

inline const auto & getOptimizations()
{
    static const std::array<Optimization, 5> optimizations =
    {{
        {tryLiftUpArrayJoin, "liftUpArrayJoin"},
        {tryPushDownLimit, "pushDownLimit"},
        {trySplitFilter, "splitFilter"},
        {tryMergeExpressions, "mergeExpressions"},
        {tryPushDownLimit, "pushDownFilter"},
     }};

    return optimizations;
}

}

}
