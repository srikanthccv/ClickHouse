#include <Functions/CastOverloadResolver.h>
#include <Functions/FunctionsLogical.h>
#include <Planner/PlannerActionsVisitor.h>
#include <Storages/SelectQueryInfo.h>
#include <Storages/MergeTree/MergeTreeRangeReader.h>
#include <Interpreters/ExpressionActions.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{

/// Stores the ist of columns required to compute a node in the DAG.
struct NodeInfo
{
    NameSet required_columns;
};

/// Fills the list of required columns for a node in the DAG.
void fillRequiredColumns(const ActionsDAG::Node * node, std::unordered_map<const ActionsDAG::Node *, NodeInfo> & nodes_info)
{
    if (nodes_info.contains(node))
        return;

    auto & node_info = nodes_info[node];

    if (node->type == ActionsDAG::ActionType::INPUT)
    {
        node_info.required_columns.insert(node->result_name);
        return;
    }

    for (const auto & child : node->children)
    {
        fillRequiredColumns(child, nodes_info);
        const auto & child_info = nodes_info[child];
        node_info.required_columns.insert(child_info.required_columns.begin(), child_info.required_columns.end());
    }
}

/// Stores information about a node that has already been cloned to one of the new DAGs.
/// This allows to avoid cloning the same sub-DAG into multiple step DAGs but reference previously cloned nodes from earliers steps.
struct DAGNodeRef
{
    ActionsDAGPtr dag;
    const ActionsDAG::Node * node;
};

using OriginalToNewNodeMap = std::unordered_map<const ActionsDAG::Node *, DAGNodeRef>;

/// Clones the part of original DAG responsible for computing the original_dag_node and adds it to the new DAG.
const ActionsDAG::Node & addClonedDAGToDAG(const ActionsDAG::Node * original_dag_node, ActionsDAGPtr new_dag, OriginalToNewNodeMap & node_remap)
{
    /// Look for the node in the map of already known nodes
    if (node_remap.contains(original_dag_node))
    {
        /// If the node is already in the new DAG, return it
        const auto & node_ref = node_remap.at(original_dag_node);
        if (node_ref.dag == new_dag)
            return *node_ref.node;

        /// If the node is known from the previous steps, add it as an input, except for constants
        if (original_dag_node->type != ActionsDAG::ActionType::COLUMN)
        {
            node_ref.dag->addOrReplaceInOutputs(*node_ref.node);
            const auto & new_node = new_dag->addInput(node_ref.node->result_name, node_ref.node->result_type);
            node_remap[original_dag_node] = {new_dag, &new_node}; /// TODO: here we update the node reference. Is it always correct?
            return new_node;
        }
    }

    /// If the node is an input, add it as an input
    if (original_dag_node->type == ActionsDAG::ActionType::INPUT)
    {
        const auto & new_node = new_dag->addInput(original_dag_node->result_name, original_dag_node->result_type);
        node_remap[original_dag_node] = {new_dag, &new_node};
        return new_node;
    }

    /// If the node is a column, add it as an input
    if (original_dag_node->type == ActionsDAG::ActionType::COLUMN)
    {
        const auto & new_node = new_dag->addColumn(
            ColumnWithTypeAndName(original_dag_node->column, original_dag_node->result_type, original_dag_node->result_name));
        node_remap[original_dag_node] = {new_dag, &new_node};
        return new_node;
    }

    /// TODO: Do we need to handle ALIAS nodes in cloning?

    /// If the node is a function, add it as a function and add its children
    if (original_dag_node->type == ActionsDAG::ActionType::FUNCTION)
    {
        ActionsDAG::NodeRawConstPtrs new_children;
        for (const auto & child : original_dag_node->children)
        {
            const auto & new_child = addClonedDAGToDAG(child, new_dag, node_remap);
            new_children.push_back(&new_child);
        }

        const auto & new_node = new_dag->addFunction(original_dag_node->function_base, new_children, original_dag_node->result_name);
        node_remap[original_dag_node] = {new_dag, &new_node};
        return new_node;
    }

    throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected node type in PREWHERE actions: {}", original_dag_node->type);
}

/// Adds a CAST node with the regular name ("CAST(...)") or with the provided name.
/// This is different from ActionsDAG::addCast() because it set the name equal to the original name effectively hiding the value before cast,
/// but it might be required for further steps with its original uncasted type.
const ActionsDAG::Node & addCast(ActionsDAGPtr dag, const ActionsDAG::Node & node_to_cast, const String & type_name, const String & new_name = {})
{
    Field cast_type_constant_value(type_name);

    ColumnWithTypeAndName column;
    column.name = calculateConstantActionNodeName(cast_type_constant_value);
    column.column = DataTypeString().createColumnConst(0, cast_type_constant_value);
    column.type = std::make_shared<DataTypeString>();

    const auto * cast_type_constant_node = &dag->addColumn(std::move(column));
    ActionsDAG::NodeRawConstPtrs children = {&node_to_cast, cast_type_constant_node};
    FunctionOverloadResolverPtr func_builder_cast = CastInternalOverloadResolver<CastType::nonAccurate>::createImpl();

    return dag->addFunction(func_builder_cast, std::move(children), new_name);
}

}

/// We want to build a sequence of steps that will compute parts of the prewhere condition.
/// Each step reads some new columns and computes some new expressions and a filter condition.
/// The last step computes the final filter condition and the remaining expressions that are required for the main query.
/// The goal of this is to, when it is possible, filter out many rows in early steps so that the remaining steps will
/// read less data from the storage.
/// NOTE: The result of executing the steps is exactly the same as if we would execute the original DAG in single step.
///
/// The steps are built in the following way:
/// 1. List all condition nodes that are combined with AND into PREWHERE condition
/// 2. Collect the set of columns that are used in each condition
/// 3. Sort condition nodes by the number of columns used in them and the overall size of those columns
/// 4. Group conditions with the same set of columns into a single read/compute step
/// 5. Build DAGs for each step:
///    - DFS from the condition root node:
///      - If the node was not computed yet, add it to the DAG and traverse its children
///      - If the node was already computed by one of the previous steps, add it as output for that step and as input for the current step
///      - If the node was already computed by the current step just stop traversing
/// 6. Find all outputs of the original DAG
/// 7. Find all outputs that were computed in the already built DAGs, mark these nodes as outputs in the steps where they were computed
/// 8. Add computation of the remaining outputs to the last step with the procedure similar to 4
bool tryBuildPrewhereSteps(PrewhereInfoPtr prewhere_info, const ExpressionActionsSettings & actions_settings, PrewhereExprInfo & prewhere)
{
    if (!prewhere_info || !prewhere_info->prewhere_actions)
        return true;

    Poco::Logger * log = &Poco::Logger::get("tryBuildPrewhereSteps");

    LOG_TRACE(log, "Original PREWHERE DAG:\n{}", prewhere_info->prewhere_actions->dumpDAG());

    /// 1. List all condition nodes that are combined with AND into PREWHERE condition
    const auto & condition_root = prewhere_info->prewhere_actions->findInOutputs(prewhere_info->prewhere_column_name);
    const bool is_conjunction = (condition_root.type == ActionsDAG::ActionType::FUNCTION && condition_root.function_base->getName() == "and");
    if (!is_conjunction)
        return false;
    auto condition_nodes = condition_root.children;

    /// 2. Collect the set of columns that are used in the condition
    std::unordered_map<const ActionsDAG::Node *, NodeInfo> nodes_info;
    for (const auto & node : condition_nodes)
    {
        fillRequiredColumns(node, nodes_info);
    }

    /// 3. Sort condition nodes by the number of columns used in them and the overall size of those columns
    /// TODO: not sorting for now because the conditions are already sorted by Where Optimizer

    /// 4. Group conditions with the same set of columns into a single read/compute step
    std::vector<std::vector<const ActionsDAG::Node *>> condition_groups;
    for (const auto & node : condition_nodes)
    {
        const auto & node_info = nodes_info[node];
        if (!condition_groups.empty() && nodes_info[condition_groups.back().back()].required_columns == node_info.required_columns)
            condition_groups.back().push_back(node);    /// Add to the last group
        else
            condition_groups.push_back({node}); /// Start new group
    }

    /// 5. Build DAGs for each step
    struct Step
    {
        ActionsDAGPtr actions;
        String column_name;
    };
    std::vector<Step> steps;

    OriginalToNewNodeMap node_remap;

    for (const auto & condition_group : condition_groups)
    {
        ActionsDAGPtr step_dag = std::make_shared<ActionsDAG>();
        String result_name;

        std::vector<const ActionsDAG::Node *> new_condition_nodes;
        for (const auto * node : condition_group)
        {
            const auto & node_in_new_dag = addClonedDAGToDAG(node, step_dag, node_remap);
            new_condition_nodes.push_back(&node_in_new_dag);
        }

        if (new_condition_nodes.size() > 1)
        {
            /// Add AND function to combine the conditions
            FunctionOverloadResolverPtr func_builder_and = std::make_unique<FunctionToOverloadResolverAdaptor>(std::make_shared<FunctionAnd>());
            const auto & and_function_node = step_dag->addFunction(func_builder_and, new_condition_nodes, "");
            step_dag->addOrReplaceInOutputs(and_function_node);
            result_name = and_function_node.result_name;
        }
        else
        {
            const auto & result_node = *new_condition_nodes.front();
            /// Add cast to UInt8 if needed
            if (result_node.result_type->getTypeId() == TypeIndex::UInt8)
            {
                step_dag->addOrReplaceInOutputs(result_node);
                result_name = result_node.result_name;
            }
            else
            {
                const auto & cast_node = addCast(step_dag, result_node, "UInt8");
                step_dag->addOrReplaceInOutputs(cast_node);
                result_name = cast_node.result_name;
            }
        }

        steps.push_back({step_dag, result_name});
    }

    /// 6. Find all outputs of the original DAG
    auto original_outputs = prewhere_info->prewhere_actions->getOutputs();
    /// 7. Find all outputs that were computed in the already built DAGs, mark these nodes as outputs in the steps where they were computed
    /// 8. Add computation of the remaining outputs to the last step with the procedure similar to 4
    NameSet all_output_names;
    for (const auto * output : original_outputs)
    {
        all_output_names.insert(output->result_name);
        if (node_remap.contains(output))
        {
            const auto & new_node_info = node_remap[output];
            new_node_info.dag->addOrReplaceInOutputs(*new_node_info.node);
        }
        else if (output->result_name == prewhere_info->prewhere_column_name)
        {
            /// Special case for final PREWHERE column: it is an AND combination of all conditions,
            /// but we have only the condition for the last step here.
            /// However we know that the ultimate result after filtering is constant 1 for the PREWHERE column.
            auto const_true = output->result_type->createColumnConst(0, Field{1});
            const auto & prewhere_result_node =
                steps.back().actions->addColumn(ColumnWithTypeAndName(const_true, output->result_type, output->result_name));
            steps.back().actions->addOrReplaceInOutputs(prewhere_result_node);
        }
        else
        {
            const auto & node_in_new_dag = addClonedDAGToDAG(output, steps.back().actions, node_remap);
            steps.back().actions->addOrReplaceInOutputs(node_in_new_dag);
        }
    }

    /// 9. Build PrewhereExprInfo
    {
        for (const auto & step : steps)
        {
            prewhere.steps.push_back(
            {
                .actions = std::make_shared<ExpressionActions>(step.actions, actions_settings),
                .column_name = step.column_name,
                .remove_column = !all_output_names.contains(step.column_name), /// Don't remove if it's in the list of original outputs
                .need_filter = false,
            });
        }
        prewhere.steps.back().need_filter = prewhere_info->need_filter;
    }

    LOG_TRACE(log, "Resulting PREWHERE:\n{}", prewhere.dump());

    return true;
}

}
