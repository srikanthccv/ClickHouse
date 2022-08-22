#include <Parsers/ASTWindowDefinition.h>
#include <Processors/QueryPlan/Optimizations/Optimizations.h>
#include <Processors/QueryPlan/ITransformingStep.h>
#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPlan/ArrayJoinStep.h>
#include <Processors/QueryPlan/CreatingSetsStep.h>
#include <Processors/QueryPlan/CubeStep.h>
#include <Processors/QueryPlan/ReadFromMergeTree.h>
#include <Processors/QueryPlan/SortingStep.h>
#include <Processors/QueryPlan/TotalsHavingStep.h>
#include <Processors/QueryPlan/DistinctStep.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Processors/QueryPlan/WindowStep.h>
#include <Interpreters/ActionsDAG.h>
#include <Interpreters/ArrayJoinAction.h>
#include <Interpreters/InterpreterSelectQuery.h>
#include <Interpreters/TableJoin.h>
#include <Common/typeid_cast.h>
#include <Functions/IFunction.h>
#include <DataTypes/DataTypeAggregateFunction.h>
#include <Columns/IColumn.h>
#include <stack>


namespace DB::QueryPlanOptimizations
{

ReadFromMergeTree * findReadingStep(QueryPlan::Node * node)
{
    IQueryPlanStep * step = node->step.get();
    if (auto * reading = typeid_cast<ReadFromMergeTree *>(step))
        return reading;

    if (node->children.size() != 1)
        return nullptr;

    if (typeid_cast<ExpressionStep *>(step) || typeid_cast<FilterStep *>(step) || typeid_cast<ArrayJoinStep *>(step))
        return findReadingStep(node->children.front());

    return nullptr;
}

/// FixedColumns are columns which values become constants after filtering.
/// In a query "SELECT x, y, z FROM table WHERE x = 1 AND y = 'a' ORDER BY x, y, z"
/// Fixed columns are 'x' and 'y'.
using FixedColumns = std::unordered_set<const ActionsDAG::Node *>;

/// Right now we find only simple cases like 'and(..., and(..., and(column = value, ...), ...'
void appendFixedColumnsFromFilterExpression(const ActionsDAG::Node & filter_expression, FixedColumns & fiexd_columns)
{
    std::stack<const ActionsDAG::Node *> stack;
    stack.push(&filter_expression);

    while (!stack.empty())
    {
        const auto * node = stack.top();
        stack.pop();
        if (node->type == ActionsDAG::ActionType::FUNCTION)
        {
            const auto & name = node->function_base->getName();
            if (name == "and")
            {
                for (const auto * arg : node->children)
                    stack.push(arg);
            }
            else if (name == "equals")
            {
                const ActionsDAG::Node * maybe_fixed_column = nullptr;
                bool is_singe = true;
                for (const auto & child : node->children)
                {
                    if (!child->column)
                    {
                        if (maybe_fixed_column)
                            maybe_fixed_column = child;
                        else
                            is_singe = false;
                    }
                }

                if (maybe_fixed_column && is_singe)
                    fiexd_columns.insert(maybe_fixed_column);
            }
        }
    }
}

void appendExpression(ActionsDAGPtr & dag, const ActionsDAGPtr & expression)
{
    if (dag)
        dag->mergeInplace(std::move(*expression->clone()));
    else
        dag = expression->clone();
}

void buildSortingDAG(QueryPlan::Node * node, ActionsDAGPtr & dag, FixedColumns & fixed_columns)
{
    IQueryPlanStep * step = node->step.get();
    if (auto * reading = typeid_cast<ReadFromMergeTree *>(step))
        return;

    if (node->children.size() != 1)
        return;

    buildSortingDAG(node->children.front(), dag, fixed_columns);

    if (auto * expression = typeid_cast<ExpressionStep *>(step))
        appendExpression(dag, expression->getExpression());

    if (auto * filter = typeid_cast<FilterStep *>(step))
    {
        appendExpression(dag, filter->getExpression());
        if (const auto * filter_expression = dag->tryFindInOutputs(filter->getFilterColumnName()))
            appendFixedColumnsFromFilterExpression(*filter_expression, fixed_columns);
    }

    if (auto * array_join = typeid_cast<ArrayJoinStep *>(step))
    {
        const auto & array_joined_columns = array_join->arrayJoin()->columns;

        /// Remove array joined columns from outputs.
        /// Types are changed after ARRAY JOIN, and we can't use this columns anyway.
        ActionsDAG::NodeRawConstPtrs outputs;
        outputs.reserve(dag->getOutputs().size());

        for (const auto & output : dag->getOutputs())
        {
            if (!array_joined_columns.contains(output->result_name))
                outputs.push_back(output);
        }
    }
}

void enreachFixedColumns(ActionsDAGPtr & dag, FixedColumns & fixed_columns)
{
    struct Frame
    {
        const ActionsDAG::Node * node;
        size_t next_child = 0;
    };

    std::stack<Frame> stack;
    std::unordered_set<const ActionsDAG::Node *> visited;
    for (const auto & node : dag->getNodes())
    {
        if (visited.contains(&node))
            continue;

        stack.push({&node});
        visited.insert(&node);
        while (!stack.empty())
        {
            auto & frame = stack.top();
            for (; frame.next_child < frame.node->children.size(); ++frame.next_child)
                if (!visited.contains(frame.node->children[frame.next_child]))
                    break;

            if (frame.next_child < frame.node->children.size())
            {
                const auto * child = frame.node->children[frame.next_child];
                visited.insert(child);
                stack.push({child});
                ++frame.next_child;
            }
            else
            {
                /// Ignore constants here, will check them separately
                if (!frame.node->column)
                {
                    if (frame.node->type == ActionsDAG::ActionType::ALIAS)
                    {
                        if (fixed_columns.contains(frame.node->children.at(0)))
                            fixed_columns.insert(frame.node);
                    }
                    else if (frame.node->type == ActionsDAG::ActionType::FUNCTION)
                    {
                        if (frame.node->function_base->isDeterministicInScopeOfQuery())
                        {
                            bool all_args_fixed_or_const = true;
                            for (const auto * child : frame.node->children)
                                if (!child->column || !fixed_columns.contains(child))
                                    all_args_fixed_or_const = false;

                            if (all_args_fixed_or_const)
                                fixed_columns.insert(frame.node);
                        }
                    }
                }
            }
        }
    }
}

/// Here we try to find inner DAG inside outer DAG.
/// Build a map: inner.nodes -> outer.nodes.
// using NodesMap = std::unordered_map<const ActionsDAG::Node *, const ActionsDAG::Node *>;
int isMonotonicSubtree(const ActionsDAG::Node * inner, const ActionsDAG::Node * outer)
{
    using Parents = std::set<const ActionsDAG::Node *>;
    std::unordered_map<const ActionsDAG::Node *, Parents> inner_parents;
    std::unordered_map<std::string_view, const ActionsDAG::Node *> inner_inputs;

    {
        std::stack<const ActionsDAG::Node *> stack;
        stack.push(inner);
        inner_parents.emplace(inner, Parents());
        while (!stack.empty())
        {
            const auto * node = stack.top();
            stack.pop();

            if (node->type == ActionsDAG::ActionType::INPUT)
                inner_inputs.emplace(node->result_name, node);

            for (const auto * child : node->children)
            {
                auto [it, inserted] = inner_parents.emplace(child, Parents());
                it->second.emplace(node);

                if (inserted)
                    stack.push(child);
            }
        }
    }

    std::unordered_map<const ActionsDAG::Node *, const ActionsDAG::Node *> outer_to_inner;
    std::unordered_map<const ActionsDAG::Node *, int> direction;

    {
        struct Frame
        {
            const ActionsDAG::Node * node;
            ActionsDAG::NodeRawConstPtrs mapped_children;
            int direction = 1;
        };

        std::stack<Frame> stack;
        stack.push(Frame{outer, {}});
        while (!stack.empty())
        {
            auto & frame = stack.top();
            frame.mapped_children.reserve(frame.node->children.size());

            while (frame.mapped_children.size() < frame.node->children.size())
            {
                const auto * child = frame.node->children[frame.mapped_children.size()];
                auto it = outer_to_inner.find(child);
                if (it == outer_to_inner.end())
                {
                    stack.push(Frame{child, {}});
                    break;
                }
                frame.mapped_children.push_back(it->second);
            }

            if (frame.mapped_children.size() < frame.node->children.size())
                continue;

            if (frame.node->type == ActionsDAG::ActionType::INPUT)
            {
                const ActionsDAG::Node * mapped = nullptr;
                if (auto it = inner_inputs.find(frame.node->result_name); it != inner_inputs.end())
                    mapped = it->second;

                outer_to_inner.emplace(frame.node, mapped);
            }
            else if (frame.node->type == ActionsDAG::ActionType::ALIAS)
            {
                outer_to_inner.emplace(frame.node, frame.mapped_children.at(0));
            }
            else if (frame.node->type == ActionsDAG::ActionType::FUNCTION)
            {
                bool found_all_children = true;
                size_t num_found_inner_roots = 0;
                for (const auto * child : frame.mapped_children)
                {
                    if (!child)
                        found_all_children = false;
                    else if (child == inner)
                        ++num_found_inner_roots;
                }

                bool found_monotonic_wrapper = false;
                if (num_found_inner_roots == 1)
                {
                    if (frame.node->function_base->hasInformationAboutMonotonicity())
                    {
                        size_t num_const_args = 0;
                        const ActionsDAG::Node * monotonic_child = nullptr;
                        for (const auto * child : frame.node->children)
                        {
                            if (child->column)
                                ++num_const_args;
                            else
                                monotonic_child = child;
                        }

                        if (monotonic_child && num_const_args + 1 == frame.node->children.size())
                        {
                            auto info = frame.node->function_base->getMonotonicityForRange(*monotonic_child->result_type, {}, {});
                            if (info.is_always_monotonic)
                            {
                                found_monotonic_wrapper = true;
                                outer_to_inner[frame.node] = inner;

                                int cur_direction = info.is_positive ? 1 : -1;
                                auto it = direction.find(monotonic_child);
                                if (it != direction.end())
                                    cur_direction *= it->second;

                                direction[frame.node] = cur_direction;
                            }
                        }
                    }
                }

                if (!found_monotonic_wrapper && found_all_children && !frame.mapped_children.empty())
                {
                    Parents container;
                    Parents * intersection = &inner_parents[frame.mapped_children[0]];

                    if (frame.mapped_children.size() > 1)
                    {
                        std::vector<Parents *> other_parents;
                        other_parents.reserve(frame.mapped_children.size());
                        for (size_t i = 1; i < frame.mapped_children.size(); ++i)
                            other_parents.push_back(&inner_parents[frame.mapped_children[i]]);

                        for (const auto * parent : *intersection)
                        {
                            bool is_common = true;
                            for (const auto * set : other_parents)
                            {
                                if (!set->contains(parent))
                                {
                                    is_common = false;
                                    break;
                                }
                            }

                            if (is_common)
                                container.insert(parent);
                        }

                        intersection = &container;
                    }

                    if (!intersection->empty())
                    {
                        auto func_name = frame.node->function_base->getName();
                        for (const auto * parent : *intersection)
                            if (parent->type == ActionsDAG::ActionType::FUNCTION && func_name == parent->function_base->getName())
                                outer_to_inner[frame.node] = parent;
                    }
                }
            }

            stack.pop();
        }
    }

    if (outer_to_inner[outer] != inner)
        return 0;

    int res = 1;
    if (auto it = direction.find(outer); it != direction.end())
        res = it->second;

    return res;
}

SortDescription buildPrefixSortDescription(
    const FixedColumns & fixed_columns,
    const ActionsDAGPtr & dag,
    const SortDescription & description,
    const ActionsDAG & sorting_key_dag,
    const Names & sorting_key_columns,
    int & read_direction)
{
    SortDescription order_key_prefix_descr;
    order_key_prefix_descr.reserve(description.size());

    /// This is a result direction we will read from MergeTree
    ///  1 - in order,
    /// -1 - in reverse order,
    ///  0 - usual read, don't apply optimization
    ///
    /// So far, 0 means any direction is possible. It is ok for constant prefix.
    read_direction = 0;

    for (size_t i = 0, next_sort_key = 0; i < description.size() && next_sort_key < sorting_key_columns.size(); ++i)
    {
        const auto & sort_column = description[i];
        const auto & sorting_key_column = sorting_key_columns[next_sort_key];

        /// If required order depend on collation, it cannot be matched with primary key order.
        /// Because primary keys cannot have collations.
        if (sort_column.collator)
            return order_key_prefix_descr;

        /// Direction for current sort key.
        int current_direction = 0;

        if (!dag)
        {
            if (sort_column.column_name != sorting_key_column)
                return order_key_prefix_descr;

            current_direction = sort_column.direction;
            ++next_sort_key;
        }
        else
        {
            const ActionsDAG::Node * sort_node = dag->tryFindInOutputs(sort_column.column_name);
             /// It is possible when e.g. sort by array joined column.
            if (!sort_node)
                return order_key_prefix_descr;

            const ActionsDAG::Node * sort_column_node = sorting_key_dag.tryFindInOutputs(sorting_key_column);
            /// This should not happen.
            if (!sort_column_node)
                return order_key_prefix_descr;

            bool is_fixed_column = sort_node->column || fixed_columns.contains(sort_node);

            /// We try to find the match even if column is fixed. In this case, potentially more keys will match.
            /// Example: 'table (x Int32, y Int32) ORDER BY x + 1, y + 1'
            ///          'SELECT x, y FROM table WHERE x = 42 ORDER BY x + 1, y + 1'
            /// Here, 'x + 1' would be a fixed point. But it is reasonable to read-in-order.
            current_direction = isMonotonicSubtree(sort_column_node, sort_node) * sort_column.direction;

            if (current_direction == 0 || !is_fixed_column)
                return order_key_prefix_descr;

            if (current_direction)
                ++next_sort_key;

            if (is_fixed_column)
                current_direction = 0;
        }

        /// read_direction == 0 means we can choose any global direction.
        /// current_direction == 0 means current key if fixed and any direction is possible for it.
        if (current_direction && read_direction && current_direction != read_direction)
            break;

        read_direction = current_direction;
        order_key_prefix_descr.push_back(description[i]);
    }

    return order_key_prefix_descr;
}

void optimizeReadInOrder(QueryPlan::Node & node)
{
    if (node.children.size() != 1)
        return;

    auto * sorting = typeid_cast<SortingStep *>(node.step.get());
    if (!sorting)
        return;

    ReadFromMergeTree * reading = findReadingStep(node.children.front());
    if (!reading)
        return;

    const auto & sorting_key = reading->getStorageMetadata()->getSortingKey();
    if (sorting_key.column_names.empty())
        return;

    ActionsDAGPtr dag;
    FixedColumns fixed_columns;
    buildSortingDAG(node.children.front(), dag, fixed_columns);

    const auto & description = sorting->getSortDescription();
    const auto & sorting_key_columns = sorting_key.column_names;

    int read_direction = 0;
    auto prefix_description = buildPrefixSortDescription(
        fixed_columns,
        dag, description,
        sorting_key.expression->getActionsDAG(), sorting_key_columns,
        read_direction);

    /// It is possible that prefix_description is not empty, but read_direction is 0.
    /// It means that some prefix of sorting key matched, but it was constant.
    /// In this case, read-in-order is useless.
    if (read_direction == 0 || prefix_description.empty())
        return;

    auto limit = sorting->getLimit();

    auto order_info = std::make_shared<InputOrderInfo>(
        SortDescription{},
        std::move(prefix_description),
        read_direction, limit);

    reading->setQueryInfoInputOrderInfo(order_info);
    sorting->convertToFinishSorting(order_info->order_key_prefix_descr);
}

size_t tryReuseStorageOrderingForWindowFunctions(QueryPlan::Node * parent_node, QueryPlan::Nodes & /*nodes*/)
{
    /// Find the following sequence of steps, add InputOrderInfo and apply prefix sort description to
    /// SortingStep:
    /// WindowStep <- SortingStep <- [Expression] <- [SettingQuotaAndLimits] <- ReadFromMergeTree

    auto * window_node = parent_node;
    auto * window = typeid_cast<WindowStep *>(window_node->step.get());
    if (!window)
        return 0;
    if (window_node->children.size() != 1)
        return 0;

    auto * sorting_node = window_node->children.front();
    auto * sorting = typeid_cast<SortingStep *>(sorting_node->step.get());
    if (!sorting)
        return 0;
    if (sorting_node->children.size() != 1)
        return 0;

    auto * possible_read_from_merge_tree_node = sorting_node->children.front();

    if (typeid_cast<ExpressionStep *>(possible_read_from_merge_tree_node->step.get()))
    {
        if (possible_read_from_merge_tree_node->children.size() != 1)
            return 0;

        possible_read_from_merge_tree_node = possible_read_from_merge_tree_node->children.front();
    }

    auto * read_from_merge_tree = typeid_cast<ReadFromMergeTree *>(possible_read_from_merge_tree_node->step.get());
    if (!read_from_merge_tree)
    {
        return 0;
    }

    auto context = read_from_merge_tree->getContext();
    if (!context->getSettings().optimize_read_in_window_order)
    {
        return 0;
    }

    const auto & query_info = read_from_merge_tree->getQueryInfo();
    const auto * select_query = query_info.query->as<ASTSelectQuery>();

    ManyExpressionActions order_by_elements_actions;
    const auto & window_desc = window->getWindowDescription();

    for (const auto & actions_dag : window_desc.partition_by_actions)
    {
        order_by_elements_actions.emplace_back(
            std::make_shared<ExpressionActions>(actions_dag, ExpressionActionsSettings::fromContext(context, CompileExpressions::yes)));
    }

    for (const auto & actions_dag : window_desc.order_by_actions)
    {
        order_by_elements_actions.emplace_back(
            std::make_shared<ExpressionActions>(actions_dag, ExpressionActionsSettings::fromContext(context, CompileExpressions::yes)));
    }

    auto order_optimizer = std::make_shared<ReadInOrderOptimizer>(
            *select_query,
            order_by_elements_actions,
            window->getWindowDescription().full_sort_description,
            query_info.syntax_analyzer_result);

    read_from_merge_tree->setQueryInfoOrderOptimizer(order_optimizer);

    /// If we don't have filtration, we can pushdown limit to reading stage for optimizations.
    UInt64 limit = (select_query->hasFiltration() || select_query->groupBy()) ? 0 : InterpreterSelectQuery::getLimitForSorting(*select_query, context);

    auto order_info = order_optimizer->getInputOrder(
            query_info.projection ? query_info.projection->desc->metadata : read_from_merge_tree->getStorageMetadata(),
            context,
            limit);

    if (order_info)
    {
        read_from_merge_tree->setQueryInfoInputOrderInfo(order_info);
        sorting->convertToFinishSorting(order_info->order_key_prefix_descr);
    }

    return 0;
}

}
