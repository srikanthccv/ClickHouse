#pragma once

#include <Interpreters/PreparedSets.h>
#include <Core/SortDescription.h>
#include <memory>

namespace DB
{

class ExpressionActions;
using ExpressionActionsPtr = std::shared_ptr<ExpressionActions>;

struct PrewhereInfo
{
    /// Actions which are executed in order to alias columns are used for prewhere actions.
    ExpressionActionsPtr alias_actions;
    /// Actions which are executed on block in order to get filter column for prewhere step.
    ExpressionActionsPtr prewhere_actions;
    /// Actions which are executed after reading from storage in order to remove unused columns.
    ExpressionActionsPtr remove_columns_actions;
    String prewhere_column_name;
    bool remove_prewhere_column = false;

    PrewhereInfo() = default;
    explicit PrewhereInfo(ExpressionActionsPtr prewhere_actions_, String prewhere_column_name_)
        : prewhere_actions(std::move(prewhere_actions_)), prewhere_column_name(std::move(prewhere_column_name_)) {}
};

/// Helper struct to store all the information about the filter expression.
struct FilterInfo
{
    ExpressionActionsPtr actions;
    String column_name;
    bool do_remove_column = false;
};

struct SortingInfo
{
    int direction;
    SortDescription prefix_order_descr;
    UInt64 limit = 0;
};

using PrewhereInfoPtr = std::shared_ptr<PrewhereInfo>;
using FilterInfoPtr = std::shared_ptr<FilterInfo>;
using SortingInfoPtr = std::shared_ptr<SortingInfo>;

struct SyntaxAnalyzerResult;
using SyntaxAnalyzerResultPtr = std::shared_ptr<const SyntaxAnalyzerResult>;

/** Query along with some additional data,
  *  that can be used during query processing
  *  inside storage engines.
  */
struct SelectQueryInfo
{
    ASTPtr query;

    SyntaxAnalyzerResultPtr syntax_analyzer_result;

    PrewhereInfoPtr prewhere_info;

    SortingInfoPtr sorting_info;

    /// If set to true, the query from MergeTree will return a set of streams,
    /// each of them will read data in sorted by sorting key order.
    // bool do_not_steal_task = false;
    // bool read_in_pk_order = false;
    // bool read_in_reverse_order = false;

    /// Prepared sets are used for indices by storage engine.
    /// Example: x IN (1, 2, 3)
    PreparedSets sets;
};

}
