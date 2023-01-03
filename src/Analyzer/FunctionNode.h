#pragma once

#include <memory>
#include <Core/IResolvedFunction.h>
#include <Analyzer/IQueryTreeNode.h>
#include <Analyzer/ListNode.h>
#include <Analyzer/ConstantValue.h>
#include <Common/typeid_cast.h>
#include <Core/ColumnsWithTypeAndName.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int UNSUPPORTED_METHOD;
}

class IFunctionOverloadResolver;
using FunctionOverloadResolverPtr = std::shared_ptr<IFunctionOverloadResolver>;

class IFunctionBase;
using FunctionBasePtr = std::shared_ptr<const IFunctionBase>;

class IAggregateFunction;
using AggregateFunctionPtr = std::shared_ptr<const IAggregateFunction>;

/** Function node represents function in query tree.
  * Function syntax: function_name(parameter_1, ...)(argument_1, ...).
  * If function does not have parameters its syntax is function_name(argument_1, ...).
  * If function does not have arguments its syntax is function_name().
  *
  * In query tree function parameters and arguments are represented by ListNode.
  *
  * Function can be:
  * 1. Aggregate function. Example: quantile(0.5)(x), sum(x).
  * 2. Non aggregate function. Example: plus(x, x).
  * 3. Window function. Example: sum(x) OVER (PARTITION BY expr ORDER BY expr).
  *
  * Initially function node is initialized with function name.
  * For window function client must initialize function window node.
  *
  * During query analysis pass function must be resolved using `resolveAsFunction`, `resolveAsAggregateFunction`, `resolveAsWindowFunction` methods.
  * Resolved function is function that has result type and is initialized with concrete aggregate or non aggregate function.
  */
class FunctionNode;
using FunctionNodePtr = std::shared_ptr<FunctionNode>;

enum class FunctionKind
{
    UNKNOWN,
    ORDINARY,
    AGGREGATE,
    WINDOW,
};

class FunctionNode final : public IQueryTreeNode
{
public:
    /** Initialize function node with function name.
      * Later during query analysis pass function must be resolved.
      */
    explicit FunctionNode(String function_name_);

    /// Get function name
    const String & getFunctionName() const
    {
        return function_name;
    }

    /// Get parameters
    const ListNode & getParameters() const
    {
        return children[parameters_child_index]->as<const ListNode &>();
    }

    /// Get parameters
    ListNode & getParameters()
    {
        return children[parameters_child_index]->as<ListNode &>();
    }

    /// Get parameters node
    const QueryTreeNodePtr & getParametersNode() const
    {
        return children[parameters_child_index];
    }

    /// Get parameters node
    QueryTreeNodePtr & getParametersNode()
    {
        return children[parameters_child_index];
    }

    /// Get arguments
    const ListNode & getArguments() const
    {
        return children[arguments_child_index]->as<const ListNode &>();
    }

    /// Get arguments
    ListNode & getArguments()
    {
        return children[arguments_child_index]->as<ListNode &>();
    }

    /// Get arguments node
    const QueryTreeNodePtr & getArgumentsNode() const
    {
        return children[arguments_child_index];
    }

    /// Get arguments node
    QueryTreeNodePtr & getArgumentsNode()
    {
        return children[arguments_child_index];
    }

    const DataTypes & getExpectedArgumentTypes() const;
    ColumnsWithTypeAndName getArgumentTypes() const;

    /// Returns true if function node has window, false otherwise
    bool hasWindow() const
    {
        return children[window_child_index] != nullptr;
    }

    /** Get window node.
      * Valid only for window function node.
      * Result window node can be identifier node or window node.
      * 1. It can be identifier node if window function is defined as expr OVER window_name.
      * 2. It can be window node if window function is defined as expr OVER (window_name ...).
      */
    const QueryTreeNodePtr & getWindowNode() const
    {
        return children[window_child_index];
    }

    /** Get window node.
      * Valid only for window function node.
      */
    QueryTreeNodePtr & getWindowNode()
    {
        return children[window_child_index];
    }

    /** Get non aggregate function.
      * If function is not resolved nullptr returned.
      */
    FunctionBasePtr getFunction() const
    {
        if (kind != FunctionKind::ORDINARY)
            return {};
        return std::reinterpret_pointer_cast<const IFunctionBase>(function);
    }

    /** Get aggregate function.
      * If function is not resolved nullptr returned.
      * If function is resolved as non aggregate function nullptr returned.
      */
    AggregateFunctionPtr getAggregateFunction() const
    {
        if (kind == FunctionKind::UNKNOWN || kind == FunctionKind::ORDINARY)
            return {};
        return std::reinterpret_pointer_cast<const IAggregateFunction>(function);
    }

    /// Is function node resolved
    bool isResolved() const
    {
        return function != nullptr;
    }

    /// Is function node window function
    bool isWindowFunction() const
    {
        return hasWindow();
    }

    /// Is function node aggregate function
    bool isAggregateFunction() const
    {
        return kind == FunctionKind::AGGREGATE;
    }

    /// Is function node ordinary function
    bool isOrdinaryFunction() const
    {
        return kind == FunctionKind::ORDINARY;
    }

    /** Resolve function node as non aggregate function.
      * It is important that function name is updated with resolved function name.
      * Main motivation for this is query tree optimizations.
      * Assume we have `multiIf` function with single condition, it can be converted to `if` function.
      * Function name must be updated accordingly.
      */
    void resolveAsFunction(FunctionBasePtr function_value);

    /** Resolve function node as aggregate function.
      * It is important that function name is updated with resolved function name.
      * Main motivation for this is query tree optimizations.
      */
    void resolveAsAggregateFunction(AggregateFunctionPtr aggregate_function_value);

    /** Resolve function node as window function.
      * It is important that function name is updated with resolved function name.
      * Main motivation for this is query tree optimizations.
      */
    void resolveAsWindowFunction(AggregateFunctionPtr window_function_value);

    QueryTreeNodeType getNodeType() const override
    {
        return QueryTreeNodeType::FUNCTION;
    }

    DataTypePtr getResultType() const override
    {
        if (!function)
            throw Exception(ErrorCodes::UNSUPPORTED_METHOD,
                "Function node with name '{}' is not resolved",
                function_name);
        return function->getResultType();
    }

    void dumpTreeImpl(WriteBuffer & buffer, FormatState & format_state, size_t indent) const override;

protected:
    bool isEqualImpl(const IQueryTreeNode & rhs) const override;

    void updateTreeHashImpl(HashState & hash_state) const override;

    QueryTreeNodePtr cloneImpl() const override;

    ASTPtr toASTImpl() const override;

private:
    String function_name;
    FunctionKind kind = FunctionKind::UNKNOWN;
    IResolvedFunctionPtr function;

    static constexpr size_t parameters_child_index = 0;
    static constexpr size_t arguments_child_index = 1;
    static constexpr size_t window_child_index = 2;
    static constexpr size_t children_size = window_child_index + 1;
};

}
