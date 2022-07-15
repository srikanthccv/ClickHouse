#pragma once

#include <re2/re2.h>

#include <Analyzer/Identifier.h>
#include <Analyzer/IQueryTreeNode.h>
#include <Analyzer/ListNode.h>

namespace DB
{

/** Transformers are query tree nodes that handle additional logic that you can apply after MatcherQueryTreeNode is resolved.
  * Check MatcherQueryTreeNode.h before reading this documentation.
  *
  * They main purpose it to apply some logic for expressions after matcher is resolved.
  * There are 3 types of transformers:
  *
  * 1. APPLY transformer:
  * APPLY transformer transform expression using lambda or function into another expression.
  * It has 2 syntax variants:
  *     1. lambda variant: SELECT matcher APPLY (x -> expr(x)).
  *     2. function variant: SELECT matcher APPLY function_name(optional_parameters).
  *
  * 2. EXCEPT transformer:
  * EXCEPT transformer discard some columns.
  * It has 2 syntax variants:
  *     1. regexp variant: SELECT matcher EXCEPT ('regexp').
  *     2. column names list variant: SELECT matcher EXCEPT (column_name_1, ...).
  *
  * 3. REPLACE transfomer:
  * REPLACE transformer applies similar transformation as APPLY transformer, but only for expressions
  * that match replacement expression name.
  *
  * Example:
  * CREATE TABLE test_table (id UInt64) ENGINE=TinyLog;
  * SELECT * REPLACE (id + 1 AS id) FROM test_table.
  * This query is transformed into SELECT id + 1 FROM test_table.
  * It is important that AS id is not alias, it is replacement name. id + 1 is replacement expression.
  *
  * REPLACE transformer cannot contain multiple replacements with same name.
  *
  * REPLACE transformer expression does not necessary include replacement column name.
  * Example:
  * SELECT * REPLACE (1 AS id) FROM test_table.
  *
  * REPLACE transformer expression does not throw exception if there are no columns to apply replacement.
  * Example:
  * SELECT * REPLACE (1 AS unknown_column) FROM test_table;
  *
  * REPLACE transform can contain multiple replacements.
  * Example:
  * SELECT * REPLACE (1 AS id, 2 AS value).
  *
  * Matchers can be combined together and chained.
  * Example:
  * SELECT * EXCEPT (id) APPLY (x -> toString(x)) APPLY (x -> length(x)) FROM test_table.
  */

/// Column transformer type
enum class ColumnTransfomerType
{
    APPLY,
    EXCEPT,
    REPLACE
};

/// Get column transformer type name
const char * toString(ColumnTransfomerType type);

class IColumnTransformerNode;
using ColumnTransformerNodePtr = std::shared_ptr<IColumnTransformerNode>;
using ColumnTransformersNodes = std::vector<ColumnTransformerNodePtr>;

/// IColumnTransformer base interface.
class IColumnTransformerNode : public IQueryTreeNode
{
public:

    /// Get transformer type
    virtual ColumnTransfomerType getTransformerType() const = 0;

    /// Get transformer type name
    const char * getTransformerTypeName() const
    {
        return toString(getTransformerType());
    }

    QueryTreeNodeType getNodeType() const final
    {
        return QueryTreeNodeType::TRANSFORMER;
    }
};

enum class ApplyColumnTransformerType
{
    LAMBDA,
    FUNCTION
};

/// Get apply column transformer type name
const char * toString(ApplyColumnTransformerType type);

class ApplyColumnTransformerNode;
using ApplyColumnTransformerNodePtr = std::shared_ptr<ApplyColumnTransformerNode>;

/// Apply column transformer
class ApplyColumnTransformerNode final : public IColumnTransformerNode
{
public:
    /** Initialize apply column transformer with expression node.
      * Expression node must be lambda or function otherwise exception is throwned.
      */
    explicit ApplyColumnTransformerNode(QueryTreeNodePtr expression_node_);

    /// Get apply transformer type
    ApplyColumnTransformerType getApplyTransformerType() const
    {
        return apply_transformer_type;
    }

    /// Get apply transformer expression node
    const QueryTreeNodePtr & getExpressionNode() const
    {
        return children[expression_child_index];
    }

    ColumnTransfomerType getTransformerType() const override
    {
        return ColumnTransfomerType::APPLY;
    }

    void dumpTree(WriteBuffer & buffer, size_t indent) const override;

protected:
    bool isEqualImpl(const IQueryTreeNode & rhs) const override;

    void updateTreeHashImpl(IQueryTreeNode::HashState & hash_state) const override;

    ASTPtr toASTImpl() const override;

    QueryTreeNodePtr cloneImpl() const override;

private:
    ApplyColumnTransformerNode() = default;

    ApplyColumnTransformerType apply_transformer_type = ApplyColumnTransformerType::LAMBDA;
    static constexpr size_t expression_child_index = 0;
};

/// Except column transformer type
enum class ExceptColumnTransformerType
{
    REGEXP,
    COLUMN_LIST,
};

const char * toString(ExceptColumnTransformerType type);

class ExceptColumnTransformerNode;
using ExceptColumnTransformerNodePtr = std::shared_ptr<ExceptColumnTransformerNode>;

/** Except column transformer
  * Strict column transformer must use all column names during matched nodes transformation.
  *
  * Example:
  * CREATE TABLE test_table (id UInt64, value String) ENGINE=TinyLog;
  * SELECT * EXCEPT STRICT (id, value1) FROM test_table;
  * Such query will throw exception because column name with value1 was not matched by strict EXCEPT transformer.
  *
  * Strict is valid only for EXCEPT COLUMN_LIST transformer.
  */
class ExceptColumnTransformerNode final : public IColumnTransformerNode
{
public:
    /// Initialize except column transformer with column names
    explicit ExceptColumnTransformerNode(Names except_column_names_, bool is_strict_)
        : except_transformer_type(ExceptColumnTransformerType::COLUMN_LIST)
        , is_strict(is_strict_)
        , except_column_names(std::move(except_column_names_))
    {
    }

    /// Initialize except column transformer with regexp column matcher
    explicit ExceptColumnTransformerNode(std::shared_ptr<re2::RE2> column_matcher_)
        : except_transformer_type(ExceptColumnTransformerType::REGEXP)
        , is_strict(false)
        , column_matcher(std::move(column_matcher_))
    {
    }

    /// Get except transformer type
    ExceptColumnTransformerType getExceptTransformerType() const
    {
        return except_transformer_type;
    }

    /** Get is except transformer strict.
      * Valid only for EXCEPT COLUMN_LIST transformer.
      */
    bool isStrict() const
    {
        return is_strict;
    }

    /// Returns true if except transformer match column name, false otherwise.
    bool isColumnMatching(const std::string & column_name) const;

    /** Get except column names.
      * Valid only for column list except transformer.
      */
    const Names & getExceptColumnNames() const
    {
        return except_column_names;
    }

    ColumnTransfomerType getTransformerType() const override
    {
        return ColumnTransfomerType::EXCEPT;
    }

    void dumpTree(WriteBuffer & buffer, size_t indent) const override;

protected:
    bool isEqualImpl(const IQueryTreeNode & rhs) const override;

    void updateTreeHashImpl(IQueryTreeNode::HashState & hash_state) const override;

    ASTPtr toASTImpl() const override;

    QueryTreeNodePtr cloneImpl() const override;

private:
    ExceptColumnTransformerType except_transformer_type;
    bool is_strict;
    Names except_column_names;
    std::shared_ptr<re2::RE2> column_matcher;
};

class ReplaceColumnTransformerNode;
using ReplaceColumnTransformerNodePtr = std::shared_ptr<ReplaceColumnTransformerNode>;

/** Replace column transformer
  * Strict replace column transformer must use all replacements during matched nodes transformation.
  *
  * Example:
  * REATE TABLE test_table (id UInt64, value String) ENGINE=TinyLog;
  * SELECT * REPLACE STRICT (1 AS id, 2 AS value_1) FROM test_table;
  * Such query will throw exception because column name with value1 was not matched by strict REPLACE transformer.
  */
class ReplaceColumnTransformerNode final : public IColumnTransformerNode
{
public:
    /// Replacement is column name and replace expression
    struct Replacement
    {
        std::string column_name;
        QueryTreeNodePtr expression_node;
    };

    /// Initialize replace column transformer with replacements
    explicit ReplaceColumnTransformerNode(const std::vector<Replacement> & replacements_, bool is_strict);

    ColumnTransfomerType getTransformerType() const override
    {
        return ColumnTransfomerType::REPLACE;
    }

    /// Is replace column transformer strict
    bool isStrict() const
    {
        return is_strict;
    }

    /// Get replacements
    ListNode & getReplacements() const
    {
        return children[replacements_child_index]->as<ListNode &>();
    }

    /// Get replacements node
    const QueryTreeNodePtr & getReplacementsNode() const
    {
        return children[replacements_child_index];
    }

    /// Get replacements names
    const Names & getReplacementsNames() const
    {
        return replacements_names;
    }

    /** Returns replacement expression if for expression name replacements exists, nullptr otherwise.
      * Returned replacement expression must be cloned by caller.
      */
    QueryTreeNodePtr findReplacementExpression(const std::string & expression_name);

    void dumpTree(WriteBuffer & buffer, size_t indent) const override;

protected:
    bool isEqualImpl(const IQueryTreeNode & rhs) const override;

    void updateTreeHashImpl(IQueryTreeNode::HashState & hash_state) const override;

    ASTPtr toASTImpl() const override;

    QueryTreeNodePtr cloneImpl() const override;

private:
    ReplaceColumnTransformerNode() = default;

    bool is_strict;
    Names replacements_names;
    static constexpr size_t replacements_child_index = 0;
};

}
