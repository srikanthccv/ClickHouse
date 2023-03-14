#include <Analyzer/TableFunctionNode.h>

#include <IO/WriteBuffer.h>
#include <IO/WriteHelpers.h>
#include <IO/Operators.h>

#include <Storages/IStorage.h>

#include <Parsers/ASTFunction.h>

#include <Interpreters/Context.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

TableFunctionNode::TableFunctionNode(String table_function_name_)
    : IQueryTreeNode(children_size)
    , table_function_name(table_function_name_)
    , storage_id("system", "one")
{
    children[arguments_child_index] = std::make_shared<ListNode>();
}

void TableFunctionNode::resolve(TableFunctionPtr table_function_value, StoragePtr storage_value, ContextPtr context)
{
    table_function = std::move(table_function_value);
    storage = std::move(storage_value);
    storage_id = storage->getStorageID();
    storage_snapshot = storage->getStorageSnapshot(storage->getInMemoryMetadataPtr(), context);
}

const StorageID & TableFunctionNode::getStorageID() const
{
    if (!storage)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Table function node {} is not resolved", table_function_name);

    return storage_id;
}

const StorageSnapshotPtr & TableFunctionNode::getStorageSnapshot() const
{
    if (!storage)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Table function node {} is not resolved", table_function_name);

    return storage_snapshot;
}

void TableFunctionNode::dumpTreeImpl(WriteBuffer & buffer, FormatState & format_state, size_t indent) const
{
    buffer << std::string(indent, ' ') << "TABLE_FUNCTION id: " << format_state.getNodeId(this);

    if (hasAlias())
        buffer << ", alias: " << getAlias();

    buffer << ", table_function_name: " << table_function_name;

    if (table_expression_modifiers)
    {
        buffer << ", ";
        table_expression_modifiers->dump(buffer);
    }

    const auto & arguments = getArguments();
    if (!arguments.getNodes().empty())
    {
        buffer << '\n' << std::string(indent + 2, ' ') << "ARGUMENTS\n";
        arguments.dumpTreeImpl(buffer, format_state, indent + 4);
    }
}

bool TableFunctionNode::isEqualImpl(const IQueryTreeNode & rhs) const
{
    const auto & rhs_typed = assert_cast<const TableFunctionNode &>(rhs);
    if (table_function_name != rhs_typed.table_function_name)
        return false;

    if (storage && rhs_typed.storage)
        return storage_id == rhs_typed.storage_id;

    return table_expression_modifiers == rhs_typed.table_expression_modifiers;
}

void TableFunctionNode::updateTreeHashImpl(HashState & state) const
{
    state.update(table_function_name.size());
    state.update(table_function_name);

    if (storage)
    {
        auto full_name = storage_id.getFullNameNotQuoted();
        state.update(full_name.size());
        state.update(full_name);
    }

    if (table_expression_modifiers)
        table_expression_modifiers->updateTreeHash(state);
}

QueryTreeNodePtr TableFunctionNode::cloneImpl() const
{
    auto result = std::make_shared<TableFunctionNode>(table_function_name);

    result->storage = storage;
    result->storage_id = storage_id;
    result->storage_snapshot = storage_snapshot;
    result->table_expression_modifiers = table_expression_modifiers;

    return result;
}

ASTPtr TableFunctionNode::toASTImpl() const
{
    auto table_function_ast = std::make_shared<ASTFunction>();

    table_function_ast->name = table_function_name;

    const auto & arguments = getArguments();
    table_function_ast->children.push_back(arguments.toAST());
    table_function_ast->arguments = table_function_ast->children.back();

    return table_function_ast;
}

}
