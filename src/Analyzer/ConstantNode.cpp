#include <Analyzer/ConstantNode.h>

#include <Common/FieldVisitorToString.h>
#include <Common/SipHash.h>

#include <IO/WriteBuffer.h>
#include <IO/WriteHelpers.h>
#include <IO/Operators.h>

#include <DataTypes/FieldToDataType.h>

#include <Parsers/ASTLiteral.h>

namespace DB
{

ConstantNode::ConstantNode(ConstantValuePtr constant_value_)
    : constant_value(std::move(constant_value_))
    , value_string_dump(applyVisitor(FieldVisitorToString(), constant_value->getValue()))
{
}

ConstantNode::ConstantNode(Field value_, DataTypePtr value_data_type_)
    : ConstantNode(std::make_shared<ConstantValue>(std::move(value_), std::move(value_data_type_)))
{}

ConstantNode::ConstantNode(Field value_)
    : ConstantNode(std::make_shared<ConstantValue>(value_, applyVisitor(FieldToDataType(), value_)))
{}

void ConstantNode::dumpTreeImpl(WriteBuffer & buffer, FormatState & format_state, size_t indent) const
{
    buffer << std::string(indent, ' ') << "CONSTANT id: " << format_state.getNodeId(this);

    if (hasAlias())
        buffer << ", alias: " << getAlias();

    buffer << ", constant_value: " << constant_value->getValue().dump();
    buffer << ", constant_value_type: " << constant_value->getType()->getName();
}

bool ConstantNode::isEqualImpl(const IQueryTreeNode & rhs) const
{
    const auto & rhs_typed = assert_cast<const ConstantNode &>(rhs);
    return *constant_value == *rhs_typed.constant_value && value_string_dump == rhs_typed.value_string_dump;
}

void ConstantNode::updateTreeHashImpl(HashState & hash_state) const
{
    auto type_name = constant_value->getType()->getName();
    hash_state.update(type_name.size());
    hash_state.update(type_name);

    hash_state.update(value_string_dump.size());
    hash_state.update(value_string_dump);
}

ASTPtr ConstantNode::toASTImpl() const
{
    return std::make_shared<ASTLiteral>(constant_value->getValue());
}

QueryTreeNodePtr ConstantNode::cloneImpl() const
{
    return std::make_shared<ConstantNode>(constant_value);
}

}
