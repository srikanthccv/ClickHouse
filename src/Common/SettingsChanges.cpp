#include <Common/SettingsChanges.h>
#include <Parsers/formatAST.h>
#include <Common/FieldVisitorToString.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{
    SettingChange * find(SettingsChanges & changes, std::string_view name)
    {
        auto it = std::find_if(changes.begin(), changes.end(), [&name](const SettingChange & change) { return change.getName() == name; });
        if (it == changes.end())
            return nullptr;
        return &*it;
    }

    const SettingChange * find(const SettingsChanges & changes, std::string_view name)
    {
        auto it = std::find_if(changes.begin(), changes.end(), [&name](const SettingChange & change) { return change.getName() == name; });
        if (it == changes.end())
            return nullptr;
        return &*it;
    }
}

String SettingChange::getValueString() const
{
    if (ast_value)
        return serializeAST(*ast_value);
    return convertFieldToString(field_value);
}

const Field & SettingChange::getFieldValue() const
{
    throwIfASTValueNotConvertedToField();
    return field_value;
}

Field & SettingChange::getFieldValue()
{
    throwIfASTValueNotConvertedToField();
    return field_value;
}

void SettingChange::setFieldValue(const Field & field)
{
    field_value = field;
}

void SettingChange::setASTValue(const ASTPtr & ast)
{
    ast_value = ast ? ast->clone() : ast;
}

void SettingChange::throwIfASTValueNotConvertedToField() const
{
    if (getASTValue() != nullptr && field_value == Field{})
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "AST value of the setting must be converted to Field value");
}

bool SettingsChanges::tryGet(std::string_view name, Field & out_value) const
{
    const auto * change = find(*this, name);
    if (!change)
        return false;
    out_value = change->getFieldValue();
    return true;
}

const Field * SettingsChanges::tryGet(std::string_view name) const
{
    const auto * change = find(*this, name);
    if (!change)
        return nullptr;
    return &change->getFieldValue();
}

Field * SettingsChanges::tryGet(std::string_view name)
{
    auto * change = find(*this, name);
    if (!change)
        return nullptr;
    return &change->getFieldValue();
}

}
