#include <DataTypes/DataTypeFactory.h>
#include <Parsers/parseQuery.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTLiteral.h>
#include <Common/typeid_cast.h>
#include <Poco/String.h>
#include <Common/StringUtils/StringUtils.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int UNKNOWN_TYPE;
    extern const int ILLEGAL_SYNTAX_FOR_DATA_TYPE;
    extern const int UNEXPECTED_AST_STRUCTURE;
    extern const int DATA_TYPE_CANNOT_HAVE_ARGUMENTS;
}


DataTypePtr DataTypeFactory::get(const String & full_name) const
{
    ParserIdentifierWithOptionalParameters parser;
    ASTPtr ast = parseQuery(parser, full_name.data(), full_name.data() + full_name.size(), "data type", 0);
    return get(ast);
}

DataTypePtr DataTypeFactory::get(const ASTPtr & ast) const
{
    if (const ASTFunction * func = typeid_cast<const ASTFunction *>(ast.get()))
    {
        if (func->parameters)
            throw Exception("Data type cannot have multiple parenthesed parameters.", ErrorCodes::ILLEGAL_SYNTAX_FOR_DATA_TYPE);
        return get(func->name, func->arguments);
    }

    if (const ASTIdentifier * ident = typeid_cast<const ASTIdentifier *>(ast.get()))
    {
        return get(ident->name, {});
    }

    if (const ASTLiteral * lit = typeid_cast<const ASTLiteral *>(ast.get()))
    {
        if (lit->value.isNull())
            return get("Null", {});
    }

    throw Exception("Unexpected AST element for data type.", ErrorCodes::UNEXPECTED_AST_STRUCTURE);
}

DataTypePtr DataTypeFactory::get(const String & family_name_param, const ASTPtr & parameters) const
{
    String family_name = getAliasToOrName(family_name_param);

    if (endsWith(family_name, "WithDictionary"))
    {
        ASTPtr low_cardinality_params = std::make_shared<ASTExpressionList>();
        String param_name = family_name.substr(0, family_name.size() - strlen("WithDictionary"));
        if (parameters)
        {
            auto func = std::make_shared<ASTFunction>();
            func->name = param_name;
            func->arguments = parameters;
            low_cardinality_params->children.push_back(func);
        }
        else
            low_cardinality_params->children.push_back(std::make_shared<ASTIdentifier>(param_name));

        return get("LowCardinality", low_cardinality_params);
    }

    {
        DataTypesDictionary::const_iterator it = data_types.find(family_name);
        if (data_types.end() != it)
            return it->second(parameters);
    }

    String family_name_lowercase = Poco::toLower(family_name);

    {
        DataTypesDictionary::const_iterator it = case_insensitive_data_types.find(family_name_lowercase);
        if (case_insensitive_data_types.end() != it)
            return it->second(parameters);
    }

    throw Exception("Unknown data type family: " + family_name, ErrorCodes::UNKNOWN_TYPE);
}


void DataTypeFactory::registerDataType(const String & family_name, Creator creator, CaseSensitiveness case_sensitiveness)
{
    if (creator == nullptr)
        throw Exception("DataTypeFactory: the data type family " + family_name + " has been provided "
            " a null constructor", ErrorCodes::LOGICAL_ERROR);

    String family_name_lowercase = Poco::toLower(family_name);

    if (isAlias(family_name) || isAlias(family_name_lowercase))
        throw Exception("DataTypeFactory: the data type family name '" + family_name + "' is already registered as alias",
                        ErrorCodes::LOGICAL_ERROR);

    if (!data_types.emplace(family_name, creator).second)
        throw Exception("DataTypeFactory: the data type family name '" + family_name + "' is not unique",
            ErrorCodes::LOGICAL_ERROR);


    if (case_sensitiveness == CaseInsensitive
        && !case_insensitive_data_types.emplace(family_name_lowercase, creator).second)
        throw Exception("DataTypeFactory: the case insensitive data type family name '" + family_name + "' is not unique",
            ErrorCodes::LOGICAL_ERROR);
}

void DataTypeFactory::registerSimpleDataType(const String & name, SimpleCreator creator, CaseSensitiveness case_sensitiveness)
{
    if (creator == nullptr)
        throw Exception("DataTypeFactory: the data type " + name + " has been provided "
            " a null constructor", ErrorCodes::LOGICAL_ERROR);

    registerDataType(name, [name, creator](const ASTPtr & ast)
    {
        if (ast)
            throw Exception("Data type " + name + " cannot have arguments", ErrorCodes::DATA_TYPE_CANNOT_HAVE_ARGUMENTS);
        return creator();
    }, case_sensitiveness);
}

void registerDataTypeNumbers(DataTypeFactory & factory);
void registerDataTypeDate(DataTypeFactory & factory);
void registerDataTypeDateTime(DataTypeFactory & factory);
void registerDataTypeString(DataTypeFactory & factory);
void registerDataTypeFixedString(DataTypeFactory & factory);
void registerDataTypeEnum(DataTypeFactory & factory);
void registerDataTypeArray(DataTypeFactory & factory);
void registerDataTypeTuple(DataTypeFactory & factory);
void registerDataTypeNullable(DataTypeFactory & factory);
void registerDataTypeNothing(DataTypeFactory & factory);
void registerDataTypeUUID(DataTypeFactory & factory);
void registerDataTypeAggregateFunction(DataTypeFactory & factory);
void registerDataTypeNested(DataTypeFactory & factory);
void registerDataTypeInterval(DataTypeFactory & factory);
void registerDataTypeWithDictionary(DataTypeFactory & factory);


DataTypeFactory::DataTypeFactory()
{
    registerDataTypeNumbers(*this);
    registerDataTypeDate(*this);
    registerDataTypeDateTime(*this);
    registerDataTypeString(*this);
    registerDataTypeFixedString(*this);
    registerDataTypeEnum(*this);
    registerDataTypeArray(*this);
    registerDataTypeTuple(*this);
    registerDataTypeNullable(*this);
    registerDataTypeNothing(*this);
    registerDataTypeUUID(*this);
    registerDataTypeAggregateFunction(*this);
    registerDataTypeNested(*this);
    registerDataTypeInterval(*this);
    registerDataTypeWithDictionary(*this);
}

}
