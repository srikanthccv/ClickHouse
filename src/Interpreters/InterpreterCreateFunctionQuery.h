#pragma once

#include <Interpreters/IInterpreter.h>
#include <Parsers/ASTCreateFunctionQuery.h>

namespace DB
{

class ASTCreateFunctionQuery;
class Context;

class InterpreterCreateFunctionQuery : public IInterpreter, WithContext
{
public:
    InterpreterCreateFunctionQuery(const ASTPtr & query_ptr_, ContextPtr context_, bool is_internal_)
        : WithContext(context_)
        , query_ptr(query_ptr_)
        , is_internal(is_internal_) {}

    BlockIO execute() override;

    void setInternal(bool internal_);

private:
    static void validateFunction(ASTPtr function, const String & name);
    static void getIdentifiers(ASTPtr node, std::set<String> & identifiers);
    static void validateFunctionRecursiveness(ASTPtr node, const String & function_to_create);

    ASTPtr query_ptr;
    bool is_internal;
};

}
