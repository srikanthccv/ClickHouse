#include <IO/ReadBufferFromString.h>

#include <Interpreters/Context.h>
#include <Interpreters/executeQuery.h>
#include <Interpreters/InterpreterShowProcesslistQuery.h>

#include <Parsers/ASTQueryWithOutput.h>


namespace DB
{

BlockIO InterpreterShowProcesslistQuery::execute()
{
    getContext()->applySettingChange({"is_reinterpreted_execution", true});
    return executeQuery("SELECT * FROM system.processes", getContext(), true);
}

}
