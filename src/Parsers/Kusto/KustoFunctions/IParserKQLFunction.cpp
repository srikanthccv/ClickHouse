#include <Parsers/ASTExpressionList.h>
#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/IParserBase.h>
#include <Parsers/Kusto/KustoFunctions/IParserKQLFunction.h>
#include <Parsers/Kusto/KustoFunctions/KQLAggregationFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLBinaryFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLCastingFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLDateTimeFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLDynamicFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLFunctionFactory.h>
#include <Parsers/Kusto/KustoFunctions/KQLGeneralFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLIPFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLStringFunctions.h>
#include <Parsers/Kusto/KustoFunctions/KQLTimeSeriesFunctions.h>
#include <Parsers/Kusto/ParserKQLDateTypeTimespan.h>
#include <Parsers/Kusto/ParserKQLOperators.h>
#include <Parsers/Kusto/ParserKQLQuery.h>
#include <Parsers/Kusto/ParserKQLStatement.h>
#include <Parsers/ParserSetQuery.h>
#include <boost/lexical_cast.hpp>


#include <pcg_random.hpp>

#include <format>
#include <stack>

namespace DB::ErrorCodes
{
extern const int NOT_IMPLEMENTED;
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int SYNTAX_ERROR;
extern const int UNKNOWN_FUNCTION;
}

namespace
{
constexpr DB::TokenType determineClosingPair(const DB::TokenType token_type)
{
    if (token_type == DB::TokenType::OpeningCurlyBrace)
        return DB::TokenType::ClosingCurlyBrace;
    else if (token_type == DB::TokenType::OpeningRoundBracket)
        return DB::TokenType::ClosingRoundBracket;
    else if (token_type == DB::TokenType::OpeningSquareBracket)
        return DB::TokenType::ClosingSquareBracket;

    throw DB::Exception(DB::ErrorCodes::NOT_IMPLEMENTED, "Unhandled token: {}", magic_enum::enum_name(token_type));
}

constexpr bool isClosingBracket(const DB::TokenType token_type)
{
    return token_type == DB::TokenType::ClosingCurlyBrace || token_type == DB::TokenType::ClosingRoundBracket
        || token_type == DB::TokenType::ClosingSquareBracket;
}

constexpr bool isOpeningBracket(const DB::TokenType token_type)
{
    return token_type == DB::TokenType::OpeningCurlyBrace || token_type == DB::TokenType::OpeningRoundBracket
        || token_type == DB::TokenType::OpeningSquareBracket;
}
}

namespace DB
{
bool IParserKQLFunction::convert(String & out, IParser::Pos & pos)
{
    return wrapConvertImpl(
        pos,
        IncreaseDepthTag{},
        [&]
        {
            bool res = convertImpl(out, pos);
            if (!res)
                out = "";
            return res;
        });
}

bool IParserKQLFunction::directMapping(
    String & out, IParser::Pos & pos, const std::string_view ch_fn, const Interval & argument_count_interval)
{
    const auto fn_name = getKQLFunctionName(pos);
    if (fn_name.empty())
        return false;

    out.append(ch_fn.data(), ch_fn.length());
    out.push_back('(');

    int argument_count = 0;
    const auto begin = pos;
    while (!pos->isEnd() && pos->type != TokenType::PipeMark && pos->type != TokenType::Semicolon)
    {
        if (pos != begin)
            out.append(", ");

        if (const auto argument = getOptionalArgument(fn_name, pos))
        {
            ++argument_count;
            out.append(*argument);
        }

        if (pos->type == TokenType::ClosingRoundBracket)
        {
            if (!argument_count_interval.IsWithinBounds(argument_count))
                throw Exception(
                    ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                    "{}: between {} and {} arguments are expected, but {} were provided",
                    fn_name,
                    argument_count_interval.Min(),
                    argument_count_interval.Max(),
                    argument_count);

            out.push_back(')');
            return true;
        }
    }

    out.clear();
    pos = begin;
    return false;
}

String IParserKQLFunction::generateUniqueIdentifier()
{
    static pcg32_unique unique_random_generator;
    return std::to_string(unique_random_generator());
}

String IParserKQLFunction::getArgument(const String & function_name, DB::IParser::Pos & pos, const ArgumentState argument_state)
{
    if (auto optional_argument = getOptionalArgument(function_name, pos, argument_state))
        return std::move(*optional_argument);

    throw Exception(std::format("Required argument was not provided in {}", function_name), ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
}

String IParserKQLFunction::getConvertedArgument(const String & fn_name, IParser::Pos & pos)
{
    String converted_arg;
    std::vector<String> tokens;
    std::unique_ptr<IParserKQLFunction> fun;

    if (pos->type == TokenType::ClosingRoundBracket || pos->type == TokenType::ClosingSquareBracket)
        return converted_arg;

    if (pos->isEnd() || pos->type == TokenType::PipeMark || pos->type == TokenType::Semicolon)
        throw Exception("Need more argument(s) in function: " + fn_name, ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    while (!pos->isEnd() && pos->type != TokenType::PipeMark && pos->type != TokenType::Semicolon)
    {
        String new_token;
        if (!KQLOperators().convert(tokens, pos))
        {
            if (pos->type == TokenType::BareWord)
            {
                tokens.push_back(IParserKQLFunction::getExpression(pos));
            }
            else if (
                pos->type == TokenType::Comma || pos->type == TokenType::ClosingRoundBracket
                || pos->type == TokenType::ClosingSquareBracket)
            {
                break;
            }
            else
            {
                String token;
                if (pos->type == TokenType::QuotedIdentifier)
                    token = "'" + String(pos->begin + 1, pos->end - 1) + "'";
                else if (pos->type == TokenType::OpeningSquareBracket)
                {
                    ++pos;
                    String array_index;
                    while (!pos->isEnd() && pos->type != TokenType::ClosingSquareBracket)
                    {
                        array_index += getExpression(pos);
                        ++pos;
                    }
                    token = std::format("[ {0} >=0 ? {0} + 1 : {0}]", array_index);
                }
                else
                    token = String(pos->begin, pos->end);

                tokens.push_back(token);
            }
        }
        ++pos;
        if (pos->type == TokenType::Comma || pos->type == TokenType::ClosingRoundBracket || pos->type == TokenType::ClosingSquareBracket)
            break;
    }
    for (auto const & token : tokens)
        converted_arg = converted_arg.empty() ? token : converted_arg + " " + token;

    return converted_arg;
}

std::optional<String>
IParserKQLFunction::getOptionalArgument(const String & function_name, DB::IParser::Pos & pos, const ArgumentState argument_state)
{
    if (const auto type = pos->type; type != DB::TokenType::Comma && type != DB::TokenType::OpeningRoundBracket)
        return {};

    ++pos;
    if (const auto type = pos->type; type == DB::TokenType::ClosingRoundBracket || type == DB::TokenType::ClosingSquareBracket)
        return {};

    if (argument_state == ArgumentState::Parsed)
        return getConvertedArgument(function_name, pos);

    if (argument_state != ArgumentState::Raw)
        throw Exception(
            ErrorCodes::NOT_IMPLEMENTED,
            "Argument extraction is not implemented for {}::{}",
            magic_enum::enum_type_name<ArgumentState>(),
            magic_enum::enum_name(argument_state));

    String expression;
    std::stack<DB::TokenType> scopes;
    while (!pos->isEnd() && (!scopes.empty() || (pos->type != DB::TokenType::Comma && pos->type != DB::TokenType::ClosingRoundBracket)))
    {
        const auto token_type = pos->type;
        if (isOpeningBracket(token_type))
            scopes.push(token_type);
        else if (isClosingBracket(token_type))
        {
            if (scopes.empty() || determineClosingPair(scopes.top()) != token_type)
                throw Exception(
                    DB::ErrorCodes::SYNTAX_ERROR, "Unmatched token: {} when parsing {}", magic_enum::enum_name(token_type), function_name);

            scopes.pop();
        }

        if (token_type == DB::TokenType::QuotedIdentifier)
        {
            expression.push_back('\'');
            expression.append(pos->begin + 1, pos->end - 1);
            expression.push_back('\'');
        }
        else
            expression.append(pos->begin, pos->end);

        ++pos;
    }

    return expression;
}

String IParserKQLFunction::getKQLFunctionName(IParser::Pos & pos)
{
    String fn_name = String(pos->begin, pos->end);
    ++pos;
    if (pos->type != TokenType::OpeningRoundBracket)
    {
        --pos;
        return "";
    }
    return fn_name;
}

String IParserKQLFunction::kqlCallToExpression(
    const std::string_view function_name, const std::initializer_list<const std::string_view> params, const uint32_t max_depth)
{
    return kqlCallToExpression(function_name, std::span(params), max_depth);
}

String IParserKQLFunction::kqlCallToExpression(
    const std::string_view function_name, const std::span<const std::string_view> params, const uint32_t max_depth)
{
    const auto params_str = std::accumulate(
        std::cbegin(params),
        std::cend(params),
        String(),
        [](String acc, const std::string_view param)
        {
            if (!acc.empty())
                acc.append(", ");

            acc.append(param.data(), param.length());
            return acc;
        });

    const auto kql_call = std::format("{}({})", function_name, params_str);
    DB::Tokens call_tokens(kql_call.c_str(), kql_call.c_str() + kql_call.length());
    DB::IParser::Pos tokens_pos(call_tokens, max_depth);
    return DB::IParserKQLFunction::getExpression(tokens_pos);
}

void IParserKQLFunction::validateEndOfFunction(const String & fn_name, IParser::Pos & pos)
{
    if (pos->type != TokenType::ClosingRoundBracket)
        throw Exception("Too many arguments in function: " + fn_name, ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
}

String IParserKQLFunction::getExpression(IParser::Pos & pos)
{
    String arg = String(pos->begin, pos->end);
    if (pos->type == TokenType::BareWord)
    {
        String new_arg;
        auto fun = KQLFunctionFactory::get(arg);
        if (fun && fun->convert(new_arg, pos))
        {
            validateEndOfFunction(arg, pos);
            arg = new_arg;
        }
        else
        {
            if (!fun)
            {
                ++pos;
                if (pos->type == TokenType::OpeningRoundBracket)
                {
                    if (Poco::toLower(arg) != "and" && Poco::toLower(arg) != "or")
                        throw Exception(arg + " is not a supported kusto function", ErrorCodes::UNKNOWN_FUNCTION);
                }
                --pos;
            }
            ParserKQLDateTypeTimespan time_span;
            ASTPtr node;
            Expected expected;

            if (time_span.parse(pos, node, expected))
                arg = boost::lexical_cast<std::string>(time_span.toSeconds());
        }
    }
    else if (pos->type == TokenType::QuotedIdentifier)
        arg = "'" + String(pos->begin + 1, pos->end - 1) + "'";
    else if (pos->type == TokenType::OpeningSquareBracket)
    {
        ++pos;
        String array_index;
        while (!pos->isEnd() && pos->type != TokenType::ClosingSquareBracket)
        {
            array_index += getExpression(pos);
            ++pos;
        }
        arg = std::format("[ {0} >=0 ? {0} + 1 : {0}]", array_index);
    }

    return arg;
}

int IParserKQLFunction::getNullCounts(String arg)
{
    size_t index = 0;
    int null_counts = 0;
    for (char & i : arg)
    {
        if (i == 'n')
            i = 'N';
        if (i == 'u')
            i = 'U';
        if (i == 'l')
            i = 'L';
    }
    while ((index = arg.find("NULL", index)) != std::string::npos)
    {
        index += 4;
        null_counts += 1;
    }
    return null_counts;
}

int IParserKQLFunction::IParserKQLFunction::getArrayLength(String arg)
{
    int array_length = 0;
    bool comma_found = false;
    for (char i : arg)
    {
        if (i == ',')
        {
            comma_found = true;
            array_length += 1;
        }
    }
    return comma_found ? array_length + 1 : 0;
}

String IParserKQLFunction::ArraySortHelper(String & out, IParser::Pos & pos, bool ascending)
{
    String fn_name = getKQLFunctionName(pos);
    if (fn_name.empty())
        return "false";

    String reverse;
    String second_arg;
    String expr;

    if (!ascending)
        reverse = "Reverse";
    ++pos;
    String first_arg = getConvertedArgument(fn_name, pos);
    int null_count = getNullCounts(first_arg);
    if (pos->type == TokenType::Comma)
        ++pos;
    out = "array(";
    if (pos->type != TokenType::ClosingRoundBracket && String(pos->begin, pos->end) != "dynamic")
    {
        second_arg = getConvertedArgument(fn_name, pos);
        out += "if (" + second_arg + ", array" + reverse + "Sort(" + first_arg + "), concat(arraySlice(array" + reverse + "Sort("
            + first_arg + ") as as1, indexOf(as1, NULL) as len1), arraySlice(as1, 1, len1-1)))";
        out += " )";
        return out;
    }
    --pos;
    std::vector<String> argument_list;
    if (pos->type != TokenType::ClosingRoundBracket)
    {
        while (pos->type != TokenType::ClosingRoundBracket)
        {
            ++pos;
            if (String(pos->begin, pos->end) != "dynamic")
            {
                expr = getConvertedArgument(fn_name, pos);
                break;
            }
            second_arg = getConvertedArgument(fn_name, pos);
            argument_list.push_back(second_arg);
        }
    }
    else
    {
        ++pos;
        out += "array" + reverse + "Sort(" + first_arg + ")";
    }

    if (!argument_list.empty())
    {
        String temp_first_arg = first_arg;
        int first_arg_length = getArrayLength(temp_first_arg);

        if (null_count > 0 && expr.empty())
            expr = "true";
        if (null_count > 0)
            first_arg = "if (" + expr + ", array" + reverse + "Sort(" + first_arg + "), concat(arraySlice(array" + reverse + "Sort("
                + first_arg + ") as as1, indexOf(as1, NULL) as len1 ), arraySlice( as1, 1, len1-1) ) )";
        else
            first_arg = "array" + reverse + "Sort(" + first_arg + ")";

        out += first_arg;

        for (auto & i : argument_list)
        {
            out += " , ";
            if (first_arg_length != getArrayLength(i))
                out += "array(NULL)";
            else if (null_count > 0)
                out += "If (" + expr + "," + "array" + reverse + "Sort((x, y) -> y, " + i + "," + temp_first_arg
                    + "), arrayConcat(arraySlice(" + "array" + reverse + "Sort((x, y) -> y, " + i + "," + temp_first_arg
                    + ") , length(" + temp_first_arg + ") - " + std::to_string(null_count) + " + 1) , arraySlice(" + "array" + reverse
                    + "Sort((x, y) -> y, " + i + "," + temp_first_arg + ") , 1, length(" + temp_first_arg + ") - "
                    + std::to_string(null_count) + ") ) )";
            else
                out += "array" + reverse + "Sort((x, y) -> y, " + i + "," + temp_first_arg + ")";
        }
    }
    out += " )";
    return out;
}

}
