#include <Parsers/TokenIterator.h>
#include <unordered_set>
#include <base/types.h>

namespace DB
{

Tokens::Tokens(const char * begin, const char * end, size_t max_query_size, bool skip_insignificant)
{
    Lexer lexer(begin, end, max_query_size);

    bool stop = false;
    do
    {
        Token token = lexer.nextToken();
        stop = token.isEnd() || token.type == TokenType::ErrorMaxQuerySizeExceeded;
        if (token.isSignificant() || (!skip_insignificant && !data.empty() && data.back().isSignificant()))
            data.emplace_back(std::move(token));
    } while (!stop);
}

UnmatchedParentheses checkUnmatchedParentheses(TokenIterator begin)
{
    std::unordered_set<String> valid_kql_negative_suffix
    ({
        "contains",
        "contains_cs",
        "endswith",
        "endswith_cs",
        "~",
        "=",
        "has",
        "has_cs",
        "hasprefix",
        "hasprefix_cs",
        "hassuffix",
        "hassuffix_cs",
        "in",
        "startswith",
        "startswith_cs"
    });
    /// We have just two kind of parentheses: () and [].
    UnmatchedParentheses stack;

    /// We have to iterate through all tokens until the end to avoid false positive "Unmatched parentheses" error
    /// when parser failed in the middle of the query.
    bool is_kql_included = false;
    for (TokenIterator it = begin; !it->isEnd() ; ++it)
    {
        //for kql function in sql like : select * from kql(Customers | where FirstName !in ("test", "test2"));
        if (String(it.get().begin, it.get().end) == "kql")
        {
            ++it;
            if (it->type == TokenType::OpeningRoundBracket)
                is_kql_included = true;
            --it;
        }

        if (!it.isValid())
        {
            if (!is_kql_included)
                break;

            if (it->type == TokenType::ErrorSingleExclamationMark)
             {
                 ++it;
                 if (valid_kql_negative_suffix.find(String(it.get().begin, it.get().end)) == valid_kql_negative_suffix.end())
                    break;
                --it;
             }
        }

        if (it->type == TokenType::OpeningRoundBracket || it->type == TokenType::OpeningSquareBracket)
        {
            stack.push_back(*it);
        }
        else if (it->type == TokenType::ClosingRoundBracket || it->type == TokenType::ClosingSquareBracket)
        {
            if (stack.empty())
            {
                /// Excessive closing bracket.
                stack.push_back(*it);
                return stack;
            }
            else if ((stack.back().type == TokenType::OpeningRoundBracket && it->type == TokenType::ClosingRoundBracket)
                || (stack.back().type == TokenType::OpeningSquareBracket && it->type == TokenType::ClosingSquareBracket))
            {
                /// Valid match.
                stack.pop_back();
            }
            else
            {
                /// Closing bracket type doesn't match opening bracket type.
                stack.push_back(*it);
                return stack;
            }
        }
    }

    /// If stack is not empty, we have unclosed brackets.
    return stack;
}

}
