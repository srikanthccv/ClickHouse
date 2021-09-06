#include <Common/StringUtils/StringUtils.h>
#include <Parsers/CommonParsers.h>
#include <common/find_symbols.h>
#include <IO/Operators.h>

#include <string.h>        /// strncmp, strncasecmp


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

bool ParserKeyword::parseImpl(Pos & pos, ASTPtr & /*node*/, Expected & expected)
{
    if (pos->type != TokenType::BareWord)
        return false;

    const char * current_word = s.begin();
    const char * s_end = s.end();
    const size_t s_length = s.length();

    if (!s_length)
        throw Exception("Logical error: keyword cannot be empty string", ErrorCodes::LOGICAL_ERROR);

    while (true)
    {
        expected.add(pos, current_word);
        if (pos->type != TokenType::BareWord)
            return false;

        const char * next_whitespace = find_first_symbols<' ', '\0'>(current_word, s_end);
        size_t word_length = next_whitespace - current_word;

        if (word_length != pos->size())
            return false;

        if (0 != strncasecmp(pos->begin, current_word, word_length))
            return false;

        ++pos;

        if (!*next_whitespace)
            break;

        current_word = next_whitespace + 1;
    }

    return true;
}

}
