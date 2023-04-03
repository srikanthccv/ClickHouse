#pragma once

#include <Functions/keyvaluepair/impl/Configuration.h>
#include <Functions/keyvaluepair/impl/StateHandler.h>
#include <Functions/keyvaluepair/impl/NeedleFactory.h>

#include <IO/ReadBufferFromMemory.h>
#include <IO/ReadHelpers.h>
#include <Columns/ColumnString.h>
#include <base/find_symbols.h>

#include <string_view>
#include <string>
#include <vector>

namespace DB
{

namespace extractKV
{

template <bool WITH_ESCAPING>
class StateHandlerImpl : public StateHandler
{
public:
    explicit StateHandlerImpl(Configuration configuration_)
        : configuration(std::move(configuration_))
    {
        NeedleFactory<WITH_ESCAPING> needle_factory;

        wait_needles = needle_factory.getWaitNeedles(configuration);
        read_needles = needle_factory.getReadNeedles(configuration);
        read_quoted_needles = needle_factory.getReadQuotedNeedles(configuration);
    }

    [[nodiscard]] NextState waitKey(std::string_view file) const
    {
        if (const auto * p = find_first_not_symbols_or_null(file, wait_needles))
        {
            const size_t character_position = p - file.begin();
            if (isQuotingCharacter(*p))
            {
                // +1 to skip quoting character
                return {character_position + 1u, State::READING_QUOTED_KEY};
            }
            else
            {
                return {character_position, State::READING_KEY};
            }
        }

        return {file.size(), State::END};
    }

    [[nodiscard]] NextState readKey(std::string_view file, auto & key) const
    {
        key.reset();

        size_t pos = 0;

        while (const auto * p = find_first_symbols_or_null({file.begin() + pos, file.end()}, read_needles))
        {
            auto character_position = p - file.begin();
            size_t next_pos = character_position + 1u;

            if (WITH_ESCAPING && isEscapeCharacter(*p))
            {
                if constexpr (WITH_ESCAPING)
                {
                    auto [parsed_successfully, escape_sequence_length] = consumeWithEscapeSequence(file, pos, character_position, key);
                    next_pos = character_position + escape_sequence_length;

                    if (!parsed_successfully)
                    {
                        return {next_pos, State::WAITING_KEY};
                    }
                }
            }
            else if (isKeyValueDelimiter(*p))
            {
                key.append(file.begin() + pos, file.begin() + character_position);

                return {next_pos, State::WAITING_VALUE};
            }
            else if (isPairDelimiter(*p))
            {
                return {next_pos, State::WAITING_KEY};
            }

            pos = next_pos;
        }

        return {file.size(), State::END};
    }

    [[nodiscard]] NextState readQuotedKey(std::string_view file, auto & key) const
    {
        key.reset();

        size_t pos = 0;

        while (const auto * p = find_first_symbols_or_null({file.begin() + pos, file.end()}, read_quoted_needles))
        {
            size_t character_position = p - file.begin();
            size_t next_pos = character_position + 1u;

            if (WITH_ESCAPING && isEscapeCharacter(*p))
            {
                if constexpr (WITH_ESCAPING)
                {
                    auto [parsed_successfully, escape_sequence_length] = consumeWithEscapeSequence(file, pos, character_position, key);
                    next_pos = character_position + escape_sequence_length;

                    if (!parsed_successfully)
                    {
                        return {next_pos, State::WAITING_KEY};
                    }
                }
            }
            else if (isQuotingCharacter(*p))
            {
                key.append(file.begin() + pos, file.begin() + character_position);

                if (key.isEmpty())
                {
                    return {next_pos, State::WAITING_KEY};
                }

                return {next_pos, State::READING_KV_DELIMITER};
            }

            pos = next_pos;
        }

        return {file.size(), State::END};
    }

    [[nodiscard]] NextState readKeyValueDelimiter(std::string_view file) const
    {
        if (!file.empty())
        {
            const auto current_character = file[0];

            if (isKeyValueDelimiter(current_character))
            {
                return {1, WAITING_VALUE};
            }
        }

        return {0, State::WAITING_KEY};
    }

    [[nodiscard]] NextState waitValue(std::string_view file) const
    {
        size_t pos = 0;

        if (!file.empty())
        {
            const auto current_character = file[pos];

            if (isQuotingCharacter(current_character))
            {
                return {pos + 1u, State::READING_QUOTED_VALUE};
            }
            else if (isKeyValueDelimiter(current_character))
            {
                return {pos, State::WAITING_KEY};
            }
        }

        return {pos, State::READING_VALUE};
    }

    [[nodiscard]] NextState readValue(std::string_view file, auto & value) const
    {
        value.reset();

        size_t pos = 0;

        while (const auto * p = find_first_symbols_or_null({file.begin() + pos, file.end()}, read_needles))
        {
            const size_t character_position = p - file.begin();
            size_t next_pos = character_position + 1u;

            if (WITH_ESCAPING && isEscapeCharacter(*p))
            {
                if constexpr (WITH_ESCAPING)
                {
                    auto [parsed_successfully, escape_sequence_length] = consumeWithEscapeSequence(file, pos, character_position, value);
                    next_pos = character_position + escape_sequence_length;

                    if (!parsed_successfully)
                    {
                        return {next_pos, State::WAITING_KEY};
                    }
                }
            }
            else if (isKeyValueDelimiter(*p))
            {
                return {next_pos, State::WAITING_KEY};
            }
            else if (isPairDelimiter(*p))
            {
                value.append(file.begin() + pos, file.begin() + character_position);

                return {next_pos, State::FLUSH_PAIR};
            }

            pos = next_pos;
        }

        // Reached end of input, consume rest of the file as value and make sure KV pair is produced.
        value.append(file.begin() + pos, file.end());
        return {file.size(), State::FLUSH_PAIR};
    }

    [[nodiscard]] NextState readQuotedValue(std::string_view file, auto & value) const
    {
        size_t pos = 0;

        value.reset();

        while (const auto * p = find_first_symbols_or_null({file.begin() + pos, file.end()}, read_quoted_needles))
        {
            const size_t character_position = p - file.begin();
            size_t next_pos = character_position + 1u;

            if (WITH_ESCAPING && isEscapeCharacter(*p))
            {
                if constexpr (WITH_ESCAPING)
                {
                    auto [parsed_successfully, escape_sequence_length] = consumeWithEscapeSequence(file, pos, character_position, value);
                    next_pos = character_position + escape_sequence_length;

                    if (!parsed_successfully)
                    {
                        return {next_pos, State::WAITING_KEY};
                    }
                }
            }
            else if (isQuotingCharacter(*p))
            {
                value.append(file.begin() + pos, file.begin() + character_position);

                return {next_pos, State::FLUSH_PAIR};
            }

            pos = next_pos;
        }

        return {file.size(), State::END};
    }

    const Configuration configuration;

private:
    SearchSymbols wait_needles;
    SearchSymbols read_needles;
    SearchSymbols read_quoted_needles;

    std::pair<bool, std::size_t> consumeWithEscapeSequence(std::string_view file, size_t start_pos, size_t character_pos, auto & output) const
    {
        std::string escaped_sequence;
        DB::ReadBufferFromMemory buf(file.begin() + character_pos, file.size() - character_pos);

        if (DB::parseComplexEscapeSequence(escaped_sequence, buf))
        {
            output.append(file.begin() + start_pos, file.begin() + character_pos);
            output.append(escaped_sequence);

            return {true, buf.getPosition()};
        }

        return {false, buf.getPosition()};
    }

    bool isKeyValueDelimiter(char character) const
    {
        return configuration.key_value_delimiter == character;
    }

    bool isPairDelimiter(char character) const
    {
        const auto & pair_delimiters = configuration.pair_delimiters;
        return std::find(pair_delimiters.begin(), pair_delimiters.end(), character) != pair_delimiters.end();
    }

    bool isQuotingCharacter(char character) const
    {
        return configuration.quoting_character == character;
    }

    bool isEscapeCharacter(char character) const
    {
        return character == '\\';
    }
};

struct NoEscapingStateHandler : public StateHandlerImpl<false>
{
    class StringWriter
    {
        ColumnString & col;

        std::string_view element;

    public:
        explicit StringWriter(ColumnString & col_)
            : col(col_)
        {}

        ~StringWriter()
        {
            // Make sure that ColumnString invariants are not broken.
            if (!isEmpty())
            {
                reset();
            }
        }

        void append(std::string_view new_data)
        {
            element = new_data;
        }

        template <typename T>
        void append(const T * begin, const T * end)
        {
            append({begin, end});
        }

        void reset()
        {
            element = {};
        }

        bool isEmpty() const
        {
            return element.empty();
        }

        void commit()
        {
            col.insertData(element.begin(), element.size());
            reset();
        }

        std::string_view uncommittedChunk() const
        {
            return element;
        }
    };

    template <typename ... Args>
    NoEscapingStateHandler(Args && ... args)
    : StateHandlerImpl<false>(std::forward<Args>(args)...) {}
};

struct InlineEscapingStateHandler: public StateHandlerImpl<true>
{
    class StringWriter
    {
        ColumnString & col;
        ColumnString::Chars & chars;
        UInt64 prev_commit_pos;

    public:
        explicit StringWriter(ColumnString & col_)
            : col(col_),
            chars(col.getChars()),
            prev_commit_pos(chars.size())
        {}

        ~StringWriter()
        {
            // Make sure that ColumnString invariants are not broken.
            if (!isEmpty())
            {
                reset();
            }
        }

        void append(std::string_view new_data)
        {
            chars.insert(new_data.begin(), new_data.end());
        }

        template <typename T>
        void append(const T * begin, const T * end)
        {
            chars.insert(begin, end);
        }

        void reset()
        {
            chars.resize_assume_reserved(prev_commit_pos);
        }

        bool isEmpty() const
        {
            return chars.size() == prev_commit_pos;
        }

        void commit()
        {
            col.insertData(nullptr, 0);
            prev_commit_pos = chars.size();
        }

        std::string_view uncommittedChunk() const
        {
            return std::string_view(chars.raw_data() + prev_commit_pos, chars.raw_data() + chars.size());
        }
    };

    template <typename ... Args>
    InlineEscapingStateHandler(Args && ... args)
        : StateHandlerImpl<true>(std::forward<Args>(args)...) {}
};

}

}
