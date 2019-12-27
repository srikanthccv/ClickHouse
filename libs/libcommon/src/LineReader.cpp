#include <common/LineReader.h>

#include <iostream>

namespace
{

/// Trim ending whitespace inplace
void trim(String & s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
}

}

LineReader::LineReader(const String & history_file_path_, char extender_, char delimiter_)
    : history_file_path(history_file_path_), extender(extender_), delimiter(delimiter_)
{
#ifdef USE_REPLXX
    if (!history_file_path.empty())
        rx.history_load(history_file_path);
#endif
    /// FIXME: check extender != delimiter
}

LineReader::~LineReader()
{
#ifdef USE_REPLXX
    if (!history_file_path.empty())
        rx.history_save(history_file_path);
#endif
}

String LineReader::readLine(const String & first_prompt, const String & second_prompt)
{
    String line;
    bool is_multiline = false;

    while (auto status = readOneLine(is_multiline ? second_prompt : first_prompt))
    {
        if (status == RESET_LINE)
        {
            line.clear();
            is_multiline = false;
            continue;
        }

        if (input.empty())
            continue;

        is_multiline = (input.back() == extender) || (delimiter && input.back() != delimiter);

        if (input.back() == extender)
        {
            input = input.substr(0, input.size() - 1);
            trim(input);
            if (input.empty())
                continue;
        }

        line += (line.empty() ? "" : " ") + input;

        if (!is_multiline)
        {
            if (line != prev_line)
            {
                addToHistory(line);
                prev_line = line;
            }

            return line;
        }
    }

    return {};
}

LineReader::InputStatus LineReader::readOneLine(const String & prompt)
{
    input.clear();

#ifdef USE_REPLXX
    const char* cinput = rx.input(prompt);
    if (cinput == nullptr)
        return (errno != EAGAIN) ? ABORT : RESET_LINE;
    input = cinput;
#else
    std::cout << prompt;
    std::getline(std::cin, input);
    if (!std::cin.good())
        return ABORT;
#endif

    trim(input);
    return INPUT_LINE;
}

void LineReader::addToHistory(const String & line)
{
#ifdef USE_REPLXX
    rx.history_add(line);
#endif
}
