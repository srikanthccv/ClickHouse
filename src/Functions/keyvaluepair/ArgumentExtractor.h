#pragma once

#include <Columns/IColumn.h>
#include <Columns/ColumnsNumber.h>
#include <Core/ColumnsWithTypeAndName.h>

#include <optional>

namespace DB
{

/*
 * Validate (API level validation, no business logic validation) and extracts input arguments from
 * `ColumnsWithTypeAndName` into ArgumentExtractor::ParsedArguments.
 * */
class ArgumentExtractor
{
public:
    using CharArgument = std::optional<char>;
    using VectorArgument = std::vector<char>;

    struct ParsedArguments
    {
        ColumnPtr data_column;

        CharArgument key_value_delimiter = {};
        VectorArgument pair_delimiters = {};
        CharArgument quoting_character = {};
    };


    static ParsedArguments extract(const ColumnsWithTypeAndName & arguments);

private:
    static ArgumentExtractor::CharArgument extractControlCharacter(ColumnPtr column);
};

}
