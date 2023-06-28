#include <Columns/ColumnConst.h>
#include <Columns/ColumnString.h>
#include <DataTypes/DataTypeString.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunction.h>
#include <Functions/PositionImpl.h>
#include <Interpreters/Context_fwd.h>
#include <base/find_symbols.h>
#include <Common/UTF8Helpers.h>
#include <Common/register_objects.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ZERO_ARRAY_OR_TUPLE_INDEX;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

namespace
{

    template <bool is_utf8>
    class FunctionSubstringIndex : public IFunction
    {
    public:
        static constexpr auto name = is_utf8 ? "substringIndexUTF8" : "substringIndex";


        static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionSubstringIndex>(); }

        String getName() const override { return name; }

        size_t getNumberOfArguments() const override { return 3; }

        bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

        bool useDefaultImplementationForConstants() const override { return true; }
        ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {1}; }

        DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
        {
            if (!isString(arguments[0]))
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of first argument of function {}",
                    arguments[0]->getName(),
                    getName());

            if (!isString(arguments[1]))
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of second argument of function {}",
                    arguments[1]->getName(),
                    getName());

            if (!isNativeNumber(arguments[2]))
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Illegal type {} of third argument of function {}",
                    arguments[2]->getName(),
                    getName());

            return std::make_shared<DataTypeString>();
        }

        ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t /*input_rows_count*/) const override
        {
            ColumnPtr column_string = arguments[0].column;
            ColumnPtr column_delim = arguments[1].column;
            ColumnPtr column_index = arguments[2].column;

            const ColumnConst * column_delim_const = checkAndGetColumnConst<ColumnString>(column_delim.get());
            if (!column_delim_const)
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Second argument to {} must be a constant String", getName());

            String delim = column_delim_const->getValue<String>();
            if constexpr (!is_utf8)
            {
                if (delim.size() != 1)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Second argument to {} must be a single character", getName());
            }
            else
            {
                if (UTF8::countCodePoints(reinterpret_cast<const UInt8 *>(delim.data()), delim.size()) != 1)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Second argument to {} must be a single UTF-8 character", getName());
            }

            auto column_res = ColumnString::create();
            ColumnString::Chars & vec_res = column_res->getChars();
            ColumnString::Offsets & offsets_res = column_res->getOffsets();

            const ColumnConst * column_string_const = checkAndGetColumnConst<ColumnString>(column_string.get());
            if (column_string_const)
            {
                String str = column_string_const->getValue<String>();
                constantVector(str, delim, column_index.get(), vec_res, offsets_res);
            }
            else
            {
                const auto * col_str = checkAndGetColumn<ColumnString>(column_string.get());
                if (!col_str)
                    throw Exception(ErrorCodes::ILLEGAL_COLUMN, "First argument to {} must be a String", getName());

                bool is_index_const = isColumnConst(*column_index);
                if (is_index_const)
                {
                    Int64 index = column_index->getInt(0);
                    vectorConstant(col_str, delim, index, vec_res, offsets_res);
                }
                else
                    vectorVector(col_str, delim, column_index.get(), vec_res, offsets_res);
            }
            return column_res;
        }

    protected:
        static void vectorVector(
            const ColumnString * str_column,
            const String & delim,
            const IColumn * index_column,
            ColumnString::Chars & res_data,
            ColumnString::Offsets & res_offsets)
        {
            size_t rows = str_column->size();
            res_data.reserve(str_column->getChars().size() / 2);
            res_offsets.reserve(rows);

            std::unique_ptr<PositionCaseSensitiveUTF8::SearcherInBigHaystack> searcher
                = !is_utf8 ? nullptr : std::make_unique<PositionCaseSensitiveUTF8::SearcherInBigHaystack>(delim.data(), delim.size());

            for (size_t i = 0; i < rows; ++i)
            {
                StringRef str_ref = str_column->getDataAt(i);
                Int64 index = index_column->getInt(i);
                StringRef res_ref
                    = !is_utf8 ? substringIndex(str_ref, delim[0], index) : substringIndexUTF8(searcher.get(), str_ref, delim, index);
                appendToResultColumn(res_ref, res_data, res_offsets);
            }
        }

        static void vectorConstant(
            const ColumnString * str_column,
            const String & delim,
            Int64 index,
            ColumnString::Chars & res_data,
            ColumnString::Offsets & res_offsets)
        {
            size_t rows = str_column->size();
            res_data.reserve(str_column->getChars().size() / 2);
            res_offsets.reserve(rows);

            std::unique_ptr<PositionCaseSensitiveUTF8::SearcherInBigHaystack> searcher
                = !is_utf8 ? nullptr : std::make_unique<PositionCaseSensitiveUTF8::SearcherInBigHaystack>(delim.data(), delim.size());

            for (size_t i = 0; i < rows; ++i)
            {
                StringRef str_ref = str_column->getDataAt(i);
                StringRef res_ref
                    = !is_utf8 ? substringIndex(str_ref, delim[0], index) : substringIndexUTF8(searcher.get(), str_ref, delim, index);
                std::cout << "result:" << res_ref.toString() << std::endl;
                appendToResultColumn(res_ref, res_data, res_offsets);
            }
        }

        static void constantVector(
            const String & str,
            const String & delim,
            const IColumn * index_column,
            ColumnString::Chars & res_data,
            ColumnString::Offsets & res_offsets)
        {
            size_t rows = index_column->size();
            res_data.reserve(str.size() * rows / 2);
            res_offsets.reserve(rows);

            std::unique_ptr<PositionCaseSensitiveUTF8::SearcherInBigHaystack> searcher
                = !is_utf8 ? nullptr : std::make_unique<PositionCaseSensitiveUTF8::SearcherInBigHaystack>(delim.data(), delim.size());

            StringRef str_ref{str.data(), str.size()};
            for (size_t i = 0; i < rows; ++i)
            {
                Int64 index = index_column->getInt(i);
                StringRef res_ref
                    = !is_utf8 ? substringIndex(str_ref, delim[0], index) : substringIndexUTF8(searcher.get(), str_ref, delim, index);
                appendToResultColumn(res_ref, res_data, res_offsets);
            }
        }

        static void appendToResultColumn(const StringRef & res_ref, ColumnString::Chars & res_data, ColumnString::Offsets & res_offsets)
        {
            size_t res_offset = res_data.size();
            res_data.resize(res_offset + res_ref.size + 1);
            memcpySmallAllowReadWriteOverflow15(&res_data[res_offset], res_ref.data, res_ref.size);
            res_offset += res_ref.size;
            res_data[res_offset] = 0;
            ++res_offset;

            res_offsets.emplace_back(res_offset);
        }

        static StringRef substringIndexUTF8(
            const PositionCaseSensitiveUTF8::SearcherInBigHaystack * searcher, const StringRef & str_ref, const String & delim, Int64 index)
        {
            std::cout << "str:" << str_ref.toString() << ", delim" << delim << ",index:" << index << std::endl;

            if (index == 0)
                return {str_ref.data, 0};

            const auto * begin = reinterpret_cast<const UInt8 *>(str_ref.data);
            const auto * end = reinterpret_cast<const UInt8 *>(str_ref.data + str_ref.size);
            const auto * pos = begin;
            if (index > 0)
            {
                Int64 i = 0;
                while (i < index)
                {
                    pos = searcher->search(pos, end - pos);

                    if (pos != end)
                    {
                        pos += delim.size();
                        ++i;
                    }
                    else
                        return str_ref;
                }
                return {begin, static_cast<size_t>(pos - begin - delim.size())};
            }
            else
            {
                Int64 total = 0;
                while (pos < end && end != (pos = searcher->search(pos, end - pos)))
                {
                    pos += delim.size();
                    ++total;
                }

                if (total + index < 0)
                    return str_ref;

                Int64 index_from_left = total + 1 + index;
                std::cout << "total:" << total << ", index_from_left" << index_from_left << std::endl;
                pos = begin;
                Int64 i = 0;
                while (i < index_from_left && pos < end && end != (pos = searcher->search(pos, end - pos)))
                {
                    pos += delim.size();
                    ++i;
                    std::cout << "pos offset:" << pos - begin << ", total size:" << end - begin << std::endl;
                }
                std::cout << "pos offset:" << pos - begin << ", size:" << end - pos << std::endl;
                StringRef res = {pos, static_cast<size_t>(end - pos)};
                std::cout << "result:" << res.toString() << std::endl;
                return res;
            }
        }

        static StringRef substringIndex(const StringRef & str_ref, char delim, Int64 index)
        {
            std::cout << "str:" << str_ref.toString() << ", delim" << delim << ",index:" << index << std::endl;

            if (index == 0)
                return {str_ref.data, 0};

            if (index > 0)
            {
                const auto * end = str_ref.data + str_ref.size;
                const auto * pos = str_ref.data;
                Int64 i = 0;
                while (i < index)
                {
                    pos = std::find(pos, end, delim);
                    if (pos != end)
                    {
                        ++pos;
                        ++i;
                    }
                    else
                        return str_ref;
                }
                return {str_ref.data, static_cast<size_t>(pos - str_ref.data - 1)};
            }
            else
            {
                const auto * begin = str_ref.data;
                const auto * pos = str_ref.data + str_ref.size;
                Int64 i = 0;
                while (i + index < 0)
                {
                    --pos;
                    while (pos >= begin && *pos != delim)
                        --pos;

                    if (pos >= begin)
                        ++i;
                    else
                        return str_ref;
                }
                return {pos + 1, static_cast<size_t>(str_ref.data + str_ref.size - pos - 1)};
            }
        }
    };
}


REGISTER_FUNCTION(SubstringIndex)
{
    factory.registerFunction<FunctionSubstringIndex<false>>(); /// substringIndex
    factory.registerFunction<FunctionSubstringIndex<true>>(); /// substringIndexUTF8

    factory.registerAlias("SUBSTRING_INDEX", "substringIndex", FunctionFactory::CaseInsensitive);
}


}
