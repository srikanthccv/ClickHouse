#include <sstream>
#include <Common/typeid_cast.h>

#include <DataTypes/getLeastCommonType.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypesNumber.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int NO_COMMON_TYPE;
}

namespace
{
    String getExceptionMessagePrefix(const DataTypes & types)
    {
        std::stringstream res;
        res << "There is no common type for types ";

        bool first = true;
        for (const auto & type : types)
        {
            if (first)
                res << ", ";
            first = false;

            res << type->getName();
        }

        return res.str();
    }
}


DataTypePtr getLeastCommonType(const DataTypes & types)
{
    /// Trivial cases

    if (types.empty())
        throw Exception("Empty list of types passed to getLeastCommonType function", ErrorCodes::BAD_ARGUMENTS);

    if (types.size() == 1)
        return types[0];

    /// All types are equal
    {
        bool all_equal = true;
        for (size_t i = 1, size = types.size(); i < size; ++i)
        {
            if (!types[i]->equals(*types[0]))
            {
                all_equal = false;
                break;
            }
        }

        if (all_equal)
            return types[0];
    }

    /// Recursive rules

    /// For Arrays
    {
        bool have_array = false;
        bool all_arrays = true;

        DataTypes nested_types;
        nested_types.reserve(types.size());

        for (const auto & type : types)
        {
            if (const DataTypeArray * type_array = typeid_cast<const DataTypeArray *>(type.get()))
            {
                have_array = true;
                nested_types.emplace_back(type_array->getNestedType());
            }
            else
                all_arrays = false;
        }

        if (have_array)
        {
            if (!all_arrays)
                throw Exception(getExceptionMessagePrefix(types) + " because some of them are Array and some of them are not", ErrorCodes::NO_COMMON_TYPE);

            return std::make_shared<DataTypeArray>(getLeastCommonType(nested_types));
        }
    }

    /// For tuples
    {
        bool have_tuple = false;
        bool all_tuples = true;
        size_t tuple_size = 0;

        std::vector<DataTypes> nested_types;

        for (const auto & type : types)
        {
            if (const DataTypeTuple * type_tuple = typeid_cast<const DataTypeTuple *>(type.get()))
            {
                if (!have_tuple)
                {
                    tuple_size = type_tuple->getElements().size();
                    nested_types.resize(tuple_size);
                    for (size_t elem_idx = 0; elem_idx < tuple_size; ++elem_idx)
                        nested_types[elem_idx].reserve(types.size());
                }
                else if (tuple_size != type_tuple->getElements().size())
                    throw Exception(getExceptionMessagePrefix(types) + " because Tuples have different sizes", ErrorCodes::NO_COMMON_TYPE);

                have_tuple = true;

                for (size_t elem_idx = 0; elem_idx < tuple_size; ++elem_idx)
                    nested_types[elem_idx].emplace_back(type_tuple->getElements()[elem_idx]);
            }
            else
                all_tuples = false;

            if (have_tuple)
            {
                if (!all_tuples)
                    throw Exception(getExceptionMessagePrefix(types) + " because some of them are Tuple and some of them are not", ErrorCodes::NO_COMMON_TYPE);

                DataTypes common_tuple_types(tuple_size);
                for (size_t elem_idx = 0; elem_idx < tuple_size; ++elem_idx)
                    common_tuple_types[elem_idx] = getLeastCommonType(nested_types[elem_idx]);

                return std::make_shared<DataTypeTuple>(common_tuple_types);
            }
        }
    }

    /// For Nullable
    {
        bool have_nullable = false;

        DataTypes nested_types;
        nested_types.reserve(types.size());

        for (const auto & type : types)
        {
            if (const DataTypeNullable * type_nullable = typeid_cast<const DataTypeNullable *>(type.get()))
            {
                have_nullable = true;
                nested_types.emplace_back(type_nullable->getNestedType());
            }
            else
                nested_types.emplace_back(type);
        }

        if (have_nullable)
        {
            return std::make_shared<DataTypeNullable>(getLeastCommonType(nested_types));
        }
    }

    /// Non-recursive rules

    /// For String and FixedString, or for different FixedStrings, the common type is String.
    /// No other types are compatible with Strings.
    {
        bool have_string = false;
        bool all_strings = true;

        for (const auto & type : types)
        {
            if (typeid_cast<const DataTypeString *>(type.get())
                || typeid_cast<const DataTypeFixedString *>(type.get()))
                have_string = true;
            else
                all_strings = false;
        }

        if (have_string)
        {
            if (!all_strings)
                throw Exception(getExceptionMessagePrefix(types) + " because some of them are String/FixedString and some of them are not", ErrorCodes::NO_COMMON_TYPE);

            return std::make_shared<DataTypeString>();
        }
    }

    /// For Date and DateTime, the common type is DateTime. No other types are compatible.
    {
        bool have_date_or_datetime = false;
        bool all_date_or_datetime = true;

        for (const auto & type : types)
        {
            if (typeid_cast<const DataTypeDate *>(type.get())
                || typeid_cast<const DataTypeDateTime *>(type.get()))
                have_date_or_datetime = true;
            else
                all_date_or_datetime = false;
        }

        if (have_date_or_datetime)
        {
            if (!all_date_or_datetime)
                throw Exception(getExceptionMessagePrefix(types) + " because some of them are Date/DateTime and some of them are not", ErrorCodes::NO_COMMON_TYPE);

            return std::make_shared<DataTypeDateTime>();
        }
    }

    /// For numeric types, the most complicated part.
    {
        bool all_numbers = true;

        size_t max_bits_of_signed_integer = 0;
        size_t max_bits_of_unsigned_integer = 0;
        size_t max_mantissa_bits_of_floating = 0;

        auto maximize = [](size_t & what, size_t value)
        {
            if (value > what)
                what = value;
        };

        for (const auto & type : types)
        {
            if (typeid_cast<const DataTypeUInt8 *>(type.get()))
                maximize(max_bits_of_unsigned_integer, 8);
            else if (typeid_cast<const DataTypeUInt16 *>(type.get()))
                maximize(max_bits_of_unsigned_integer, 16);
            else if (typeid_cast<const DataTypeUInt32 *>(type.get()))
                maximize(max_bits_of_unsigned_integer, 32);
            else if (typeid_cast<const DataTypeUInt64 *>(type.get()))
                maximize(max_bits_of_unsigned_integer, 64);
            else if (typeid_cast<const DataTypeInt8 *>(type.get()))
                maximize(max_bits_of_signed_integer, 8);
            else if (typeid_cast<const DataTypeInt16 *>(type.get()))
                maximize(max_bits_of_signed_integer, 16);
            else if (typeid_cast<const DataTypeInt32 *>(type.get()))
                maximize(max_bits_of_signed_integer, 32);
            else if (typeid_cast<const DataTypeInt64 *>(type.get()))
                maximize(max_bits_of_signed_integer, 64);
            else if (typeid_cast<const DataTypeFloat32 *>(type.get()))
                maximize(max_mantissa_bits_of_floating, 24);
            else if (typeid_cast<const DataTypeFloat64 *>(type.get()))
                maximize(max_mantissa_bits_of_floating, 53);
            else
                all_numbers = false;
        }

        if (max_bits_of_signed_integer || max_bits_of_unsigned_integer || max_mantissa_bits_of_floating)
        {
            if (!all_numbers)
                throw Exception(getExceptionMessagePrefix(types) + " because some of them are numbers and some of them are not", ErrorCodes::NO_COMMON_TYPE);

            /// If there are signed and unsigned types of same bit-width, the result must be signed number with at least one more bit.
            /// Example, common of Int32, UInt32 = Int64.

            size_t min_bit_width_of_integer = std::max(max_bits_of_signed_integer, max_bits_of_unsigned_integer);
            if (max_bits_of_signed_integer == max_bits_of_unsigned_integer)
                ++min_bit_width_of_integer;

            /// If the result must be floating.
            if (max_mantissa_bits_of_floating)
            {
                size_t min_mantissa_bits = std::max(min_bit_width_of_integer, max_mantissa_bits_of_floating);
                if (min_mantissa_bits <= 24)
                    return std::make_shared<DataTypeFloat32>();
                else if (min_mantissa_bits <= 53)
                    return std::make_shared<DataTypeFloat64>();
                else
                    throw Exception(getExceptionMessagePrefix(types)
                        + " because some of them are integers and some are floating point,"
                        " but there is no floating point type, that can exactly represent all required integers", ErrorCodes::NO_COMMON_TYPE);
            }

            /// If the result must be signed integer.
            if (max_bits_of_signed_integer)
            {
                if (min_bit_width_of_integer <= 8)
                    return std::make_shared<DataTypeInt8>();
                else if (min_bit_width_of_integer <= 16)
                    return std::make_shared<DataTypeInt16>();
                else if (min_bit_width_of_integer <= 32)
                    return std::make_shared<DataTypeInt32>();
                else if (min_bit_width_of_integer <= 64)
                    return std::make_shared<DataTypeInt64>();
                else
                    throw Exception(getExceptionMessagePrefix(types)
                        + " because some of them are signed integers and some are unsigned integers,"
                        " but there is no signed integer type, that can exactly represent all required unsigned integer values", ErrorCodes::NO_COMMON_TYPE);
            }

            /// All unsigned.
            {
                if (min_bit_width_of_integer <= 8)
                    return std::make_shared<DataTypeUInt8>();
                else if (min_bit_width_of_integer <= 16)
                    return std::make_shared<DataTypeUInt16>();
                else if (min_bit_width_of_integer <= 32)
                    return std::make_shared<DataTypeUInt32>();
                else if (min_bit_width_of_integer <= 64)
                    return std::make_shared<DataTypeUInt64>();
                else
                    throw Exception("Logical error: " + getExceptionMessagePrefix(types)
                        + " but as all data types are unsigned integers, we must have found maximum unsigned integer type", ErrorCodes::NO_COMMON_TYPE);
            }
        }
    }

    /// All other data types (UUID, AggregateFunction, Enum...) are compatible only if they are the same (checked in trivial cases).
    throw Exception(getExceptionMessagePrefix(types), ErrorCodes::NO_COMMON_TYPE);
}

}
