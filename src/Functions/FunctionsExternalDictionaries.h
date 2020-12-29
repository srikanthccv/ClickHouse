#pragma once

#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeUUID.h>

#include <Common/typeid_cast.h>
#include <Common/assert_cast.h>

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnTuple.h>

#include <Access/AccessFlags.h>

#include <Interpreters/Context.h>
#include <Interpreters/ExternalDictionariesLoader.h>

#include <Functions/IFunctionImpl.h>
#include <Functions/FunctionHelpers.h>

#include <Dictionaries/FlatDictionary.h>
#include <Dictionaries/HashedDictionary.h>
#include <Dictionaries/CacheDictionary.h>
#if defined(OS_LINUX) || defined(__FreeBSD__)
#include <Dictionaries/SSDCacheDictionary.h>
#include <Dictionaries/SSDComplexKeyCacheDictionary.h>
#endif
#include <Dictionaries/ComplexKeyHashedDictionary.h>
#include <Dictionaries/ComplexKeyCacheDictionary.h>
#include <Dictionaries/ComplexKeyDirectDictionary.h>
#include <Dictionaries/RangeHashedDictionary.h>
#include <Dictionaries/IPAddressDictionary.h>
#include <Dictionaries/PolygonDictionaryImplementations.h>
#include <Dictionaries/DirectDictionary.h>

#include <ext/range.h>

#include <type_traits>

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int UNSUPPORTED_METHOD;
    extern const int UNKNOWN_TYPE;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int ILLEGAL_COLUMN;
    extern const int BAD_ARGUMENTS;
}


/** Functions that use plug-ins (external) dictionaries_loader.
  *
  * Get the value of the attribute of the specified type.
  *     dictGetType(dictionary, attribute, id),
  *         Type - placeholder for the type name, any numeric and string types are currently supported.
  *        The type must match the actual attribute type with which it was declared in the dictionary structure.
  *
  * Get an array of identifiers, consisting of the source and parents chain.
  *  dictGetHierarchy(dictionary, id).
  *
  * Is the first identifier the child of the second.
  *  dictIsIn(dictionary, child_id, parent_id).
  */


class FunctionDictHelper
{
public:
    explicit FunctionDictHelper(const Context & context_) : context(context_), external_loader(context.getExternalDictionariesLoader()) {}

    std::shared_ptr<const IDictionaryBase> getDictionary(const String & dictionary_name)
    {
        String resolved_name = DatabaseCatalog::instance().resolveDictionaryName(dictionary_name);
        auto dict = external_loader.getDictionary(resolved_name);
        if (!access_checked)
        {
            context.checkAccess(AccessType::dictGet, dict->getDatabaseOrNoDatabaseTag(), dict->getDictionaryID().getTableName());
            access_checked = true;
        }
        return dict;
    }

    std::shared_ptr<const IDictionaryBase> getDictionary(const ColumnWithTypeAndName & column)
    {
        const auto * dict_name_col = checkAndGetColumnConst<ColumnString>(column.column.get());
        return getDictionary(dict_name_col->getValue<String>());
    }

    bool isDictGetFunctionInjective(const Block & sample_columns)
    {
        /// Assume non-injective by default
        if (!sample_columns)
            return false;

        if (sample_columns.columns() < 3)
            throw Exception{"Wrong arguments count", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

        const auto * dict_name_col = checkAndGetColumnConst<ColumnString>(sample_columns.getByPosition(0).column.get());
        if (!dict_name_col)
            throw Exception{"First argument of function dictGet... must be a constant string", ErrorCodes::ILLEGAL_COLUMN};

        const auto * attr_name_col = checkAndGetColumnConst<ColumnString>(sample_columns.getByPosition(1).column.get());
        if (!attr_name_col)
            throw Exception{"Second argument of function dictGet... must be a constant string", ErrorCodes::ILLEGAL_COLUMN};

        return getDictionary(dict_name_col->getValue<String>())->isInjective(attr_name_col->getValue<String>());
    }

private:
    const Context & context;
    const ExternalDictionariesLoader & external_loader;
    /// Access cannot be not granted, since in this case checkAccess() will throw and access_checked will not be updated.
    std::atomic<bool> access_checked = false;
};


class FunctionDictHas final : public IFunction
{
public:
    static constexpr auto name = "dictHas";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<FunctionDictHas>(context);
    }

    explicit FunctionDictHas(const Context & context_) : helper(context_) {}

    String getName() const override { return name; }

private:
    size_t getNumberOfArguments() const override { return 2; }

    bool useDefaultImplementationForConstants() const final { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const final { return {0}; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isString(arguments[0]))
            throw Exception{"Illegal type " + arguments[0]->getName() + " of first argument of function " + getName()
                + ", expected a string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        if (!WhichDataType(arguments[1]).isUInt64() &&
            !isTuple(arguments[1]))
            throw Exception{"Illegal type " + arguments[1]->getName() + " of second argument of function " + getName()
                + ", must be UInt64 or tuple(...).", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        return std::make_shared<DataTypeUInt8>();
    }

    bool isDeterministic() const override { return false; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        /** Do not require existence of the dictionary if the function is called for empty columns.
          * This is needed to allow successful query analysis on a server,
          *  that is the initiator of a distributed query,
          *  in the case when the function will be invoked for real data only at the remote servers.
          * This feature is controversial and implemented specially
          *  for backward compatibility with the case in Yandex Banner System.
          */
        if (input_rows_count == 0)
            return result_type->createColumn();

        auto dictionary = helper.getDictionary(arguments[0]);
        auto dictionary_identifier_type = dictionary->getIdentifierType();

        const ColumnWithTypeAndName & key_column_with_type = arguments[1];
        const auto key_column = key_column_with_type.column;

        if (dictionary_identifier_type == DictionaryIdentifierType::simple)
        {
            return dictionary->has({key_column}, {std::make_shared<DataTypeUInt64>()});
        }
        else if (dictionary_identifier_type == DictionaryIdentifierType::complex)
        {
            /// Functions in external dictionaries_loader only support full-value (not constant) columns with keys.
            ColumnPtr key_column_full = key_column_with_type.column->convertToFullColumnIfConst();

            const auto & key_columns = typeid_cast<const ColumnTuple &>(*key_column_full).getColumnsCopy();
            const auto & key_types = static_cast<const DataTypeTuple &>(*key_column_with_type.type).getElements();

            return dictionary->has(key_columns, key_types);
        }
        else
        {
            /// TODO: Add support for range
            return nullptr;
        }
    }

    mutable FunctionDictHelper helper;
};

enum class DictionaryGetFunctionType
{
    withoutDefault,
    withDefault
};

template <typename DataType, typename Name, DictionaryGetFunctionType dictionary_get_function_type>
class FunctionDictGetImpl final : public IFunction
{
    using Type = typename DataType::FieldType;

public:
    static constexpr auto name = Name::name;

    static FunctionPtr create(const Context & context, UInt32 dec_scale = 0)
    {
        return std::make_shared<FunctionDictGetImpl>(context, dec_scale);
    }

    explicit FunctionDictGetImpl(const Context & context_, UInt32 dec_scale = 0)
        : helper(context_)
        , decimal_scale(dec_scale)
    {}

    String getName() const override { return name; }

private:
    size_t getNumberOfArguments() const override { return 0; }

    bool isVariadic() const override { return true; }

    bool useDefaultImplementationForConstants() const final { return true; }

    bool isDeterministic() const override { return false; }

    ColumnNumbers getArgumentsThatAreAlwaysConstant() const final { return {0, 1}; }

    DataTypePtr getReturnTypeImpl(const DataTypes &) const override
    {
        if constexpr (IsDataTypeDecimal<DataType>)
            return std::make_shared<DataType>(DataType::maxPrecision(), decimal_scale);
        else
            return std::make_shared<DataType>();
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        if (arguments.size() < 3)
            throw Exception{"Wrong argument count for function " + getName(), ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

        if (input_rows_count == 0)
            return result_type->createColumn();

        const auto * dictionary_name_col = checkAndGetColumnConst<ColumnString>(arguments[0].column.get());
        if (!dictionary_name_col)
            throw Exception{"First argument of function " + getName() + " must be a constant string", ErrorCodes::ILLEGAL_COLUMN};

        String dictionary_name = dictionary_name_col->getValue<String>();

        auto dictionary = helper.getDictionary(dictionary_name);
        if (!dictionary)
            throw Exception("First argument of function " + getName() + " does not name a dictionary", ErrorCodes::ILLEGAL_COLUMN);

        const auto * attr_name_col = checkAndGetColumnConst<ColumnString>(arguments[1].column.get());
        if (!attr_name_col)
            throw Exception{"Second argument of function " + getName() + " must be a constant string", ErrorCodes::ILLEGAL_COLUMN};

        String attr_name = attr_name_col->getValue<String>();

        /// TODO: Use accurateCast if argument is integer
        if (!WhichDataType(arguments[2].type).isUInt64() && !isTuple(arguments[2].type))
            throw Exception{"Illegal type " + arguments[2].type->getName() + " of third argument of function "
                    + getName() + ", must be UInt64 or tuple(...).",
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        auto dictionary_identifier_type = dictionary->getIdentifierType();

        size_t current_arguments_index = 3;

        /// TODO: Add more information to error messages

        ColumnPtr range_col = nullptr;
        DataTypePtr range_col_type = nullptr;

        if (dictionary_identifier_type == DictionaryIdentifierType::range)
        {
            if (current_arguments_index >= arguments.size())
                throw Exception{"Wrong argument count for function " + getName(), ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

            range_col = arguments[current_arguments_index].column;
            range_col_type = arguments[current_arguments_index].type;

            if (!(range_col_type->isValueRepresentedByInteger() && range_col_type->getSizeOfValueInMemory() <= sizeof(Int64)))
                throw Exception{"Illegal type " + range_col_type->getName() + " of fourth argument of function "
                        + getName() + " must be convertible to Int64.",
                    ErrorCodes::ILLEGAL_COLUMN};

            ++current_arguments_index;
        }

        ColumnPtr default_col = nullptr;

        if (dictionary_get_function_type == DictionaryGetFunctionType::withDefault)
        {
            if (current_arguments_index >= arguments.size())
                throw Exception{"Wrong argument count for function test " + getName(), ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

            default_col = arguments[current_arguments_index].column;
        }

        ColumnPtr res;

        const ColumnWithTypeAndName & key_col_with_type = arguments[2];
        const auto key_column = key_col_with_type.column;

        if (dictionary_identifier_type == DictionaryIdentifierType::simple)
        {
            res = dictionary->getColumn(attr_name, result_type, {key_column}, {std::make_shared<DataTypeUInt64>()}, default_col);
        }
        else if (dictionary_identifier_type == DictionaryIdentifierType::complex)
        {
            /// Functions in external dictionaries_loader only support full-value (not constant) columns with keys.
            ColumnPtr key_column_full = key_col_with_type.column->convertToFullColumnIfConst();

            const auto & key_columns = typeid_cast<const ColumnTuple &>(*key_column_full).getColumnsCopy();
            const auto & key_types = static_cast<const DataTypeTuple &>(*key_col_with_type.type).getElements();

            res = dictionary->getColumn(attr_name, result_type, key_columns, key_types, default_col);
        }
        else if (dictionary_identifier_type == DictionaryIdentifierType::range)
        {
            res = dictionary->getColumn(
                attr_name, result_type, {key_column, range_col}, {std::make_shared<DataTypeUInt64>(), range_col_type}, default_col);
        }
        else
            throw Exception{"Unknown dictionary identifier type", ErrorCodes::BAD_ARGUMENTS};

        return res;
    }

    mutable FunctionDictHelper helper;
    UInt32 decimal_scale;
};

template<typename DataType, typename Name>
using FunctionDictGet = FunctionDictGetImpl<DataType, Name, DictionaryGetFunctionType::withoutDefault>;

struct NameDictGetUInt8 { static constexpr auto name = "dictGetUInt8"; };
struct NameDictGetUInt16 { static constexpr auto name = "dictGetUInt16"; };
struct NameDictGetUInt32 { static constexpr auto name = "dictGetUInt32"; };
struct NameDictGetUInt64 { static constexpr auto name = "dictGetUInt64"; };
struct NameDictGetInt8 { static constexpr auto name = "dictGetInt8"; };
struct NameDictGetInt16 { static constexpr auto name = "dictGetInt16"; };
struct NameDictGetInt32 { static constexpr auto name = "dictGetInt32"; };
struct NameDictGetInt64 { static constexpr auto name = "dictGetInt64"; };
struct NameDictGetFloat32 { static constexpr auto name = "dictGetFloat32"; };
struct NameDictGetFloat64 { static constexpr auto name = "dictGetFloat64"; };
struct NameDictGetDate { static constexpr auto name = "dictGetDate"; };
struct NameDictGetDateTime { static constexpr auto name = "dictGetDateTime"; };
struct NameDictGetUUID { static constexpr auto name = "dictGetUUID"; };
struct NameDictGetDecimal32 { static constexpr auto name = "dictGetDecimal32"; };
struct NameDictGetDecimal64 { static constexpr auto name = "dictGetDecimal64"; };
struct NameDictGetDecimal128 { static constexpr auto name = "dictGetDecimal128"; };
struct NameDictGetString { static constexpr auto name = "dictGetString"; };

using FunctionDictGetUInt8 = FunctionDictGet<DataTypeUInt8, NameDictGetUInt8>;
using FunctionDictGetUInt16 = FunctionDictGet<DataTypeUInt16, NameDictGetUInt16>;
using FunctionDictGetUInt32 = FunctionDictGet<DataTypeUInt32, NameDictGetUInt32>;
using FunctionDictGetUInt64 = FunctionDictGet<DataTypeUInt64, NameDictGetUInt64>;
using FunctionDictGetInt8 = FunctionDictGet<DataTypeInt8, NameDictGetInt8>;
using FunctionDictGetInt16 = FunctionDictGet<DataTypeInt16, NameDictGetInt16>;
using FunctionDictGetInt32 = FunctionDictGet<DataTypeInt32, NameDictGetInt32>;
using FunctionDictGetInt64 = FunctionDictGet<DataTypeInt64, NameDictGetInt64>;
using FunctionDictGetFloat32 = FunctionDictGet<DataTypeFloat32, NameDictGetFloat32>;
using FunctionDictGetFloat64 = FunctionDictGet<DataTypeFloat64, NameDictGetFloat64>;
using FunctionDictGetDate = FunctionDictGet<DataTypeDate, NameDictGetDate>;
using FunctionDictGetDateTime = FunctionDictGet<DataTypeDateTime, NameDictGetDateTime>;
using FunctionDictGetUUID = FunctionDictGet<DataTypeUUID, NameDictGetUUID>;
using FunctionDictGetDecimal32 = FunctionDictGet<DataTypeDecimal<Decimal32>, NameDictGetDecimal32>;
using FunctionDictGetDecimal64 = FunctionDictGet<DataTypeDecimal<Decimal64>, NameDictGetDecimal64>;
using FunctionDictGetDecimal128 = FunctionDictGet<DataTypeDecimal<Decimal128>, NameDictGetDecimal128>;
using FunctionDictGetString = FunctionDictGet<DataTypeString, NameDictGetString>;

template<typename DataType, typename Name>
using FunctionDictGetOrDefault = FunctionDictGetImpl<DataType, Name, DictionaryGetFunctionType::withDefault>;

struct NameDictGetUInt8OrDefault { static constexpr auto name = "dictGetUInt8OrDefault"; };
struct NameDictGetUInt16OrDefault { static constexpr auto name = "dictGetUInt16OrDefault"; };
struct NameDictGetUInt32OrDefault { static constexpr auto name = "dictGetUInt32OrDefault"; };
struct NameDictGetUInt64OrDefault { static constexpr auto name = "dictGetUInt64OrDefault"; };
struct NameDictGetInt8OrDefault { static constexpr auto name = "dictGetInt8OrDefault"; };
struct NameDictGetInt16OrDefault { static constexpr auto name = "dictGetInt16OrDefault"; };
struct NameDictGetInt32OrDefault { static constexpr auto name = "dictGetInt32OrDefault"; };
struct NameDictGetInt64OrDefault { static constexpr auto name = "dictGetInt64OrDefault"; };
struct NameDictGetFloat32OrDefault { static constexpr auto name = "dictGetFloat32OrDefault"; };
struct NameDictGetFloat64OrDefault { static constexpr auto name = "dictGetFloat64OrDefault"; };
struct NameDictGetDateOrDefault { static constexpr auto name = "dictGetDateOrDefault"; };
struct NameDictGetDateTimeOrDefault { static constexpr auto name = "dictGetDateTimeOrDefault"; };
struct NameDictGetUUIDOrDefault { static constexpr auto name = "dictGetUUIDOrDefault"; };
struct NameDictGetDecimal32OrDefault { static constexpr auto name = "dictGetDecimal32OrDefault"; };
struct NameDictGetDecimal64OrDefault { static constexpr auto name = "dictGetDecimal64OrDefault"; };
struct NameDictGetDecimal128OrDefault { static constexpr auto name = "dictGetDecimal128OrDefault"; };
struct NameDictGetStringOrDefault { static constexpr auto name = "dictGetStringOrDefault"; };

using FunctionDictGetUInt8OrDefault = FunctionDictGetOrDefault<DataTypeUInt8, NameDictGetUInt8OrDefault>;
using FunctionDictGetUInt16OrDefault = FunctionDictGetOrDefault<DataTypeUInt16, NameDictGetUInt16OrDefault>;
using FunctionDictGetUInt32OrDefault = FunctionDictGetOrDefault<DataTypeUInt32, NameDictGetUInt32OrDefault>;
using FunctionDictGetUInt64OrDefault = FunctionDictGetOrDefault<DataTypeUInt64, NameDictGetUInt64OrDefault>;
using FunctionDictGetInt8OrDefault = FunctionDictGetOrDefault<DataTypeInt8, NameDictGetInt8OrDefault>;
using FunctionDictGetInt16OrDefault = FunctionDictGetOrDefault<DataTypeInt16, NameDictGetInt16OrDefault>;
using FunctionDictGetInt32OrDefault = FunctionDictGetOrDefault<DataTypeInt32, NameDictGetInt32OrDefault>;
using FunctionDictGetInt64OrDefault = FunctionDictGetOrDefault<DataTypeInt64, NameDictGetInt64OrDefault>;
using FunctionDictGetFloat32OrDefault = FunctionDictGetOrDefault<DataTypeFloat32, NameDictGetFloat32OrDefault>;
using FunctionDictGetFloat64OrDefault = FunctionDictGetOrDefault<DataTypeFloat64, NameDictGetFloat64OrDefault>;
using FunctionDictGetDateOrDefault = FunctionDictGetOrDefault<DataTypeDate, NameDictGetDateOrDefault>;
using FunctionDictGetDateTimeOrDefault = FunctionDictGetOrDefault<DataTypeDateTime, NameDictGetDateTimeOrDefault>;
using FunctionDictGetUUIDOrDefault = FunctionDictGetOrDefault<DataTypeUUID, NameDictGetUUIDOrDefault>;
using FunctionDictGetDecimal32OrDefault = FunctionDictGetOrDefault<DataTypeDecimal<Decimal32>, NameDictGetDecimal32OrDefault>;
using FunctionDictGetDecimal64OrDefault = FunctionDictGetOrDefault<DataTypeDecimal<Decimal64>, NameDictGetDecimal64OrDefault>;
using FunctionDictGetDecimal128OrDefault = FunctionDictGetOrDefault<DataTypeDecimal<Decimal128>, NameDictGetDecimal128OrDefault>;
using FunctionDictGetStringOrDefault = FunctionDictGetOrDefault<DataTypeString, NameDictGetStringOrDefault>;


/// TODO: Use new API
/// This variant of function derives the result type automatically.
template <DictionaryGetFunctionType dictionary_get_function_type>
class FunctionDictGetNoType final : public IFunction
{
public:
    static constexpr auto name = dictionary_get_function_type == DictionaryGetFunctionType::withDefault ? "dictGetOrDefault" : "dictGet";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<FunctionDictGetNoType>(context);
    }

    explicit FunctionDictGetNoType(const Context & context_) : context(context_), helper(context_) {}

    String getName() const override { return name; }

private:
    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }

    bool useDefaultImplementationForConstants() const final { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const final { return {0, 1}; }

    bool isDeterministic() const override { return false; }

    bool isInjective(const ColumnsWithTypeAndName & sample_columns) const override
    {
        return helper.isDictGetFunctionInjective(sample_columns);
    }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (arguments.size() < 3)
            throw Exception{"Wrong argument count for function " + getName(), ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH};

        String dict_name;
        if (const auto * name_col = checkAndGetColumnConst<ColumnString>(arguments[0].column.get()))
        {
            dict_name = name_col->getValue<String>();
        }
        else
            throw Exception{"Illegal type " + arguments[0].type->getName() + " of first argument of function " + getName()
                + ", expected a const string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        String attr_name;
        if (const auto * name_col = checkAndGetColumnConst<ColumnString>(arguments[1].column.get()))
        {
            attr_name = name_col->getValue<String>();
        }
        else
            throw Exception{"Illegal type " + arguments[1].type->getName() + " of second argument of function " + getName()
                + ", expected a const string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        auto dict = helper.getDictionary(dict_name);
        const DictionaryStructure & structure = dict->getStructure();

        for (const auto& attribute : structure.attributes)
        {
            if (attribute.name != attr_name)
            {
                continue;
            }

            WhichDataType dt = attribute.type;

            if constexpr (dictionary_get_function_type == DictionaryGetFunctionType::withoutDefault)
            {
                switch (dt.idx)
                {
                    case TypeIndex::String:
                    case TypeIndex::FixedString:
                        impl = FunctionDictGetString::create(context);
                        break;
                    case TypeIndex::UInt8:
                        impl = FunctionDictGetUInt8::create(context);
                        break;
                    case TypeIndex::UInt16:
                        impl = FunctionDictGetUInt16::create(context);
                        break;
                    case TypeIndex::UInt32:
                        impl = FunctionDictGetUInt32::create(context);
                        break;
                    case TypeIndex::UInt64:
                        impl = FunctionDictGetUInt64::create(context);
                        break;
                    case TypeIndex::Int8:
                        impl = FunctionDictGetInt8::create(context);
                        break;
                    case TypeIndex::Int16:
                        impl = FunctionDictGetInt16::create(context);
                        break;
                    case TypeIndex::Int32:
                        impl = FunctionDictGetInt32::create(context);
                        break;
                    case TypeIndex::Int64:
                        impl = FunctionDictGetInt64::create(context);
                        break;
                    case TypeIndex::Float32:
                        impl = FunctionDictGetFloat32::create(context);
                        break;
                    case TypeIndex::Float64:
                        impl = FunctionDictGetFloat64::create(context);
                        break;
                    case TypeIndex::Date:
                        impl = FunctionDictGetDate::create(context);
                        break;
                    case TypeIndex::DateTime:
                        impl = FunctionDictGetDateTime::create(context);
                        break;
                    case TypeIndex::UUID:
                        impl = FunctionDictGetUUID::create(context);
                        break;
                    case TypeIndex::Decimal32:
                        impl = FunctionDictGetDecimal32::create(context, getDecimalScale(*attribute.type));
                        break;
                    case TypeIndex::Decimal64:
                        impl = FunctionDictGetDecimal64::create(context, getDecimalScale(*attribute.type));
                        break;
                    case TypeIndex::Decimal128:
                        impl = FunctionDictGetDecimal128::create(context, getDecimalScale(*attribute.type));
                        break;
                    default:
                        throw Exception("Unknown dictGet type", ErrorCodes::UNKNOWN_TYPE);
                }
            }
            else
            {
                switch (dt.idx)
                {
                    case TypeIndex::String:
                        impl = FunctionDictGetStringOrDefault::create(context);
                        break;
                    case TypeIndex::UInt8:
                        impl = FunctionDictGetUInt8OrDefault::create(context);
                        break;
                    case TypeIndex::UInt16:
                        impl = FunctionDictGetUInt16OrDefault::create(context);
                        break;
                    case TypeIndex::UInt32:
                        impl = FunctionDictGetUInt32OrDefault::create(context);
                        break;
                    case TypeIndex::UInt64:
                        impl = FunctionDictGetUInt64OrDefault::create(context);
                        break;
                    case TypeIndex::Int8:
                        impl = FunctionDictGetInt8OrDefault::create(context);
                        break;
                    case TypeIndex::Int16:
                        impl = FunctionDictGetInt16OrDefault::create(context);
                        break;
                    case TypeIndex::Int32:
                        impl = FunctionDictGetInt32OrDefault::create(context);
                        break;
                    case TypeIndex::Int64:
                        impl = FunctionDictGetInt64OrDefault::create(context);
                        break;
                    case TypeIndex::Float32:
                        impl = FunctionDictGetFloat32OrDefault::create(context);
                        break;
                    case TypeIndex::Float64:
                        impl = FunctionDictGetFloat64OrDefault::create(context);
                        break;
                    case TypeIndex::Date:
                        impl = FunctionDictGetDateOrDefault::create(context);
                        break;
                    case TypeIndex::DateTime:
                        impl = FunctionDictGetDateTimeOrDefault::create(context);
                        break;
                    case TypeIndex::UUID:
                        impl = FunctionDictGetUUIDOrDefault::create(context);
                        break;
                    case TypeIndex::Decimal32:
                        impl = FunctionDictGetDecimal32OrDefault::create(context, getDecimalScale(*attribute.type));
                        break;
                    case TypeIndex::Decimal64:
                        impl = FunctionDictGetDecimal64OrDefault::create(context, getDecimalScale(*attribute.type));
                        break;
                    case TypeIndex::Decimal128:
                        impl = FunctionDictGetDecimal128OrDefault::create(context, getDecimalScale(*attribute.type));
                        break;
                    default:
                        throw Exception("Unknown dictGetOrDefault type", ErrorCodes::UNKNOWN_TYPE);
                }
            }

            return attribute.type;
        }

        throw Exception{"No such attribute '" + attr_name + "'", ErrorCodes::BAD_ARGUMENTS};
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        return impl->executeImpl(arguments, result_type, input_rows_count);
    }

    const Context & context;
    mutable FunctionDictHelper helper;
    mutable FunctionPtr impl; // underlying function used by dictGet function without explicit type info
};

/// Functions to work with hierarchies.

class FunctionDictGetHierarchy final : public IFunction
{
public:
    static constexpr auto name = "dictGetHierarchy";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<FunctionDictGetHierarchy>(context);
    }

    explicit FunctionDictGetHierarchy(const Context & context_) : helper(context_) {}

    String getName() const override { return name; }

private:
    size_t getNumberOfArguments() const override { return 2; }
    bool isInjective(const ColumnsWithTypeAndName & /*sample_columns*/) const override { return true; }

    bool useDefaultImplementationForConstants() const final { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const final { return {0}; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isString(arguments[0]))
            throw Exception{"Illegal type " + arguments[0]->getName() + " of first argument of function " + getName()
                + ", expected a string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        if (!WhichDataType(arguments[1]).isUInt64())
            throw Exception{"Illegal type " + arguments[1]->getName() + " of second argument of function " + getName()
                + ", must be UInt64.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        return std::make_shared<DataTypeArray>(std::make_shared<DataTypeUInt64>());
    }

    bool isDeterministic() const override { return false; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        if (input_rows_count == 0)
            return result_type->createColumn();

        auto dict = helper.getDictionary(arguments[0]);
        ColumnPtr res;

        if (!((res = executeDispatch<FlatDictionary>(arguments, result_type, dict))
            || (res = executeDispatch<DirectDictionary>(arguments, result_type, dict))
            || (res = executeDispatch<HashedDictionary>(arguments, result_type, dict))
            || (res = executeDispatch<CacheDictionary>(arguments, result_type, dict))))
            throw Exception{"Unsupported dictionary type " + dict->getTypeName(), ErrorCodes::UNKNOWN_TYPE};

        return res;
    }

    template <typename DictionaryType>
    ColumnPtr executeDispatch(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, const std::shared_ptr<const IDictionaryBase> & dict_ptr) const
    {
        const auto * dict = typeid_cast<const DictionaryType *>(dict_ptr.get());
        if (!dict)
            return nullptr;

        if (!dict->hasHierarchy())
            throw Exception{"Dictionary does not have a hierarchy", ErrorCodes::UNSUPPORTED_METHOD};

        const auto get_hierarchies = [&] (const PaddedPODArray<UInt64> & in, PaddedPODArray<UInt64> & out, PaddedPODArray<UInt64> & offsets)
        {
            const auto size = in.size();

            /// copy of `in` array
            auto in_array = std::make_unique<PaddedPODArray<UInt64>>(std::begin(in), std::end(in));
            /// used for storing and handling result of ::toParent call
            auto out_array = std::make_unique<PaddedPODArray<UInt64>>(size);
            /// resulting hierarchies
            std::vector<std::vector<IDictionary::Key>> hierarchies(size);    /// TODO Bad code, poor performance.

            /// total number of non-zero elements, used for allocating all the required memory upfront
            size_t total_count = 0;

            while (true)
            {
                auto all_zeroes = true;

                /// erase zeroed identifiers, store non-zeroed ones
                for (const auto i : ext::range(0, size))
                {
                    const auto id = (*in_array)[i];
                    if (0 == id)
                        continue;


                    auto & hierarchy = hierarchies[i];

                    /// Checking for loop
                    if (std::find(std::begin(hierarchy), std::end(hierarchy), id) != std::end(hierarchy))
                        continue;

                    all_zeroes = false;
                    /// place id at it's corresponding place
                    hierarchy.push_back(id);

                    ++total_count;
                }

                if (all_zeroes)
                    break;

                /// translate all non-zero identifiers at once
                dict->toParent(*in_array, *out_array);

                /// we're going to use the `in_array` from this iteration as `out_array` on the next one
                std::swap(in_array, out_array);
            }

            out.reserve(total_count);
            offsets.resize(size);

            for (const auto i : ext::range(0, size))
            {
                const auto & ids = hierarchies[i];
                out.insert_assume_reserved(std::begin(ids), std::end(ids));
                offsets[i] = out.size();
            }
        };

        const auto * id_col_untyped = arguments[1].column.get();
        if (const auto * id_col = checkAndGetColumn<ColumnUInt64>(id_col_untyped))
        {
            const auto & in = id_col->getData();
            auto backend = ColumnUInt64::create();
            auto offsets = ColumnArray::ColumnOffsets::create();
            get_hierarchies(in, backend->getData(), offsets->getData());
            return ColumnArray::create(std::move(backend), std::move(offsets));
        }
        else if (const auto * id_col_const = checkAndGetColumnConst<ColumnVector<UInt64>>(id_col_untyped))
        {
            const PaddedPODArray<UInt64> in(1, id_col_const->getValue<UInt64>());
            auto backend = ColumnUInt64::create();
            auto offsets = ColumnArray::ColumnOffsets::create();
            get_hierarchies(in, backend->getData(), offsets->getData());
            auto array = ColumnArray::create(std::move(backend), std::move(offsets));
            return result_type->createColumnConst(id_col_const->size(), (*array)[0].get<Array>());
        }
        else
            throw Exception{"Second argument of function " + getName() + " must be UInt64", ErrorCodes::ILLEGAL_COLUMN};
    }

    mutable FunctionDictHelper helper;
};


class FunctionDictIsIn final : public IFunction
{
public:
    static constexpr auto name = "dictIsIn";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<FunctionDictIsIn>(context);
    }

    explicit FunctionDictIsIn(const Context & context_)
        : helper(context_) {}

    String getName() const override { return name; }

private:
    size_t getNumberOfArguments() const override { return 3; }

    bool useDefaultImplementationForConstants() const final { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const final { return {0}; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (!isString(arguments[0]))
            throw Exception{"Illegal type " + arguments[0]->getName() + " of first argument of function " + getName()
                + ", expected a string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        if (!WhichDataType(arguments[1]).isUInt64())
            throw Exception{"Illegal type " + arguments[1]->getName() + " of second argument of function " + getName()
                + ", must be UInt64.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        if (!WhichDataType(arguments[2]).isUInt64())
            throw Exception{"Illegal type " + arguments[2]->getName() + " of third argument of function " + getName()
                + ", must be UInt64.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        return std::make_shared<DataTypeUInt8>();
    }

    bool isDeterministic() const override { return false; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        if (input_rows_count == 0)
            return result_type->createColumn();

        auto dict = helper.getDictionary(arguments[0]);

        ColumnPtr res;
        if (!((res = executeDispatch<FlatDictionary>(arguments, dict))
            || (res = executeDispatch<DirectDictionary>(arguments, dict))
            || (res = executeDispatch<HashedDictionary>(arguments, dict))
            || (res = executeDispatch<CacheDictionary>(arguments, dict))))
            throw Exception{"Unsupported dictionary type " + dict->getTypeName(), ErrorCodes::UNKNOWN_TYPE};

        return res;
    }

    template <typename DictionaryType>
    ColumnPtr executeDispatch(const ColumnsWithTypeAndName & arguments, const std::shared_ptr<const IDictionaryBase> & dict_ptr) const
    {
        const auto * dict = typeid_cast<const DictionaryType *>(dict_ptr.get());
        if (!dict)
            return nullptr;

        if (!dict->hasHierarchy())
            throw Exception{"Dictionary does not have a hierarchy", ErrorCodes::UNSUPPORTED_METHOD};

        const auto * child_id_col_untyped = arguments[1].column.get();
        const auto * ancestor_id_col_untyped = arguments[2].column.get();

        if (const auto * child_id_col = checkAndGetColumn<ColumnUInt64>(child_id_col_untyped))
            return execute(dict, child_id_col, ancestor_id_col_untyped);
        else if (const auto * child_id_col_const = checkAndGetColumnConst<ColumnVector<UInt64>>(child_id_col_untyped))
            return execute(dict, child_id_col_const, ancestor_id_col_untyped);
        else
            throw Exception{"Illegal column " + child_id_col_untyped->getName()
                + " of second argument of function " + getName(), ErrorCodes::ILLEGAL_COLUMN};
    }

    template <typename DictionaryType>
    ColumnPtr execute(const DictionaryType * dict,
                 const ColumnUInt64 * child_id_col, const IColumn * ancestor_id_col_untyped) const
    {
        if (const auto * ancestor_id_col = checkAndGetColumn<ColumnUInt64>(ancestor_id_col_untyped))
        {
            auto out = ColumnUInt8::create();

            const auto & child_ids = child_id_col->getData();
            const auto & ancestor_ids = ancestor_id_col->getData();
            auto & data = out->getData();
            const auto size = child_id_col->size();
            data.resize(size);

            dict->isInVectorVector(child_ids, ancestor_ids, data);
            return out;
        }
        else if (const auto * ancestor_id_col_const = checkAndGetColumnConst<ColumnVector<UInt64>>(ancestor_id_col_untyped))
        {
            auto out = ColumnUInt8::create();

            const auto & child_ids = child_id_col->getData();
            const auto ancestor_id = ancestor_id_col_const->getValue<UInt64>();
            auto & data = out->getData();
            const auto size = child_id_col->size();
            data.resize(size);

            dict->isInVectorConstant(child_ids, ancestor_id, data);
            return out;
        }
        else
        {
            throw Exception{"Illegal column " + ancestor_id_col_untyped->getName()
                + " of third argument of function " + getName(), ErrorCodes::ILLEGAL_COLUMN};
        }
    }

    template <typename DictionaryType>
    ColumnPtr execute(const DictionaryType * dict, const ColumnConst * child_id_col, const IColumn * ancestor_id_col_untyped) const
    {
        if (const auto * ancestor_id_col = checkAndGetColumn<ColumnUInt64>(ancestor_id_col_untyped))
        {
            auto out = ColumnUInt8::create();

            const auto child_id = child_id_col->getValue<UInt64>();
            const auto & ancestor_ids = ancestor_id_col->getData();
            auto & data = out->getData();
            const auto size = child_id_col->size();
            data.resize(size);

            dict->isInConstantVector(child_id, ancestor_ids, data);
            return out;
        }
        else if (const auto * ancestor_id_col_const = checkAndGetColumnConst<ColumnVector<UInt64>>(ancestor_id_col_untyped))
        {
            const auto child_id = child_id_col->getValue<UInt64>();
            const auto ancestor_id = ancestor_id_col_const->getValue<UInt64>();
            UInt8 res = 0;

            dict->isInConstantConstant(child_id, ancestor_id, res);
            return DataTypeUInt8().createColumnConst(child_id_col->size(), res);
        }
        else
            throw Exception{"Illegal column " + ancestor_id_col_untyped->getName()
                + " of third argument of function " + getName(), ErrorCodes::ILLEGAL_COLUMN};
    }

    mutable FunctionDictHelper helper;
};

}
