#pragma once

#include <Columns/IColumn.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypesDecimal.h>
#include "DictionaryStructure.h"

namespace DB
{

/**
 * In Dictionaries implementation String attribute is stored in arena and StringRefs are pointing to it.
 */
template <typename DictionaryAttributeType>
using DictionaryValueType = 
    std::conditional_t<std::is_same_v<DictionaryAttributeType, String>, StringRef, DictionaryAttributeType>;

/**
 * Used to create column with right type for DictionaryAttributeType.
 */
template <typename DictionaryAttributeType>
class DictionaryAttributeColumnProvider
{
public:
    using ColumnType = 
        std::conditional_t<std::is_same_v<DictionaryAttributeType, String>, ColumnString,
            std::conditional_t<IsDecimalNumber<DictionaryAttributeType>, ColumnDecimal<DictionaryAttributeType>, 
                ColumnVector<DictionaryAttributeType>>>;

    using ColumnPtr = typename ColumnType::MutablePtr;

    static ColumnPtr getColumn(const DictionaryAttribute & dictionary_attribute, size_t size)
    {
        if constexpr (std::is_same_v<DictionaryAttributeType, String>)
        {
            return ColumnType::create();
        }
        if constexpr (IsDecimalNumber<DictionaryAttributeType>)
        {
            auto scale = getDecimalScale(*dictionary_attribute.nested_type);
            return ColumnType::create(size, scale);
        }
        else if constexpr (IsNumber<DictionaryAttributeType>)
            return ColumnType::create(size);
        else
            throw Exception{"Unsupported attribute type.", ErrorCodes::TYPE_MISMATCH};
    }
};

/**
 * DictionaryDefaultValueExtractor used to simplify getting default value for IDictionary function `getColumn`.
 * Provides interface for getting default value with operator[];
 * 
 * If default_values_column is not null in constructor than this column values will be used as default values.
 * If default_values_column is null then attribute_default_value will be used.
 */
template <typename DefaultValueType>
class DictionaryDefaultValueExtractor
{
    using ResultColumnType = 
        std::conditional_t< std::is_same_v<DefaultValueType, StringRef>, ColumnString,
            std::conditional_t<IsDecimalNumber<DefaultValueType>, ColumnDecimal<DefaultValueType>,
                ColumnVector<DefaultValueType>>>;

public:
    DictionaryDefaultValueExtractor(DefaultValueType attribute_default_value, ColumnPtr default_values_column_ = nullptr)
    {
        if (default_values_column_ != nullptr)
        {
            if (const auto * const default_col = checkAndGetColumn<ResultColumnType>(*default_values_column))
            {
                default_values_column = default_col;
            }
            else if (const auto * const default_col_const = checkAndGetColumnConst<ResultColumnType>(default_values_column_.get()))
            {
                /// TODO: Check String lifetime safety
                /// DefaultValueType for StringColumn is StringRef, but const column getValue will return String
                using ConstColumnValue = std::conditional_t<std::is_same_v<DefaultValueType, StringRef>, String, DefaultValueType>;
                default_value = std::make_optional<DefaultValueType>(default_col_const->template getValue<ConstColumnValue>());
            }
            else
                throw Exception{"Type of default column is not the same as result type.", ErrorCodes::TYPE_MISMATCH};
        }
        else
            default_value = std::make_optional<DefaultValueType>(attribute_default_value);
    }

    DefaultValueType operator[](size_t row)
    {
        if (default_value)
            return *default_value;
        
        if constexpr (std::is_same_v<ResultColumnType, ColumnString>)
            return default_values_column->getDataAt(row);
        else
            return default_values_column->getData()[row];
    }
private:
    const ResultColumnType * default_values_column = nullptr;
    std::optional<DefaultValueType> default_value = {};
};

/**
 * Returns ColumnVector data as PaddedPodArray. 
 * 
 * If column is constant parameter backup_storage is used to store values.
 */
template <typename T>
static const PaddedPODArray<T> & getColumnVectorData(
    const IDictionaryBase * dictionary [[maybe_unused]],
    const ColumnPtr column,
    PaddedPODArray<T> & backup_storage)
{
    bool is_const_column = isColumnConst(*column);
    auto full_column = column->convertToFullColumnIfConst();
    auto vector_col = checkAndGetColumn<ColumnVector<T>>(full_column.get());

    if (!vector_col)
    {
        throw Exception{ErrorCodes::TYPE_MISMATCH,
            "{}: type mismatch: column has wrong type expected {}",
            dictionary->getDictionaryID().getNameForLogs(),
            TypeName<T>::get()};
    }

    if (is_const_column)
    {
        // With type conversion and const columns we need to use backup storage here
        auto & data = vector_col->getData();
        backup_storage.assign(data);

        return backup_storage;
    }
    else
    {
        return vector_col->getData();
    }
}

}
