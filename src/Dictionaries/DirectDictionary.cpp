#include "DirectDictionary.h"
#include <IO/WriteHelpers.h>
#include "DictionaryBlockInputStream.h"
#include "DictionaryFactory.h"
#include <Core/Defines.h>
#include <Functions/FunctionHelpers.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int TYPE_MISMATCH;
    extern const int BAD_ARGUMENTS;
    extern const int UNSUPPORTED_METHOD;
}


DirectDictionary::DirectDictionary(
    const StorageID & dict_id_,
    const DictionaryStructure & dict_struct_,
    DictionarySourcePtr source_ptr_,
    BlockPtr saved_block_)
    : IDictionary(dict_id_)
    , dict_struct(dict_struct_)
    , source_ptr{std::move(source_ptr_)}
    , saved_block{std::move(saved_block_)}
{
    if (!this->source_ptr->supportsSelectiveLoad())
        throw Exception{full_name + ": source cannot be used with DirectDictionary", ErrorCodes::UNSUPPORTED_METHOD};

    createAttributes();
}


void DirectDictionary::toParent(const PaddedPODArray<Key> & ids, PaddedPODArray<Key> & out) const
{
    const auto null_value = std::get<UInt64>(hierarchical_attribute->null_values);
    getItemsImpl<UInt64, UInt64>(
        *hierarchical_attribute,
        ids,
        [&](const size_t row, const UInt64 value) { out[row] = value; },
        [&](const size_t) { return null_value; });
}


static inline DirectDictionary::Key getAt(const PaddedPODArray<DirectDictionary::Key> & arr, const size_t idx)
{
    return arr[idx];
}
static inline DirectDictionary::Key getAt(const DirectDictionary::Key & value, const size_t)
{
    return value;
}

DirectDictionary::Key DirectDictionary::getValueOrNullByKey(const Key & to_find) const
{
    std::vector<Key> required_key = {to_find};

    auto stream = source_ptr->loadIds(required_key);
    stream->readPrefix();

    bool is_found = false;
    Key result = std::get<Key>(hierarchical_attribute->null_values);
    while (const auto block = stream->read())
    {
        const IColumn & id_column = *block.safeGetByPosition(0).column;

        for (const size_t attribute_idx : ext::range(0, attributes.size()))
        {
            if (is_found)
                break;

            const IColumn & attribute_column = *block.safeGetByPosition(attribute_idx + 1).column;

            for (const auto row_idx : ext::range(0, id_column.size()))
            {
                const auto key = id_column[row_idx].get<UInt64>();

                if (key == to_find && hierarchical_attribute->name == attribute_name_by_index.at(attribute_idx))
                {
                    result = attribute_column[row_idx].get<Key>();
                    is_found = true;
                    break;
                }
            }
        }
    }

    stream->readSuffix();

    return result;
}

template <typename ChildType, typename AncestorType>
void DirectDictionary::isInImpl(const ChildType & child_ids, const AncestorType & ancestor_ids, PaddedPODArray<UInt8> & out) const
{
    const auto null_value = std::get<UInt64>(hierarchical_attribute->null_values);
    const auto rows = out.size();

    for (const auto row : ext::range(0, rows))
    {
        auto id = getAt(child_ids, row);
        const auto ancestor_id = getAt(ancestor_ids, row);

        for (size_t i = 0; id != null_value && id != ancestor_id && i < DBMS_HIERARCHICAL_DICTIONARY_MAX_DEPTH; ++i)
            id = getValueOrNullByKey(id);

        out[row] = id != null_value && id == ancestor_id;
    }

    query_count.fetch_add(rows, std::memory_order_relaxed);
}


void DirectDictionary::isInVectorVector(
    const PaddedPODArray<Key> & child_ids, const PaddedPODArray<Key> & ancestor_ids, PaddedPODArray<UInt8> & out) const
{
    isInImpl(child_ids, ancestor_ids, out);
}

void DirectDictionary::isInVectorConstant(const PaddedPODArray<Key> & child_ids, const Key ancestor_id, PaddedPODArray<UInt8> & out) const
{
    isInImpl(child_ids, ancestor_id, out);
}

void DirectDictionary::isInConstantVector(const Key child_id, const PaddedPODArray<Key> & ancestor_ids, PaddedPODArray<UInt8> & out) const
{
    isInImpl(child_id, ancestor_ids, out);
}

ColumnPtr DirectDictionary::getColumn(
        const std::string& attribute_name,
        const DataTypePtr &,
        const Columns & key_columns,
        const DataTypes &,
        const ColumnPtr default_untyped) const
{
    ColumnPtr result;

    PaddedPODArray<Key> backup_storage;
    const auto& ids = getColumnDataAsPaddedPODArray(this, key_columns.front(), backup_storage);
    
    const auto & attribute = getAttribute(attribute_name);

    /// TODO: Check that attribute type is same as result type
    /// TODO: Check if const will work as expected

    auto type_call = [&](const auto &dictionary_attribute_type)
    {
        using Type = std::decay_t<decltype(dictionary_attribute_type)>;
        using AttributeType = typename Type::AttributeType;

        auto size = ids.size();

        if constexpr (std::is_same_v<AttributeType, String>)
        {
            auto column_string = ColumnString::create();
            auto out = column_string.get();

            if (default_untyped != nullptr)
            {
                if (const auto default_col = checkAndGetColumn<ColumnString>(*default_untyped))
                {
                    getItemsImpl<String, String>(
                        attribute,
                        ids,
                        [&](const size_t, const String value)
                        {
                            const auto ref = StringRef{value};
                            out->insertData(ref.data, ref.size);
                        },
                        [&](const size_t row)
                        {
                            const auto ref = default_col->getDataAt(row);
                            return String(ref.data, ref.size);
                        });
                }
                else if (const auto default_col_const = checkAndGetColumnConst<ColumnString>(default_untyped.get()))
                {
                    const auto & def = default_col_const->template getValue<String>();

                    getItemsImpl<String, String>(
                        attribute,
                        ids,
                        [&](const size_t, const String value)
                        {
                            const auto ref = StringRef{value};
                            out->insertData(ref.data, ref.size);
                        },
                        [&](const size_t) { return def; });
                }
            }
            else
            {
                const auto & null_value = std::get<StringRef>(attribute.null_values);

                getItemsImpl<String, String>(
                    attribute,
                    ids,
                    [&](const size_t, const String value)
                    {
                        const auto ref = StringRef{value};
                        out->insertData(ref.data, ref.size);
                    },
                    [&](const size_t) { return String(null_value.data, null_value.size); });
            }

            result = std::move(column_string);
        }
        else
        {
            using ResultColumnType
                = std::conditional_t<IsDecimalNumber<AttributeType>, ColumnDecimal<AttributeType>, ColumnVector<AttributeType>>;
            using ResultColumnPtr = typename ResultColumnType::MutablePtr;

            ResultColumnPtr column;

            if constexpr (IsDecimalNumber<AttributeType>)
            {
                // auto scale = getDecimalScale(*attribute.type);
                column = ColumnDecimal<AttributeType>::create(size, 0);
            }
            else if constexpr (IsNumber<AttributeType>)
                column = ColumnVector<AttributeType>::create(size);
 
            auto& out = column->getData();

            if (default_untyped != nullptr)
            {
                if (const auto default_col = checkAndGetColumn<ResultColumnType>(*default_untyped))
                {
                    getItemsImpl<AttributeType, AttributeType>(
                        attribute,
                        ids,
                        [&](const size_t row, const auto value) { return out[row] = value; },
                        [&](const size_t row) { return default_col->getData()[row]; }
                    );
                }
                else if (const auto default_col_const = checkAndGetColumnConst<ResultColumnType>(default_untyped.get()))
                {
                    const auto & def = default_col_const->template getValue<AttributeType>();

                    getItemsImpl<AttributeType, AttributeType>(
                        attribute,
                        ids,
                        [&](const size_t row, const auto value) { return out[row] = value; },
                        [&](const size_t) { return def; }
                    );
                }
            }
            else
            {
                const auto null_value = std::get<AttributeType>(attribute.null_values);

                getItemsImpl<AttributeType, AttributeType>(
                    attribute,
                    ids,
                    [&](const size_t row, const auto value) { return out[row] = value; },
                    [&](const size_t) { return null_value; }
                );
            }

            result = std::move(column);
        }
    };

    callOnDictionaryAttributeType(attribute.type, type_call);
   
    return result;
}

ColumnUInt8::Ptr DirectDictionary::has(const Columns & key_columns, const DataTypes &) const
{
    PaddedPODArray<Key> backup_storage;
    const auto& ids = getColumnDataAsPaddedPODArray(this, key_columns.front(), backup_storage);

    auto result = ColumnUInt8::create(ext::size(ids));
    auto& out = result->getData();

    const auto rows = ext::size(ids);

    HashMap<Key, UInt8> has_key;
    for (const auto row : ext::range(0, rows))
        has_key[ids[row]] = 0;

    std::vector<Key> to_load;
    to_load.reserve(has_key.size());
    for (auto it = has_key.begin(); it != has_key.end(); ++it)
        to_load.emplace_back(static_cast<Key>(it->getKey()));

    auto stream = source_ptr->loadIds(to_load);
    stream->readPrefix();

    while (const auto block = stream->read())
    {
        const IColumn & id_column = *block.safeGetByPosition(0).column;

        for (const auto row_idx : ext::range(0, id_column.size()))
        {
            const auto key = id_column[row_idx].get<UInt64>();
            has_key[key] = 1;
        }
    }

    stream->readSuffix();

    for (const auto row : ext::range(0, rows))
        out[row] = has_key[ids[row]];

    query_count.fetch_add(rows, std::memory_order_relaxed);

    return result;
}

void DirectDictionary::createAttributes()
{
    const auto size = dict_struct.attributes.size();
    attributes.reserve(size);

    for (const auto & attribute : dict_struct.attributes)
    {
        attribute_index_by_name.emplace(attribute.name, attributes.size());
        attribute_name_by_index.emplace(attributes.size(), attribute.name);
        attributes.push_back(createAttributeWithType(attribute.underlying_type, attribute.null_value, attribute.name));

        if (attribute.hierarchical)
        {
            hierarchical_attribute = &attributes.back();

            if (hierarchical_attribute->type != AttributeUnderlyingType::utUInt64)
                throw Exception{full_name + ": hierarchical attribute must be UInt64.", ErrorCodes::TYPE_MISMATCH};
        }
    }
}


template <typename T>
void DirectDictionary::createAttributeImpl(Attribute & attribute, const Field & null_value)
{
    attribute.null_values = T(null_value.get<NearestFieldType<T>>());
}

template <>
void DirectDictionary::createAttributeImpl<String>(Attribute & attribute, const Field & null_value)
{
    attribute.string_arena = std::make_unique<Arena>();
    const String & string = null_value.get<String>();
    const char * string_in_arena = attribute.string_arena->insert(string.data(), string.size());
    attribute.null_values.emplace<StringRef>(string_in_arena, string.size());
}


DirectDictionary::Attribute DirectDictionary::createAttributeWithType(const AttributeUnderlyingType type, const Field & null_value, const std::string & attr_name)
{
    Attribute attr{type, {}, {}, attr_name};

    auto type_call = [&](const auto &dictionary_attribute_type)
    {
        using Type = std::decay_t<decltype(dictionary_attribute_type)>;
        using AttributeType = typename Type::AttributeType;
        createAttributeImpl<AttributeType>(attr, null_value);
    };

    callOnDictionaryAttributeType(type, type_call);

    return attr;
}


template <typename AttributeType, typename OutputType, typename ValueSetter, typename DefaultGetter>
void DirectDictionary::getItemsImpl(
    const Attribute & attribute, const PaddedPODArray<Key> & ids, ValueSetter && set_value, DefaultGetter && get_default) const
{
    const auto rows = ext::size(ids);

    HashMap<Key, OutputType> value_by_key;
    for (const auto row : ext::range(0, rows))
        value_by_key[ids[row]] = get_default(row);

    std::vector<Key> to_load;
    to_load.reserve(value_by_key.size());
    for (auto it = value_by_key.begin(); it != value_by_key.end(); ++it)
        to_load.emplace_back(static_cast<Key>(it->getKey()));

    auto stream = source_ptr->loadIds(to_load);
    stream->readPrefix();

    while (const auto block = stream->read())
    {
        const IColumn & id_column = *block.safeGetByPosition(0).column;

        for (const size_t attribute_idx : ext::range(0, attributes.size()))
        {
            if (attribute.name != attribute_name_by_index.at(attribute_idx))
            {
                continue;
            }

            const IColumn & attribute_column = *block.safeGetByPosition(attribute_idx + 1).column;

            for (const auto row_idx : ext::range(0, id_column.size()))
            {
                const auto key = id_column[row_idx].get<UInt64>();

                if (value_by_key.find(key) != value_by_key.end())
                {
                    if (attribute.type == AttributeUnderlyingType::utFloat32)
                    {
                        value_by_key[key] = static_cast<Float32>(attribute_column[row_idx].get<Float64>());
                    }
                    else
                    {
                        value_by_key[key] = static_cast<OutputType>(attribute_column[row_idx].get<AttributeType>());
                    }

                }
            }
        }
    }

    stream->readSuffix();

    for (const auto row : ext::range(0, rows))
        set_value(row, value_by_key[ids[row]]);

    query_count.fetch_add(rows, std::memory_order_relaxed);
}

const DirectDictionary::Attribute & DirectDictionary::getAttribute(const std::string & attribute_name) const
{
    const auto it = attribute_index_by_name.find(attribute_name);
    if (it == std::end(attribute_index_by_name))
        throw Exception{full_name + ": no such attribute '" + attribute_name + "'", ErrorCodes::BAD_ARGUMENTS};

    return attributes[it->second];
}


BlockInputStreamPtr DirectDictionary::getBlockInputStream(const Names & /* column_names */, size_t /* max_block_size */) const
{
    return source_ptr->loadAll();
}


void registerDictionaryDirect(DictionaryFactory & factory)
{
    auto create_layout = [=](const std::string & full_name,
                             const DictionaryStructure & dict_struct,
                             const Poco::Util::AbstractConfiguration & config,
                             const std::string & config_prefix,
                             DictionarySourcePtr source_ptr) -> DictionaryPtr
    {
        if (dict_struct.key)
            throw Exception{"'key' is not supported for dictionary of layout 'direct'", ErrorCodes::UNSUPPORTED_METHOD};

        if (dict_struct.range_min || dict_struct.range_max)
            throw Exception{full_name
                                + ": elements .structure.range_min and .structure.range_max should be defined only "
                                  "for a dictionary of layout 'range_hashed'",
                            ErrorCodes::BAD_ARGUMENTS};

        const auto dict_id = StorageID::fromDictionaryConfig(config, config_prefix);

        if (config.has(config_prefix + ".lifetime.min") || config.has(config_prefix + ".lifetime.max"))
            throw Exception{"'lifetime' parameter is redundant for the dictionary' of layout 'direct'", ErrorCodes::BAD_ARGUMENTS};


        return std::make_unique<DirectDictionary>(dict_id, dict_struct, std::move(source_ptr));
    };
    factory.registerLayout("direct", create_layout, false);
}


}
