#include <Functions/IFunctionAdaptors.h>

#include <Common/typeid_cast.h>
#include <Common/assert_cast.h>
#include <Common/LRUCache.h>
#include <Common/SipHash.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnSparse.h>
#include <DataTypes/DataTypeNothing.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/Native.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <Functions/FunctionHelpers.h>
#include <Interpreters/ExpressionActions.h>
#include <IO/WriteHelpers.h>
#include <ext/collection_cast.h>
#include <cstdlib>
#include <memory>
#include <optional>

#if !defined(ARCADIA_BUILD)
#    include <Common/config.h>
#endif

#if USE_EMBEDDED_COMPILER
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    include <llvm/IR/IRBuilder.h>
#    pragma GCC diagnostic pop
#endif


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int ILLEGAL_COLUMN;
}


/// Cache for functions result if it was executed on low cardinality column.
/// It's LRUCache which stores function result executed on dictionary and index mapping.
/// It's expected that cache_size is a number of reading streams (so, will store single cached value per thread).
class ExecutableFunctionLowCardinalityResultCache
{
public:
    /// Will assume that dictionaries with same hash has the same keys.
    /// Just in case, check that they have also the same size.
    struct DictionaryKey
    {
        UInt128 hash;
        UInt64 size;

        bool operator== (const DictionaryKey & other) const { return hash == other.hash && size == other.size; }
    };

    struct DictionaryKeyHash
    {
        size_t operator()(const DictionaryKey & key) const
        {
            SipHash hash;
            hash.update(key.hash);
            hash.update(key.size);
            return hash.get64();
        }
    };

    struct CachedValues
    {
        /// Store ptr to dictionary to be sure it won't be deleted.
        ColumnPtr dictionary_holder;
        ColumnUniquePtr function_result;
        /// Remap positions. new_pos = index_mapping->index(old_pos);
        ColumnPtr index_mapping;
    };

    using CachedValuesPtr = std::shared_ptr<CachedValues>;

    explicit ExecutableFunctionLowCardinalityResultCache(size_t cache_size) : cache(cache_size) {}

    CachedValuesPtr get(const DictionaryKey & key) { return cache.get(key); }
    void set(const DictionaryKey & key, const CachedValuesPtr & mapped) { cache.set(key, mapped); }
    CachedValuesPtr getOrSet(const DictionaryKey & key, const CachedValuesPtr & mapped)
    {
        return cache.getOrSet(key, [&]() { return mapped; }).first;
    }

private:
    using Cache = LRUCache<DictionaryKey, CachedValues, DictionaryKeyHash>;
    Cache cache;
};


void ExecutableFunctionAdaptor::createLowCardinalityResultCache(size_t cache_size)
{
    if (!low_cardinality_result_cache)
        low_cardinality_result_cache = std::make_shared<ExecutableFunctionLowCardinalityResultCache>(cache_size);
}


ColumnPtr wrapInNullable(const ColumnPtr & src, const ColumnsWithTypeAndName & args, const DataTypePtr & result_type, size_t input_rows_count)
{
    ColumnPtr result_null_map_column;

    /// If result is already nullable.
    ColumnPtr src_not_nullable = src;

    if (src->onlyNull())
        return src;
    else if (const auto * nullable = checkAndGetColumn<ColumnNullable>(*src))
    {
        src_not_nullable = nullable->getNestedColumnPtr();
        result_null_map_column = nullable->getNullMapColumnPtr();
    }

    for (const auto & elem : args)
    {
        if (!elem.type->isNullable())
            continue;

        /// Const Nullable that are NULL.
        if (elem.column->onlyNull())
        {
            assert(result_type->isNullable());
            return result_type->createColumnConstWithDefaultValue(input_rows_count);
        }

        if (isColumnConst(*elem.column))
            continue;

        if (const auto * nullable = checkAndGetColumn<ColumnNullable>(*elem.column))
        {
            const ColumnPtr & null_map_column = nullable->getNullMapColumnPtr();
            if (!result_null_map_column) //-V1051
            {
                result_null_map_column = null_map_column;
            }
            else
            {
                MutableColumnPtr mutable_result_null_map_column = IColumn::mutate(std::move(result_null_map_column));

                NullMap & result_null_map = assert_cast<ColumnUInt8 &>(*mutable_result_null_map_column).getData();
                const NullMap & src_null_map = assert_cast<const ColumnUInt8 &>(*null_map_column).getData();

                for (size_t i = 0, size = result_null_map.size(); i < size; ++i)
                    result_null_map[i] |= src_null_map[i];

                result_null_map_column = std::move(mutable_result_null_map_column);
            }
        }
    }

    if (!result_null_map_column)
        return makeNullable(src);

    return ColumnNullable::create(src_not_nullable->convertToFullColumnIfConst(), result_null_map_column);
}


namespace
{

struct NullPresence
{
    bool has_nullable = false;
    bool has_null_constant = false;
};

NullPresence getNullPresense(const ColumnsWithTypeAndName & args)
{
    NullPresence res;

    for (const auto & elem : args)
    {
        res.has_nullable |= elem.type->isNullable();
        res.has_null_constant |= elem.type->onlyNull();
    }

    return res;
}

bool allArgumentsAreConstants(const ColumnsWithTypeAndName & args)
{
    for (const auto & arg : args)
        if (!isColumnConst(*arg.column))
            return false;
    return true;
}
}

ColumnPtr ExecutableFunctionAdaptor::defaultImplementationForConstantArguments(
    const ColumnsWithTypeAndName & args, const DataTypePtr & result_type, size_t input_rows_count, bool dry_run) const
{
    ColumnNumbers arguments_to_remain_constants = impl->getArgumentsThatAreAlwaysConstant();

    /// Check that these arguments are really constant.
    for (auto arg_num : arguments_to_remain_constants)
        if (arg_num < args.size() && !isColumnConst(*args[arg_num].column))
            throw Exception("Argument at index " + toString(arg_num) + " for function " + getName() + " must be constant", ErrorCodes::ILLEGAL_COLUMN);

    if (args.empty() || !impl->useDefaultImplementationForConstants() || !allArgumentsAreConstants(args))
        return nullptr;

    ColumnsWithTypeAndName temporary_columns;
    bool have_converted_columns = false;

    size_t arguments_size = args.size();
    temporary_columns.reserve(arguments_size);
    for (size_t arg_num = 0; arg_num < arguments_size; ++arg_num)
    {
        const ColumnWithTypeAndName & column = args[arg_num];

        if (arguments_to_remain_constants.end() != std::find(arguments_to_remain_constants.begin(), arguments_to_remain_constants.end(), arg_num))
        {
            temporary_columns.emplace_back(ColumnWithTypeAndName{column.column->cloneResized(1), column.type, column.name});
        }
        else
        {
            have_converted_columns = true;
            temporary_columns.emplace_back(ColumnWithTypeAndName{ assert_cast<const ColumnConst *>(column.column.get())->getDataColumnPtr(), column.type, column.name });
        }
    }

    /** When using default implementation for constants, the function requires at least one argument
      *  not in "arguments_to_remain_constants" set. Otherwise we get infinite recursion.
      */
    if (!have_converted_columns)
        throw Exception("Number of arguments for function " + getName() + " doesn't match: the function requires more arguments",
            ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    ColumnPtr result_column = executeWithoutLowCardinalityColumns(temporary_columns, result_type, 1, dry_run);

    /// extremely rare case, when we have function with completely const arguments
    /// but some of them produced by non isDeterministic function
    if (result_column->size() > 1)
        result_column = result_column->cloneResized(1);

    return ColumnConst::create(result_column, input_rows_count);
}


ColumnPtr ExecutableFunctionAdaptor::defaultImplementationForNulls(
    const ColumnsWithTypeAndName & args, const DataTypePtr & result_type, size_t input_rows_count, bool dry_run) const
{
    if (args.empty() || !impl->useDefaultImplementationForNulls())
        return nullptr;

    NullPresence null_presence = getNullPresense(args);

    if (null_presence.has_null_constant)
    {
        // Default implementation for nulls returns null result for null arguments,
        // so the result type must be nullable.
        assert(result_type->isNullable());

        return result_type->createColumnConstWithDefaultValue(input_rows_count);
    }

    if (null_presence.has_nullable)
    {
        ColumnsWithTypeAndName temporary_columns = createBlockWithNestedColumns(args);
        auto temporary_result_type = removeNullable(result_type);

        auto res = executeWithoutLowCardinalityColumns(temporary_columns, temporary_result_type, input_rows_count, dry_run);
        return wrapInNullable(res, args, result_type, input_rows_count);
    }

    return nullptr;
}

ColumnPtr ExecutableFunctionAdaptor::executeWithoutLowCardinalityColumns(
    const ColumnsWithTypeAndName & args, const DataTypePtr & result_type, size_t input_rows_count, bool dry_run) const
{
    if (auto res = defaultImplementationForConstantArguments(args, result_type, input_rows_count, dry_run))
        return res;

    if (auto res = defaultImplementationForNulls(args, result_type, input_rows_count, dry_run))
        return res;

    ColumnPtr res;
    if (dry_run)
        res = impl->executeDryRun(args, result_type, input_rows_count);
    else
        res = impl->execute(args, result_type, input_rows_count);

    if (!res)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Empty column was returned by function {}", getName());

    return res;
}

static const ColumnLowCardinality * findLowCardinalityArgument(const ColumnsWithTypeAndName & arguments)
{
    const ColumnLowCardinality * result_column = nullptr;

    for (const auto & column : arguments)
    {
        if (const auto * low_cardinality_column = checkAndGetColumn<ColumnLowCardinality>(column.column.get()))
        {
            if (result_column)
                throw Exception("Expected single dictionary argument for function.", ErrorCodes::LOGICAL_ERROR);

            result_column = low_cardinality_column;
        }
    }

    return result_column;
}

static ColumnPtr replaceLowCardinalityColumnsByNestedAndGetDictionaryIndexes(
    ColumnsWithTypeAndName & args, bool can_be_executed_on_default_arguments, size_t input_rows_count)
{
    size_t num_rows = input_rows_count;
    ColumnPtr indexes;

    /// Find first LowCardinality column and replace it to nested dictionary.
    for (auto & column : args)
    {
        if (const auto * low_cardinality_column = checkAndGetColumn<ColumnLowCardinality>(column.column.get()))
        {
            /// Single LowCardinality column is supported now.
            if (indexes)
                throw Exception("Expected single dictionary argument for function.", ErrorCodes::LOGICAL_ERROR);

            const auto * low_cardinality_type = checkAndGetDataType<DataTypeLowCardinality>(column.type.get());

            if (!low_cardinality_type)
                throw Exception("Incompatible type for low cardinality column: " + column.type->getName(),
                                ErrorCodes::LOGICAL_ERROR);

            if (can_be_executed_on_default_arguments)
            {
                /// Normal case, when function can be executed on values's default.
                column.column = low_cardinality_column->getDictionary().getNestedColumn();
                indexes = low_cardinality_column->getIndexesPtr();
            }
            else
            {
                /// Special case when default value can't be used. Example: 1 % LowCardinality(Int).
                /// LowCardinality always contains default, so 1 % 0 will throw exception in normal case.
                auto dict_encoded = low_cardinality_column->getMinimalDictionaryEncodedColumn(0, low_cardinality_column->size());
                column.column = dict_encoded.dictionary;
                indexes = dict_encoded.indexes;
            }

            num_rows = column.column->size();
            column.type = low_cardinality_type->getDictionaryType();
        }
    }

    /// Change size of constants.
    for (auto & column : args)
    {
        if (const auto * column_const = checkAndGetColumn<ColumnConst>(column.column.get()))
        {
            column.column = column_const->removeLowCardinality()->cloneResized(num_rows);
            column.type = removeLowCardinality(column.type);
        }
    }

    return indexes;
}

static void convertLowCardinalityColumnsToFull(ColumnsWithTypeAndName & args)
{
    for (auto & column : args)
    {
        column.column = recursiveRemoveLowCardinality(column.column);
        column.type = recursiveRemoveLowCardinality(column.type);
    }
}

static void convertSparseColumnsToFull(ColumnsWithTypeAndName & args)
{
    for (auto & column : args)
        column.column = recursiveRemoveSparse(column.column);
}

ColumnPtr ExecutableFunctionAdaptor::executeWithoutSparseColumns(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count, bool dry_run) const
{
    if (impl->useDefaultImplementationForLowCardinalityColumns())
    {
        ColumnsWithTypeAndName columns_without_low_cardinality = arguments;

        if (const auto * res_low_cardinality_type = typeid_cast<const DataTypeLowCardinality *>(result_type.get()))
        {
            const auto * low_cardinality_column = findLowCardinalityArgument(arguments);
            bool can_be_executed_on_default_arguments = impl->canBeExecutedOnDefaultArguments();
            bool use_cache = low_cardinality_result_cache && can_be_executed_on_default_arguments
                             && low_cardinality_column && low_cardinality_column->isSharedDictionary();
            ExecutableFunctionLowCardinalityResultCache::DictionaryKey key;

            if (use_cache)
            {
                const auto & dictionary = low_cardinality_column->getDictionary();
                key = {dictionary.getHash(), dictionary.size()};

                auto cached_values = low_cardinality_result_cache->get(key);
                if (cached_values)
                {
                    auto indexes = cached_values->index_mapping->index(low_cardinality_column->getIndexes(), 0);
                    return ColumnLowCardinality::create(cached_values->function_result, indexes, true);
                }
            }

            const auto & dictionary_type = res_low_cardinality_type->getDictionaryType();
            ColumnPtr indexes = replaceLowCardinalityColumnsByNestedAndGetDictionaryIndexes(
                    columns_without_low_cardinality, can_be_executed_on_default_arguments, input_rows_count);

            size_t new_input_rows_count = columns_without_low_cardinality.empty()
                                        ? input_rows_count
                                        : columns_without_low_cardinality.front().column->size();

            auto res = executeWithoutLowCardinalityColumns(columns_without_low_cardinality, dictionary_type, new_input_rows_count, dry_run);
            auto keys = res->convertToFullColumnIfConst();

            auto res_mut_dictionary = DataTypeLowCardinality::createColumnUnique(*res_low_cardinality_type->getDictionaryType());
            ColumnPtr res_indexes = res_mut_dictionary->uniqueInsertRangeFrom(*keys, 0, keys->size());
            ColumnUniquePtr res_dictionary = std::move(res_mut_dictionary);

            if (indexes)
            {
                if (use_cache)
                {
                    auto cache_values = std::make_shared<ExecutableFunctionLowCardinalityResultCache::CachedValues>();
                    cache_values->dictionary_holder = low_cardinality_column->getDictionaryPtr();
                    cache_values->function_result = res_dictionary;
                    cache_values->index_mapping = res_indexes;

                    cache_values = low_cardinality_result_cache->getOrSet(key, cache_values);
                    res_dictionary = cache_values->function_result;
                    res_indexes = cache_values->index_mapping;
                }

                return ColumnLowCardinality::create(res_dictionary, res_indexes->index(*indexes, 0), use_cache);
            }
            else
            {
                return ColumnLowCardinality::create(res_dictionary, res_indexes);
            }
        }
        else
        {
            convertLowCardinalityColumnsToFull(columns_without_low_cardinality);
            return executeWithoutLowCardinalityColumns(columns_without_low_cardinality, result_type, input_rows_count, dry_run);
        }
    }
    else
        return executeWithoutLowCardinalityColumns(arguments, result_type, input_rows_count, dry_run);
}

ColumnPtr ExecutableFunctionAdaptor::execute(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count, bool dry_run) const
{
    if (impl->useDefaultImplementationForSparseColumns())
    {
        size_t num_sparse_columns = 0;
        size_t num_full_columns = 0;
        size_t sparse_column_position = 0;

        for (size_t i = 0; i < arguments.size(); ++i)
        {
            const auto * column_sparse = checkAndGetColumn<ColumnSparse>(arguments[i].column.get());
            /// In rare case, when sparse column doesn't have default values,
            /// it's more convenient to convert it to full before execution of function.
            if (column_sparse && column_sparse->getNumberOfDefaults())
            {
                sparse_column_position = i;
                ++num_sparse_columns;
            }
            else if (!isColumnConst(*arguments[i].column))
            {
                ++num_full_columns;
            }
        }

        auto columns_without_sparse = arguments;
        if (num_sparse_columns == 1 && num_full_columns == 0)
        {
            auto & arg_with_sparse = columns_without_sparse[sparse_column_position];
            ColumnPtr sparse_offsets;
            {
                /// New scope to avoid possible mistakes on dangling reference.
                const auto & column_sparse = assert_cast<const ColumnSparse &>(*arg_with_sparse.column);
                sparse_offsets = column_sparse.getOffsetsPtr();
                arg_with_sparse.column = column_sparse.getValuesPtr();
            }

            size_t values_size = arg_with_sparse.column->size();
            for (size_t i = 0; i < columns_without_sparse.size(); ++i)
            {
                if (i == sparse_column_position)
                    continue;

                columns_without_sparse[i].column = columns_without_sparse[i].column->cloneResized(values_size);
            }

            auto res = executeWithoutSparseColumns(columns_without_sparse, result_type, values_size, dry_run);

            if (isColumnConst(*res))
                return res->cloneResized(input_rows_count);

            /// If default of sparse column was changed after execution of function, convert to full column.
            if (!res->isDefaultAt(0))
            {
                const auto & offsets_data = assert_cast<const ColumnVector<UInt64> &>(*sparse_offsets).getData();
                return res->createWithOffsets(offsets_data, input_rows_count, 1);
            }

            return ColumnSparse::create(res, sparse_offsets, input_rows_count);
        }

        convertSparseColumnsToFull(columns_without_sparse);
        return executeWithoutSparseColumns(columns_without_sparse, result_type, input_rows_count, dry_run);
    }

    return executeWithoutSparseColumns(arguments, result_type, input_rows_count, dry_run);
}

void FunctionOverloadResolverAdaptor::checkNumberOfArguments(size_t number_of_arguments) const
{
    if (isVariadic())
        return;

    size_t expected_number_of_arguments = getNumberOfArguments();

    if (number_of_arguments != expected_number_of_arguments)
        throw Exception("Number of arguments for function " + getName() + " doesn't match: passed "
                        + toString(number_of_arguments) + ", should be " + toString(expected_number_of_arguments),
                        ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
}


DataTypePtr FunctionOverloadResolverAdaptor::getReturnTypeDefaultImplementationForNulls(const ColumnsWithTypeAndName & arguments,
                                                                                        const DefaultReturnTypeGetter & getter)
{
    NullPresence null_presence = getNullPresense(arguments);

    if (null_presence.has_null_constant)
    {
        return makeNullable(std::make_shared<DataTypeNothing>());
    }
    if (null_presence.has_nullable)
    {
        auto nested_columns = Block(createBlockWithNestedColumns(arguments));
        auto return_type = getter(ColumnsWithTypeAndName(nested_columns.begin(), nested_columns.end()));
        return makeNullable(return_type);
    }

    return getter(arguments);
}

DataTypePtr FunctionOverloadResolverAdaptor::getReturnTypeWithoutLowCardinality(const ColumnsWithTypeAndName & arguments) const
{
    checkNumberOfArguments(arguments.size());

    if (!arguments.empty() && impl->useDefaultImplementationForNulls())
        return getReturnTypeDefaultImplementationForNulls(arguments, [&](const auto & args) { return impl->getReturnType(args); });

    return impl->getReturnType(arguments);
}

#if USE_EMBEDDED_COMPILER

static std::optional<DataTypes> removeNullables(const DataTypes & types)
{
    for (const auto & type : types)
    {
        if (!typeid_cast<const DataTypeNullable *>(type.get()))
            continue;
        DataTypes filtered;
        for (const auto & sub_type : types)
            filtered.emplace_back(removeNullable(sub_type));
        return filtered;
    }
    return {};
}

bool IFunction::isCompilable(const DataTypes & arguments) const
{
    if (useDefaultImplementationForNulls())
        if (auto denulled = removeNullables(arguments))
            return isCompilableImpl(*denulled);
    return isCompilableImpl(arguments);
}

llvm::Value * IFunction::compile(llvm::IRBuilderBase & builder, const DataTypes & arguments, Values values) const
{
    auto denulled_arguments = removeNullables(arguments);
    if (useDefaultImplementationForNulls() && denulled_arguments)
    {
        auto & b = static_cast<llvm::IRBuilder<> &>(builder);

        std::vector<llvm::Value*> unwrapped_values;
        std::vector<llvm::Value*> is_null_values;

        unwrapped_values.reserve(arguments.size());
        is_null_values.reserve(arguments.size());

        for (size_t i = 0; i < arguments.size(); ++i)
        {
            auto * value = values[i];

            WhichDataType data_type(arguments[i]);
            if (data_type.isNullable())
            {
                unwrapped_values.emplace_back(b.CreateExtractValue(value, {0}));
                is_null_values.emplace_back(b.CreateExtractValue(value, {1}));
            }
            else
            {
                unwrapped_values.emplace_back(value);
            }
        }

        auto * result = compileImpl(builder, *denulled_arguments, unwrapped_values);

        auto * nullable_structure_type = toNativeType(b, makeNullable(getReturnTypeImpl(*denulled_arguments)));
        auto * nullable_structure_value = llvm::Constant::getNullValue(nullable_structure_type);

        auto * nullable_structure_with_result_value = b.CreateInsertValue(nullable_structure_value, result, {0});
        auto * nullable_structure_result_null = b.CreateExtractValue(nullable_structure_with_result_value, {1});

        for (auto * is_null_value : is_null_values)
            nullable_structure_result_null = b.CreateOr(nullable_structure_result_null, is_null_value);

        return b.CreateInsertValue(nullable_structure_with_result_value, nullable_structure_result_null, {1});
    }

    return compileImpl(builder, arguments, std::move(values));
}

#endif

DataTypePtr FunctionOverloadResolverAdaptor::getReturnType(const ColumnsWithTypeAndName & arguments) const
{
    if (impl->useDefaultImplementationForLowCardinalityColumns())
    {
        bool has_low_cardinality = false;
        size_t num_full_low_cardinality_columns = 0;
        size_t num_full_ordinary_columns = 0;

        ColumnsWithTypeAndName args_without_low_cardinality(arguments);

        for (ColumnWithTypeAndName & arg : args_without_low_cardinality)
        {
            bool is_const = arg.column && isColumnConst(*arg.column);
            if (is_const)
                arg.column = assert_cast<const ColumnConst &>(*arg.column).removeLowCardinality();

            if (const auto * low_cardinality_type = typeid_cast<const DataTypeLowCardinality *>(arg.type.get()))
            {
                arg.type = low_cardinality_type->getDictionaryType();
                has_low_cardinality = true;

                if (!is_const)
                    ++num_full_low_cardinality_columns;
            }
            else if (!is_const)
                ++num_full_ordinary_columns;
        }

        for (auto & arg : args_without_low_cardinality)
        {
            arg.column = recursiveRemoveLowCardinality(arg.column);
            arg.type = recursiveRemoveLowCardinality(arg.type);
        }

        auto type_without_low_cardinality = getReturnTypeWithoutLowCardinality(args_without_low_cardinality);

        if (impl->canBeExecutedOnLowCardinalityDictionary() && has_low_cardinality
            && num_full_low_cardinality_columns <= 1 && num_full_ordinary_columns == 0
            && type_without_low_cardinality->canBeInsideLowCardinality())
            return std::make_shared<DataTypeLowCardinality>(type_without_low_cardinality);
        else
            return type_without_low_cardinality;
    }

    return getReturnTypeWithoutLowCardinality(arguments);
}
}
