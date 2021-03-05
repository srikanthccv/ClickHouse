#include <Interpreters/join_common.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/ActionsDAG.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/getLeastSupertype.h>
#include <DataStreams/materializeBlock.h>
#include <IO/WriteHelpers.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int TYPE_MISMATCH;
    extern const int LOGICAL_ERROR;
}

namespace
{

void changeNullability(MutableColumnPtr & mutable_column)
{
    ColumnPtr column = std::move(mutable_column);
    if (const auto * nullable = checkAndGetColumn<ColumnNullable>(*column))
        column = nullable->getNestedColumnPtr();
    else
        column = makeNullable(column);

    mutable_column = IColumn::mutate(std::move(column));
}

ColumnPtr changeLowCardinality(const ColumnPtr & column, const ColumnPtr & dst_sample)
{
    if (dst_sample->lowCardinality())
    {
        MutableColumnPtr lc = dst_sample->cloneEmpty();
        typeid_cast<ColumnLowCardinality &>(*lc).insertRangeFromFullColumn(*column, 0, column->size());
        return lc;
    }

    return column->convertToFullColumnIfLowCardinality();
}

}

namespace JoinCommon
{

void convertColumnToNullable(ColumnWithTypeAndName & column, bool low_card_nullability)
{
    if (low_card_nullability && column.type->lowCardinality())
    {
        column.column = recursiveRemoveLowCardinality(column.column);
        column.type = recursiveRemoveLowCardinality(column.type);
    }

    if (column.type->isNullable() || !column.type->canBeInsideNullable())
        return;

    column.type = makeNullable(column.type);
    if (column.column)
        column.column = makeNullable(column.column);
}

void convertColumnsToNullable(Block & block, size_t starting_pos)
{
    for (size_t i = starting_pos; i < block.columns(); ++i)
        convertColumnToNullable(block.getByPosition(i));
}

/// @warning It assumes that every NULL has default value in nested column (or it does not matter)
void removeColumnNullability(ColumnWithTypeAndName & column)
{
    if (!column.type->isNullable())
        return;

    column.type = static_cast<const DataTypeNullable &>(*column.type).getNestedType();
    if (column.column)
    {
        const auto * nullable_column = checkAndGetColumn<ColumnNullable>(*column.column);
        ColumnPtr nested_column = nullable_column->getNestedColumnPtr();
        MutableColumnPtr mutable_column = IColumn::mutate(std::move(nested_column));
        column.column = std::move(mutable_column);
    }
}

/// Change both column nullability and low cardinality
void changeColumnRepresentation(const ColumnPtr & src_column, ColumnPtr & dst_column)
{
    bool nullable_src = src_column->isNullable();
    bool nullable_dst = dst_column->isNullable();

    ColumnPtr dst_not_null = JoinCommon::emptyNotNullableClone(dst_column);
    bool lowcard_src = JoinCommon::emptyNotNullableClone(src_column)->lowCardinality();
    bool lowcard_dst = dst_not_null->lowCardinality();
    bool change_lowcard = (!lowcard_src && lowcard_dst) || (lowcard_src && !lowcard_dst);

    if (nullable_src && !nullable_dst)
    {
        const auto * nullable = checkAndGetColumn<ColumnNullable>(*src_column);
        if (change_lowcard)
            dst_column = changeLowCardinality(nullable->getNestedColumnPtr(), dst_column);
        else
            dst_column = nullable->getNestedColumnPtr();
    }
    else if (!nullable_src && nullable_dst)
    {
        if (change_lowcard)
            dst_column = makeNullable(changeLowCardinality(src_column, dst_not_null));
        else
            dst_column = makeNullable(src_column);
    }
    else /// same nullability
    {
        if (change_lowcard)
        {
            if (const auto * nullable = checkAndGetColumn<ColumnNullable>(*src_column))
            {
                dst_column = makeNullable(changeLowCardinality(nullable->getNestedColumnPtr(), dst_not_null));
                assert_cast<ColumnNullable &>(*dst_column->assumeMutable()).applyNullMap(nullable->getNullMapColumn());
            }
            else
                dst_column = changeLowCardinality(src_column, dst_not_null);
        }
        else
            dst_column = src_column;
    }
}

ColumnPtr emptyNotNullableClone(const ColumnPtr & column)
{
    if (column->isNullable())
        return checkAndGetColumn<ColumnNullable>(*column)->getNestedColumnPtr()->cloneEmpty();
    return column->cloneEmpty();
}

ColumnRawPtrs materializeColumnsInplace(Block & block, const Names & names)
{
    ColumnRawPtrs ptrs;
    ptrs.reserve(names.size());

    for (const auto & column_name : names)
    {
        auto & column = block.getByName(column_name).column;
        column = recursiveRemoveLowCardinality(column->convertToFullColumnIfConst());
        ptrs.push_back(column.get());
    }

    return ptrs;
}

Columns materializeColumns(const Block & block, const Names & names)
{
    Columns materialized;
    materialized.reserve(names.size());

    for (const auto & column_name : names)
    {
        const auto & src_column = block.getByName(column_name).column;
        materialized.emplace_back(recursiveRemoveLowCardinality(src_column->convertToFullColumnIfConst()));
    }

    return materialized;
}

ColumnRawPtrs getRawPointers(const Columns & columns)
{
    ColumnRawPtrs ptrs;
    ptrs.reserve(columns.size());

    for (const auto & column : columns)
        ptrs.push_back(column.get());

    return ptrs;
}

void removeLowCardinalityInplace(Block & block)
{
    for (size_t i = 0; i < block.columns(); ++i)
    {
        auto & col = block.getByPosition(i);
        col.column = recursiveRemoveLowCardinality(col.column);
        col.type = recursiveRemoveLowCardinality(col.type);
    }
}

void removeLowCardinalityInplace(Block & block, const Names & names, bool change_type)
{
    for (const String & column_name : names)
    {
        auto & col = block.getByName(column_name);
        col.column = recursiveRemoveLowCardinality(col.column);
        if (change_type)
            col.type = recursiveRemoveLowCardinality(col.type);
    }
}

void restoreLowCardinalityInplace(Block & block)
{
    for (size_t i = 0; i < block.columns(); ++i)
    {
        auto & col = block.getByPosition(i);
        if (col.type->lowCardinality() && col.column && !col.column->lowCardinality())
            col.column = changeLowCardinality(col.column, col.type->createColumn());
    }
}

ColumnRawPtrs extractKeysForJoin(const Block & block_keys, const Names & key_names)
{
    size_t keys_size = key_names.size();
    ColumnRawPtrs key_columns(keys_size);

    for (size_t i = 0; i < keys_size; ++i)
    {
        const String & column_name = key_names[i];
        key_columns[i] = block_keys.getByName(column_name).column.get();

        /// We will join only keys, where all components are not NULL.
        if (const auto * nullable = checkAndGetColumn<ColumnNullable>(*key_columns[i]))
            key_columns[i] = &nullable->getNestedColumn();
    }

    return key_columns;
}

void checkTypesOfKeys(const Block & block_left, const Names & key_names_left, const Block & block_right, const Names & key_names_right)
{
    size_t keys_size = key_names_left.size();

    for (size_t i = 0; i < keys_size; ++i)
    {
        DataTypePtr left_type = removeNullable(recursiveRemoveLowCardinality(block_left.getByName(key_names_left[i]).type));
        DataTypePtr right_type = removeNullable(recursiveRemoveLowCardinality(block_right.getByName(key_names_right[i]).type));

        if (!left_type->equals(*right_type))
            throw Exception("Type mismatch of columns to JOIN by: "
                + key_names_left[i] + " " + left_type->getName() + " at left, "
                + key_names_right[i] + " " + right_type->getName() + " at right",
                ErrorCodes::TYPE_MISMATCH);
    }
}

void createMissedColumns(Block & block)
{
    for (size_t i = 0; i < block.columns(); ++i)
    {
        auto & column = block.getByPosition(i);
        if (!column.column)
            column.column = column.type->createColumn();
    }
}

/// Append totals from right to left block, correct types if needed
void joinTotals(const Block & totals, const Block & columns_to_add, const JoinInfo & join_info, Block & block)
{
    if (join_info.forceNullableLeft())
        convertColumnsToNullable(block);

    if (Block totals_without_keys = totals)
    {
        for (const auto & name : join_info.key_names_right)
            totals_without_keys.erase(totals_without_keys.getPositionByName(name));

        if (join_info.forceNullableRight())
        {
            for (auto & col : totals_without_keys)
            {
                if (col.type->canBeInsideNullable())
                    JoinCommon::convertColumnToNullable(col);
            }
        }

        for (size_t i = 0; i < totals_without_keys.columns(); ++i)
            block.insert(totals_without_keys.safeGetByPosition(i));
    }
    else
    {
        /// We will join empty `totals` - from one row with the default values.

        for (size_t i = 0; i < columns_to_add.columns(); ++i)
        {
            const auto & col = columns_to_add.getByPosition(i);
            block.insert({
                col.type->createColumnConstWithDefaultValue(1)->convertToFullColumnIfConst(),
                col.type,
                col.name});
        }
    }
}

void addDefaultValues(IColumn & column, const DataTypePtr & type, size_t count)
{
    column.reserve(column.size() + count);
    for (size_t i = 0; i < count; ++i)
        type->insertDefaultInto(column);
}

bool typesEqualUpToNullability(DataTypePtr left_type, DataTypePtr right_type)
{
    DataTypePtr left_type_strict = removeNullable(recursiveRemoveLowCardinality(left_type));
    DataTypePtr right_type_strict = removeNullable(recursiveRemoveLowCardinality(right_type));
    return left_type_strict->equals(*right_type_strict);
}

ActionsDAGPtr applyKeyConvertToTable(
    const ColumnsWithTypeAndName & cols_src, const NameToTypeMap & type_mapping, bool replace_columns, Names & names_to_rename)
{
    ColumnsWithTypeAndName cols_dst = cols_src;
    for (auto & col : cols_dst)
    {
        if (auto it = type_mapping.find(col.name); it != type_mapping.end())
        {
            col.type = it->second;
            col.column = nullptr;
        }
    }

    NameToNameMap key_column_rename;
    /// Returns converting actions for tables that need to be performed before join
    auto dag = ActionsDAG::makeConvertingActions(
        cols_src, cols_dst, ActionsDAG::MatchColumnsMode::Name, true, !replace_columns, &key_column_rename);

    for (auto & name : names_to_rename)
    {
        const auto it = key_column_rename.find(name);
        if (it != key_column_rename.end())
            name = it->second;
    }
    return dag;
}

void splitAdditionalColumns(const Names & key_names_right, const Block & sample_block, Block & block_keys, Block & block_others)
{
    block_others = materializeBlock(sample_block);

    for (const String & column_name : key_names_right)
    {
        /// Extract right keys with correct keys order. There could be the same key names.
        if (!block_keys.has(column_name))
        {
            auto & col = block_others.getByName(column_name);
            block_keys.insert(col);
            block_others.erase(column_name);
        }
    }
}

Block getRequiredRightKeys(
    const Names & left_keys,
    const Names & right_keys,
    const NameSet & required_keys,
    const Block & right_table_keys,
    std::vector<String> & keys_sources)
{
    Block required_right_keys;

    for (size_t i = 0; i < right_keys.size(); ++i)
    {
        const String & right_key_name = right_keys[i];

        if (required_keys.count(right_key_name) && !required_right_keys.has(right_key_name))
        {
            const auto & right_key = right_table_keys.getByName(right_key_name);
            required_right_keys.insert(right_key);
            keys_sources.push_back(left_keys[i]);
        }
    }

    return required_right_keys;
}

}


NotJoined::NotJoined(const JoinInfo & join_info, const Block & saved_block_sample_, const Block & right_sample_block,
                     const Block & result_sample_block_)
    : saved_block_sample(saved_block_sample_)
    , result_sample_block(materializeBlock(result_sample_block_))
{
    std::vector<String> tmp;
    Block right_table_keys;
    Block sample_block_with_columns_to_add;
    JoinCommon::splitAdditionalColumns(join_info.key_names_right, right_sample_block, right_table_keys, sample_block_with_columns_to_add);

    Block required_right_keys = JoinCommon::getRequiredRightKeys(
        join_info.key_names_left, join_info.key_names_right, join_info.required_right_keys, right_table_keys, tmp);

    std::unordered_map<size_t, size_t> left_to_right_key_remap;

    if (join_info.hasUsing())
    {
        for (size_t i = 0; i < join_info.key_names_left.size(); ++i)
        {
            const String & left_key_name = join_info.key_names_left[i];
            const String & right_key_name = join_info.key_names_right[i];

            size_t left_key_pos = result_sample_block.getPositionByName(left_key_name);
            size_t right_key_pos = saved_block_sample.getPositionByName(right_key_name);

            if (!required_right_keys.has(right_key_name))
                left_to_right_key_remap[left_key_pos] = right_key_pos;
        }
    }

    /// result_sample_block: left_sample_block + left expressions, right not key columns, required right keys
    size_t left_columns_count = result_sample_block.columns() -
        sample_block_with_columns_to_add.columns() - required_right_keys.columns();

    for (size_t left_pos = 0; left_pos < left_columns_count; ++left_pos)
    {
        /// We need right 'x' for 'RIGHT JOIN ... USING(x)'.
        if (left_to_right_key_remap.count(left_pos))
        {
            size_t right_key_pos = left_to_right_key_remap[left_pos];
            setRightIndex(right_key_pos, left_pos);
        }
        else
            column_indices_left.emplace_back(left_pos);
    }

    for (size_t right_pos = 0; right_pos < saved_block_sample.columns(); ++right_pos)
    {
        const String & name = saved_block_sample.getByPosition(right_pos).name;
        if (!result_sample_block.has(name))
            continue;

        size_t result_position = result_sample_block.getPositionByName(name);

        /// Don't remap left keys twice. We need only qualified right keys here
        if (result_position < left_columns_count)
            continue;

        setRightIndex(right_pos, result_position);
    }

    if (column_indices_left.size() + column_indices_right.size() + same_result_keys.size() != result_sample_block.columns())
        throw Exception("Error in columns mapping in RIGHT|FULL JOIN. Left: " + toString(column_indices_left.size()) +
                        ", right: " + toString(column_indices_right.size()) +
                        ", same: " + toString(same_result_keys.size()) +
                        ", result: " + toString(result_sample_block.columns()),
                        ErrorCodes::LOGICAL_ERROR);
}

void NotJoined::setRightIndex(size_t right_pos, size_t result_position)
{
    if (!column_indices_right.count(right_pos))
    {
        column_indices_right[right_pos] = result_position;
        extractColumnChanges(right_pos, result_position);
    }
    else
        same_result_keys[result_position] = column_indices_right[right_pos];
}

void NotJoined::extractColumnChanges(size_t right_pos, size_t result_pos)
{
    const auto & src = saved_block_sample.getByPosition(right_pos).column;
    const auto & dst = result_sample_block.getByPosition(result_pos).column;

    if (!src->isNullable() && dst->isNullable())
        right_nullability_adds.push_back(right_pos);

    if (src->isNullable() && !dst->isNullable())
        right_nullability_removes.push_back(right_pos);

    ColumnPtr src_not_null = JoinCommon::emptyNotNullableClone(src);
    ColumnPtr dst_not_null = JoinCommon::emptyNotNullableClone(dst);

    if (src_not_null->lowCardinality() != dst_not_null->lowCardinality())
        right_lowcard_changes.push_back({right_pos, dst_not_null});
}

void NotJoined::correctLowcardAndNullability(MutableColumns & columns_right)
{
    for (size_t pos : right_nullability_removes)
        changeNullability(columns_right[pos]);

    for (auto & [pos, dst_sample] : right_lowcard_changes)
        columns_right[pos] = changeLowCardinality(std::move(columns_right[pos]), dst_sample)->assumeMutable();

    for (size_t pos : right_nullability_adds)
        changeNullability(columns_right[pos]);
}

void NotJoined::addLeftColumns(Block & block, size_t rows_added) const
{
    for (size_t pos : column_indices_left)
    {
        auto & col = block.getByPosition(pos);

        auto mut_col = col.column->cloneEmpty();
        JoinCommon::addDefaultValues(*mut_col, col.type, rows_added);
        col.column = std::move(mut_col);
    }
}

void NotJoined::addRightColumns(Block & block, MutableColumns & columns_right) const
{
    for (const auto & pr : column_indices_right)
    {
        auto & right_column = columns_right[pr.first];
        auto & result_column = block.getByPosition(pr.second).column;
#ifndef NDEBUG
        if (result_column->getName() != right_column->getName())
            throw Exception("Wrong columns assign in RIGHT|FULL JOIN: " + result_column->getName() +
                            " " + right_column->getName(), ErrorCodes::LOGICAL_ERROR);
#endif
        result_column = std::move(right_column);
    }
}

void NotJoined::copySameKeys(Block & block) const
{
    for (const auto & pr : same_result_keys)
    {
        auto & src_column = block.getByPosition(pr.second).column;
        auto & dst_column = block.getByPosition(pr.first).column;
        JoinCommon::changeColumnRepresentation(src_column, dst_column);
    }
}

}
