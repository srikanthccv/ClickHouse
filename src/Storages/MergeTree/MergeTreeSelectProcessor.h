#pragma once
#include <DataStreams/IBlockInputStream.h>
#include <Storages/MergeTree/MergeTreeBaseSelectProcessor.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/MarkRange.h>
#include <Storages/MergeTree/MergeTreeBlockReadUtils.h>
#include <Storages/SelectQueryInfo.h>

namespace DB
{


/// Used to read data from single part with select query
/// Cares about PREWHERE, virtual columns, indexes etc.
/// To read data from multiple parts, Storage (MergeTree) creates multiple such objects.
class MergeTreeSelectProcessor : public MergeTreeBaseSelectProcessor
{
public:
    MergeTreeSelectProcessor(
        const MergeTreeData & storage,
        const StorageMetadataPtr & metadata_snapshot,
        const MergeTreeData::DataPartPtr & owned_data_part,
        UInt64 max_block_size_rows,
        size_t preferred_block_size_bytes,
        size_t preferred_max_column_in_block_size_bytes,
        Names required_columns_,
        MarkRanges mark_ranges,
        bool use_uncompressed_cache,
        const PrewhereInfoPtr & prewhere_info,
        ExpressionActionsSettings actions_settings,
        bool check_columns,
        const MergeTreeReaderSettings & reader_settings,
        const Names & virt_column_names = {},
        bool has_limit_below_one_block_ = false);

    ~MergeTreeSelectProcessor() override;

    String getName() const override = 0;

    /// Closes readers and unlock part locks
    void finish();

protected:

    bool getNewTask() override = 0;

    /// Used by Task
    Names required_columns;
    /// Names from header. Used in order to order columns in read blocks.
    Names ordered_names;
    NameSet column_name_set;

    MergeTreeReadTaskColumns task_columns;

    /// Data part will not be removed if the pointer owns it
    MergeTreeData::DataPartPtr data_part;

    /// Mark ranges we should read (in ascending order)
    MarkRanges all_mark_ranges;
    /// Value of _part_index virtual column (used only in SelectExecutor)
    size_t part_index_in_query = 0;
    /// If true, every task will be created only with one range.
    /// It reduces amount of read data for queries with small LIMIT.
    bool has_limit_below_one_block = false;

    bool check_columns;
    size_t total_rows;
};

}
