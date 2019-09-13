#pragma once
#include <Storages/MergeTree/MergeTreeBaseSelectBlockInputStream.h>


namespace DB
{

class MergeTreeReadPool;


/** Used in conjunction with MergeTreeReadPool, asking it for more work to do and performing whatever reads it is asked
  * to perform.
  */
class MergeTreeThreadSelectBlockInputStream : public MergeTreeBaseSelectBlockInputProcessor
{
public:
    MergeTreeThreadSelectBlockInputStream(
        const size_t thread_,
        const std::shared_ptr<MergeTreeReadPool> & pool_,
        const size_t min_marks_to_read_,
        const UInt64 max_block_size_,
        size_t preferred_block_size_bytes_,
        size_t preferred_max_column_in_block_size_bytes_,
        const MergeTreeData & storage_,
        const bool use_uncompressed_cache_,
        const PrewhereInfoPtr & prewhere_info_,
        const Settings & settings_,
        const Names & virt_column_names_);

    String getName() const override { return "MergeTreeThread"; }

    ~MergeTreeThreadSelectBlockInputStream() override;

protected:
    /// Requests read task from MergeTreeReadPool and signals whether it got one
    bool getNewTask() override;

private:
    /// "thread" index (there are N threads and each thread is assigned index in interval [0..N-1])
    size_t thread;

    std::shared_ptr<MergeTreeReadPool> pool;
    size_t min_marks_to_read;

    /// Last part readed in this thread
    std::string last_readed_part_path;
    /// Names from header. Used in order to order columns in read blocks.
    Names ordered_names;
};

}
