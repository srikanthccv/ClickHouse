#include <Storages/MergeTree/MergeTreeSink.h>
#include <Storages/MergeTree/MergeTreeDataPartInMemory.h>
#include <Storages/StorageMergeTree.h>
#include <Interpreters/PartLog.h>


namespace DB
{

MergeTreeSink::~MergeTreeSink() = default;

MergeTreeSink::MergeTreeSink(
    StorageMergeTree & storage_,
    StorageMetadataPtr metadata_snapshot_,
    size_t max_parts_per_block_,
    ContextPtr context_)
    : SinkToStorage(metadata_snapshot_->getSampleBlock())
    , storage(storage_)
    , metadata_snapshot(metadata_snapshot_)
    , max_parts_per_block(max_parts_per_block_)
    , context(context_)
{
}

void MergeTreeSink::onStart()
{
    /// Only check "too many parts" before write,
    /// because interrupting long-running INSERT query in the middle is not convenient for users.
    storage.delayInsertOrThrowIfNeeded();
}

void MergeTreeSink::onFinish()
{
    finishPrevPart();
}

struct MergeTreeSink::PrevPart
{
    struct Partition
    {
        MergeTreeDataWriter::TempPart temp_part;
        UInt64 elapsed_ns;
    };

    std::vector<Partition> partitions;
};


void MergeTreeSink::consume(Chunk chunk)
{
    auto block = getHeader().cloneWithColumns(chunk.detachColumns());

    auto part_blocks = storage.writer.splitBlockIntoParts(block, max_parts_per_block, metadata_snapshot, context);
    std::vector<MergeTreeSink::PrevPart::Partition> partitions;
    for (auto & current_block : part_blocks)
    {
        Stopwatch watch;

        auto temp_part = storage.writer.writeTempPart(current_block, metadata_snapshot, context);

        UInt64 elapsed_ns = watch.elapsed();

        /// If optimize_on_insert setting is true, current_block could become empty after merge
        /// and we didn't create part.
        if (!temp_part.part)
            continue;

        partitions.emplace_back(MergeTreeSink::PrevPart::Partition{.temp_part = std::move(temp_part), .elapsed_ns = elapsed_ns});
    }

    finishPrevPart();
    prev_part = std::make_unique<MergeTreeSink::PrevPart>();
    prev_part->partitions = std::move(partitions);
}

void MergeTreeSink::finishPrevPart()
{
    if (!prev_part)
        return;

    for (auto & partition : prev_part->partitions)
    {
        partition.temp_part.finalize();

        auto & part = partition.temp_part.part;

        /// Part can be deduplicated, so increment counters and add to part log only if it's really added
        if (storage.renameTempPartAndAdd(part, &storage.increment, nullptr, storage.getDeduplicationLog()))
        {
            PartLog::addNewPart(storage.getContext(), part, partition.elapsed_ns);

            /// Initiate async merge - it will be done if it's good time for merge and if there are space in 'background_pool'.
            storage.background_operations_assignee.trigger();
        }
    }

    prev_part.reset();
}

}
