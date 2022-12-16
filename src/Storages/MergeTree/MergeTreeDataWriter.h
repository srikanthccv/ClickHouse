#pragma once

#include <Core/Block.h>

#include <IO/WriteBufferFromFile.h>
#include <Compression/CompressedWriteBuffer.h>

#include <Columns/ColumnsNumber.h>

#include <Interpreters/sortBlock.h>

#include <Processors/Chunk.h>

#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/MergedBlockOutputStream.h>


namespace DB
{

struct BlockWithPartition
{
    Block block;
    Row partition;
    ChunkOffsetsPtr offsets;

    BlockWithPartition(Block && block_, Row && partition_)
        : block(block_), partition(std::move(partition_))
    {
    }

    BlockWithPartition(Block && block_, Row && partition_, ChunkOffsetsPtr chunk_offsets_)
        : block(block_), partition(std::move(partition_)), offsets(chunk_offsets_)
    {
    }
};

using BlocksWithPartition = std::vector<BlockWithPartition>;

/** Writes new parts of data to the merge tree.
  */
class MergeTreeDataWriter
{
public:
    explicit MergeTreeDataWriter(MergeTreeData & data_)
        : data(data_)
        , log(&Poco::Logger::get(data.getLogName() + " (Writer)"))
    {}

    /** Split the block to blocks, each of them must be written as separate part.
      *  (split rows by partition)
      * Works deterministically: if same block was passed, function will return same result in same order.
      */
    static BlocksWithPartition splitBlockIntoParts(const Block & block, size_t max_parts, const StorageMetadataPtr & metadata_snapshot, ContextPtr context, ChunkOffsetsPtr chunk_offsets = nullptr);

    /// This structure contains not completely written temporary part.
    /// Some writes may happen asynchronously, e.g. for blob storages.
    /// You should call finalize() to wait until all data is written.

    struct TemporaryPart
    {
        MergeTreeData::MutableDataPartPtr part;

        struct Stream
        {
            std::unique_ptr<MergedBlockOutputStream> stream;
            MergedBlockOutputStream::Finalizer finalizer;
        };

        std::vector<Stream> streams;

        scope_guard temporary_directory_lock;

        void finalize();
    };

    /** All rows must correspond to same partition.
      * Returns part with unique name starting with 'tmp_', yet not added to MergeTreeData.
      */
    TemporaryPart writeTempPart(BlockWithPartition & block, const StorageMetadataPtr & metadata_snapshot, ContextPtr context);

    /// For insertion.
    static TemporaryPart writeProjectionPart(
        const MergeTreeData & data,
        Poco::Logger * log,
        Block block,
        const ProjectionDescription & projection,
        IMergeTreeDataPart * parent_part);

    /// For mutation: MATERIALIZE PROJECTION.
    static TemporaryPart writeTempProjectionPart(
        const MergeTreeData & data,
        Poco::Logger * log,
        Block block,
        const ProjectionDescription & projection,
        IMergeTreeDataPart * parent_part,
        size_t block_num);

    static Block mergeBlock(
        const Block & block,
        SortDescription sort_description,
        const Names & partition_key_columns,
        IColumn::Permutation *& permutation,
        const MergeTreeData::MergingParams & merging_params);

private:
    static TemporaryPart writeProjectionPartImpl(
        const String & part_name,
        bool is_temp,
        IMergeTreeDataPart * parent_part,
        const MergeTreeData & data,
        Poco::Logger * log,
        Block block,
        const ProjectionDescription & projection);

    MergeTreeData & data;
    Poco::Logger * log;
};

}
