#include <Core/SortCursor.h>
#include <Interpreters/SortedBlocksWriter.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Processors/Executors/PullingPipelineExecutor.h>
#include <Processors/Sources/SourceFromSingleChunk.h>
#include <Processors/Merges/MergingSortedTransform.h>
#include <Processors/Sources/TemporaryFileLazySource.h>
#include <Formats/TemporaryFileStream.h>
#include <Disks/IVolume.h>
#include <Disks/TemporaryFileOnDisk.h>


namespace ProfileEvents
{
    extern const Event ExternalJoinWritePart;
    extern const Event ExternalJoinMerge;
    extern const Event ExternalJoinCompressedBytes;
    extern const Event ExternalJoinUncompressedBytes;
    extern const Event ExternalProcessingCompressedBytesTotal;
    extern const Event ExternalProcessingUncompressedBytesTotal;
}

namespace CurrentMetrics
{
    extern const Metric TemporaryFilesForJoin;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{

TemporaryFileOnDiskHolder flushToFile(const DiskPtr & disk, const Block & header, QueryPipelineBuilder pipeline, const String & codec)
{
    auto tmp_file = std::make_unique<TemporaryFileOnDisk>(disk, CurrentMetrics::TemporaryFilesForJoin);
    auto write_stat = TemporaryFileStream::write(tmp_file->getPath(), header, std::move(pipeline), codec);

    ProfileEvents::increment(ProfileEvents::ExternalProcessingCompressedBytesTotal, write_stat.compressed_bytes);
    ProfileEvents::increment(ProfileEvents::ExternalProcessingUncompressedBytesTotal, write_stat.uncompressed_bytes);

    ProfileEvents::increment(ProfileEvents::ExternalJoinCompressedBytes, write_stat.compressed_bytes);
    ProfileEvents::increment(ProfileEvents::ExternalJoinUncompressedBytes, write_stat.uncompressed_bytes);
    ProfileEvents::increment(ProfileEvents::ExternalJoinWritePart);

    return tmp_file;
}

SortedBlocksWriter::SortedFiles flushToManyFiles(const DiskPtr & disk, const Block & header, QueryPipelineBuilder builder,
                                                 const String & codec, std::function<void(const Block &)> callback = [](const Block &){})
{
    std::vector<TemporaryFileOnDiskHolder> files;
    auto pipeline = QueryPipelineBuilder::getPipeline(std::move(builder));
    PullingPipelineExecutor executor(pipeline);

    Block block;
    while (executor.pull(block))
    {
        if (!block.rows())
            continue;

        callback(block);

        QueryPipelineBuilder one_block_pipeline;
        Chunk chunk(block.getColumns(), block.rows());
        one_block_pipeline.init(Pipe(std::make_shared<SourceFromSingleChunk>(block.cloneEmpty(), std::move(chunk))));
        auto tmp_file = flushToFile(disk, header, std::move(one_block_pipeline), codec);
        files.emplace_back(std::move(tmp_file));
    }

    return files;
}

}


void SortedBlocksWriter::insert(Block && block)
{
    bool can_insert_more = false;
    bool has_data_to_flush = false;

    BlocksList current_blocks;
    size_t row_count = 0;
    size_t bytes = 0;
    size_t flush_no = 0;

    if (!block.rows())
        return;

    {
        std::lock_guard lock{insert_mutex};

        /// insert block into BlocksList under lock
        inserted_blocks.insert(std::move(block));

        size_t total_row_count = inserted_blocks.row_count + row_count_in_flush;
        size_t total_bytes = inserted_blocks.bytes + bytes_in_flush;

        can_insert_more = size_limits.softCheck(total_row_count, total_bytes);
        has_data_to_flush = !size_limits.softCheck(inserted_blocks.row_count * num_streams, inserted_blocks.bytes * num_streams);

        if (has_data_to_flush)
        {
            ++flush_inflight;
            current_blocks.swap(inserted_blocks.blocks);
            row_count_in_flush = total_row_count;
            bytes_in_flush = total_bytes;

            row_count = inserted_blocks.row_count;
            bytes = inserted_blocks.bytes;
            inserted_blocks.clear();
        }
        else if (can_insert_more)
            flush_no = flush_number;
    }

    if (has_data_to_flush)
    {
        /// flush new blocks without lock
        auto flushed = flush(current_blocks);
        current_blocks.clear();

        std::lock_guard lock{insert_mutex};

        sorted_files.emplace_back(std::move(flushed));
        row_count_in_flush -= row_count;
        bytes_in_flush -= bytes;

        /// notify another insert (flush_number) and merge (flush_inflight)
        ++flush_number;
        --flush_inflight;
        flush_condvar.notify_all();
    }
    else if (!can_insert_more)
    {
        /// wakeup insert blocked by out of limit
        std::unique_lock lock{insert_mutex};
        flush_condvar.wait(lock, [&]{ return flush_no < flush_number; });
    }
}

SortedBlocksWriter::TmpFilePtr SortedBlocksWriter::flush(const BlocksList & blocks) const
{
    Pipes pipes;
    pipes.reserve(blocks.size());
    for (const auto & block : blocks)
        if (auto num_rows = block.rows())
            pipes.emplace_back(std::make_shared<SourceFromSingleChunk>(block.cloneEmpty(), Chunk(block.getColumns(), num_rows)));

    if (pipes.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Empty block");

    QueryPipelineBuilder pipeline;
    pipeline.init(Pipe::unitePipes(std::move(pipes)));

    if (pipeline.getNumStreams() > 1)
    {
        auto transform = std::make_shared<MergingSortedTransform>(
            pipeline.getHeader(),
            pipeline.getNumStreams(),
            sort_description,
            rows_in_block,
            SortingQueueStrategy::Default);

        pipeline.addTransform(std::move(transform));
    }

    return flushToFile(volume->getDisk(), sample_block, std::move(pipeline), codec);
}

SortedBlocksWriter::PremergedFiles SortedBlocksWriter::premerge()
{
    SortedFiles files;
    BlocksList blocks;

    /// wait other flushes if any
    {
        std::unique_lock lock{insert_mutex};

        files.swap(sorted_files);
        blocks.swap(inserted_blocks.blocks);
        inserted_blocks.clear();

        flush_condvar.wait(lock, [&]{ return !flush_inflight; });
    }

    /// flush not flushed
    if (!blocks.empty())
        files.emplace_back(flush(blocks));

    Pipes pipes;
    pipes.reserve(num_files_for_merge);

    /// Merge by parts to save memory. It's possible to exchange disk I/O and memory by num_files_for_merge.
    {
        SortedFiles new_files;
        new_files.reserve(files.size() / num_files_for_merge + 1);

        while (files.size() > num_files_for_merge)
        {
            for (const auto & file : files)
            {
                pipes.emplace_back(streamFromFile(file));

                if (pipes.size() == num_files_for_merge || &file == &files.back())
                {
                    QueryPipelineBuilder pipeline;
                    pipeline.init(Pipe::unitePipes(std::move(pipes)));
                    pipes = Pipes();

                    if (pipeline.getNumStreams() > 1)
                    {
                        auto transform = std::make_shared<MergingSortedTransform>(
                            pipeline.getHeader(),
                            pipeline.getNumStreams(),
                            sort_description,
                            rows_in_block,
                            SortingQueueStrategy::Default);

                        pipeline.addTransform(std::move(transform));
                    }

                    new_files.emplace_back(flushToFile(volume->getDisk(), sample_block, std::move(pipeline), codec));
                }
            }

            files.clear();
            files.swap(new_files);
        }

        for (const auto & file : files)
            pipes.emplace_back(streamFromFile(file));
    }

    return PremergedFiles{std::move(files), Pipe::unitePipes(std::move(pipes))};
}

SortedBlocksWriter::SortedFiles SortedBlocksWriter::finishMerge(std::function<void(const Block &)> callback)
{
    PremergedFiles files = premerge();
    QueryPipelineBuilder pipeline;
    pipeline.init(std::move(files.pipe));

    if (pipeline.getNumStreams() > 1)
    {
        ProfileEvents::increment(ProfileEvents::ExternalJoinMerge);
        auto transform = std::make_shared<MergingSortedTransform>(
            pipeline.getHeader(),
            pipeline.getNumStreams(),
            sort_description,
            rows_in_block,
            SortingQueueStrategy::Default);

        pipeline.addTransform(std::move(transform));
    }

    return flushToManyFiles(volume->getDisk(), sample_block, std::move(pipeline), codec, callback);
}

Pipe SortedBlocksWriter::streamFromFile(const TmpFilePtr & file) const
{
    return Pipe(std::make_shared<TemporaryFileLazySource>(file->path(), materializeBlock(sample_block)));
}


Block SortedBlocksBuffer::exchange(Block && block)
{
    static constexpr const double reserve_coefficient = 1.2;

    Blocks out_blocks;
    Block empty_out = block.cloneEmpty();

    {
        std::lock_guard lock(mutex);

        if (block)
        {
            current_bytes += block.bytes();
            buffer.emplace_back(std::move(block));

            /// Saved. Return empty block with same structure.
            if (current_bytes < max_bytes)
                return empty_out;
        }

        /// Not saved. Return buffered.
        out_blocks.swap(buffer);
        buffer.reserve(static_cast<size_t>(out_blocks.size() * reserve_coefficient));
        current_bytes = 0;
    }

    if (size_t size = out_blocks.size())
    {
        if (size == 1)
            return out_blocks[0];
        return mergeBlocks(std::move(out_blocks));
    }

    return {};
}

Block SortedBlocksBuffer::mergeBlocks(Blocks && blocks) const
{
    size_t num_rows = 0;

    { /// Merge sort blocks
        Pipes pipes;
        pipes.reserve(blocks.size());

        for (auto & block : blocks)
        {
            num_rows += block.rows();
            Chunk chunk(block.getColumns(), block.rows());
            pipes.emplace_back(std::make_shared<SourceFromSingleChunk>(block.cloneEmpty(), std::move(chunk)));
        }

        Blocks tmp_blocks;

        QueryPipelineBuilder builder;
        builder.init(Pipe::unitePipes(std::move(pipes)));

        if (builder.getNumStreams() > 1)
        {
            auto transform = std::make_shared<MergingSortedTransform>(
                builder.getHeader(),
                builder.getNumStreams(),
                sort_description,
                num_rows,
                SortingQueueStrategy::Default);

            builder.addTransform(std::move(transform));
        }

        auto pipeline = QueryPipelineBuilder::getPipeline(std::move(builder));
        PullingPipelineExecutor executor(pipeline);
        Block block;
        while (executor.pull(block))
            tmp_blocks.emplace_back(block);

        blocks.swap(tmp_blocks);
    }

    if (blocks.size() == 1)
        return blocks[0];

    return concatenateBlocks(blocks);
}

}
