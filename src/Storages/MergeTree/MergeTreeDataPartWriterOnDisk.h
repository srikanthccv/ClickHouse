#pragma once

#include <Storages/MergeTree/IMergeTreeDataPartWriter.h>
#include <IO/WriteBufferFromFile.h>
#include <IO/WriteBufferFromFileBase.h>
#include <Compression/CompressedWriteBuffer.h>
#include <IO/HashingWriteBuffer.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/IMergeTreeDataPart.h>
#include <Disks/IDisk.h>
#include <Parsers/ExpressionElementParsers.h>
#include <Parsers/parseQuery.h>

namespace DB
{

/// Single unit for writing data to disk. Contains information about
/// amount of rows to write and marks.
struct Granule
{
    /// Start row in block for granule
    size_t start_row;
    /// Amount of rows from block which have to be written to disk from start_row
    size_t rows_to_write;
    /// Global mark number in the list of all marks (index_granularity) for this part
    size_t mark_number;
    /// Should writer write mark for the first of this granule to disk.
    /// NOTE: Sometimes we don't write mark for the start row, because
    /// this granule can be continuation of the previous one.
    bool mark_on_start;
    /// if true: When this granule will be written to disk all rows for corresponding mark will
    /// be wrtten. It doesn't mean that rows_to_write == index_granularity.getMarkRows(mark_number),
    /// We may have a lot of small blocks between two marks and this may be the last one.
    bool is_complete;
};

/// Multiple granules to write for concrete block.
using Granules = std::vector<Granule>;

/// Writes data part to disk in different formats.
/// Calculates and serializes primary and skip indices if needed.
class MergeTreeDataPartWriterOnDisk : public IMergeTreeDataPartWriter
{
public:
    using WrittenOffsetColumns = std::set<std::string>;

    /// Helper class, which holds chain of buffers to write data file with marks.
    /// It is used to write: one column, skip index or all columns (in compact format).
    struct Stream
    {
        Stream(
            const String & escaped_column_name_,
            const MutableDataPartStoragePtr & data_part_storage,
            const String & data_path_,
            const std::string & data_file_extension_,
            const std::string & marks_path_,
            const std::string & marks_file_extension_,
            const CompressionCodecPtr & compression_codec_,
            size_t max_compress_block_size_,
            const CompressionCodecPtr & marks_compression_codec_,
            size_t marks_compress_block_size_,
            const WriteSettings & query_write_settings);

        String escaped_column_name;
        std::string data_file_extension;
        std::string marks_file_extension;

        /// compressed_hashing -> compressor -> plain_hashing -> plain_file
        std::unique_ptr<WriteBufferFromFileBase> plain_file;
        HashingWriteBuffer plain_hashing;
        CompressedWriteBuffer compressor;
        HashingWriteBuffer compressed_hashing;

        /// marks_compressed_hashing -> marks_compressor -> marks_hashing -> marks_file
        std::unique_ptr<WriteBufferFromFileBase> marks_file;
        HashingWriteBuffer marks_hashing;
        CompressedWriteBuffer marks_compressor;
        HashingWriteBuffer marks_compressed_hashing;
        bool compress_marks;

        bool is_prefinalized = false;

        void preFinalize();

        void finalize();

        void sync() const;

        void addToChecksums(IMergeTreeDataPart::Checksums & checksums);
    };

    using StreamPtr = std::unique_ptr<Stream>;

    MergeTreeDataPartWriterOnDisk(
        const MergeTreeMutableDataPartPtr & data_part_,
        const NamesAndTypesList & columns_list,
        const StorageMetadataPtr & metadata_snapshot_,
        const std::vector<MergeTreeIndexPtr> & indices_to_recalc,
        const String & marks_file_extension,
        const CompressionCodecPtr & default_codec,
        const MergeTreeWriterSettings & settings,
        const MergeTreeIndexGranularity & index_granularity);

    void setWrittenOffsetColumns(WrittenOffsetColumns * written_offset_columns_)
    {
        written_offset_columns = written_offset_columns_;
    }

protected:
     /// Count index_granularity for block and store in `index_granularity`
    size_t computeIndexGranularity(const Block & block) const;

    /// Write primary index according to granules_to_write
    void calculateAndSerializePrimaryIndex(const Block & primary_index_block, const Granules & granules_to_write);
    /// Write skip indices according to granules_to_write. Skip indices also have their own marks
    /// and one skip index granule can contain multiple "normal" marks. So skip indices serialization
    /// require additional state: skip_indices_aggregators and skip_index_accumulated_marks
    void calculateAndSerializeSkipIndices(const Block & skip_indexes_block, const Granules & granules_to_write);

    /// Finishes primary index serialization: write final primary index row (if required) and compute checksums
    void fillPrimaryIndexChecksums(MergeTreeData::DataPart::Checksums & checksums);
    void finishPrimaryIndexSerialization(bool sync);
    /// Finishes skip indices serialization: write all accumulated data to disk and compute checksums
    void fillSkipIndicesChecksums(MergeTreeData::DataPart::Checksums & checksums);
    void finishSkipIndicesSerialization(bool sync);

    /// Get global number of the current which we are writing (or going to start to write)
    size_t getCurrentMark() const { return current_mark; }

    void setCurrentMark(size_t mark) { current_mark = mark; }

    /// Get unique non ordered skip indices column.
    Names getSkipIndicesColumns() const;

    const MergeTreeIndices skip_indices;

    const String marks_file_extension;
    const CompressionCodecPtr default_codec;

    const bool compute_granularity;

    std::vector<StreamPtr> skip_indices_streams;
    MergeTreeIndexAggregators skip_indices_aggregators;
    std::vector<size_t> skip_index_accumulated_marks;

    std::unique_ptr<WriteBufferFromFileBase> index_file_stream;
    std::unique_ptr<HashingWriteBuffer> index_file_hashing_stream;
    std::unique_ptr<CompressedWriteBuffer> index_compressor_stream;
    std::unique_ptr<HashingWriteBuffer> index_source_hashing_stream;
    bool compress_primary_key;

    DataTypes index_types;
    /// Index columns from the last block
    /// It's written to index file in the `writeSuffixAndFinalizePart` method
    Columns last_block_index_columns;

    bool data_written = false;

    /// To correctly write Nested elements column-by-column.
    WrittenOffsetColumns * written_offset_columns = nullptr;

    /// Data is already written up to this mark.
    size_t current_mark = 0;

    GinIndexStoreFactory::GinIndexStores gin_index_stores;
private:
    void initSkipIndices();
    void initPrimaryIndex();

    virtual void fillIndexGranularity(size_t index_granularity_for_block, size_t rows_in_block) = 0;
};

}
