#include <Storages/MergeTree/IMergedBlockOutputStream.h>
#include <Storages/MergeTree/MergeTreeIOSettings.h>
#include <Storages/MergeTree/IMergeTreeDataPartWriter.h>

namespace DB
{
IMergedBlockOutputStream::IMergedBlockOutputStream(
    const MergeTreeDataPartPtr & data_part,
    const StorageMetadataPtr & metadata_snapshot_)
    : storage(data_part->storage)
    , metadata_snapshot(metadata_snapshot_)
    , volume(data_part->volume)
    , part_path(data_part->isStoredOnDisk() ? data_part->getFullRelativePath() : "")
    , new_serialization_info(data_part->storage.getSettings()->ratio_of_defaults_for_sparse_serialization)
{
}

NameSet IMergedBlockOutputStream::removeEmptyColumnsFromPart(
    const MergeTreeDataPartPtr & data_part,
    NamesAndTypesList & columns,
    MergeTreeData::DataPart::Checksums & checksums)
{
    const NameSet & empty_columns = data_part->expired_columns;

    /// For compact part we have to override whole file with data, it's not
    /// worth it
    if (empty_columns.empty() || isCompactPart(data_part))
        return {};

    /// Collect counts for shared streams of different columns. As an example, Nested columns have shared stream with array sizes.
    std::map<String, size_t> stream_counts;
    std::unordered_map<String, SerializationPtr> serialziations;
    for (const NameAndTypePair & column : columns)
    {
        auto serialization = IDataType::getSerialization(column, [&](const String & stream_name)
        {
            /// Checksums of data_part are not initialized here.
            return checksums.files.count(stream_name + IMergeTreeDataPart::DATA_FILE_EXTENSION) != 0;
        });

        serialization->enumerateStreams(
            [&](const ISerialization::SubstreamPath & substream_path)
            {
                ++stream_counts[ISerialization::getFileNameForStream(column, substream_path)];
            },
            {});

        serialziations[column.name] = std::move(serialization);
    }

    NameSet remove_files;
    const String mrk_extension = data_part->getMarksFileExtension();
    for (const auto & column_name : empty_columns)
    {
        auto column_with_type = columns.tryGetByName(column_name);
        if (!column_with_type)
           continue;

        ISerialization::StreamCallback callback = [&](const ISerialization::SubstreamPath & substream_path)
        {
            String stream_name = ISerialization::getFileNameForStream(*column_with_type, substream_path);
            /// Delete files if they are no longer shared with another column.
            if (--stream_counts[stream_name] == 0)
            {
                remove_files.emplace(stream_name + ".bin");
                remove_files.emplace(stream_name + mrk_extension);
            }
        };

        serialziations[column_name]->enumerateStreams(callback);
    }

    /// Remove files on disk and checksums
    for (const String & removed_file : remove_files)
    {
        if (checksums.files.count(removed_file))
        {
            data_part->volume->getDisk()->removeFile(data_part->getFullRelativePath() + removed_file);
            checksums.files.erase(removed_file);
        }
    }

    /// Remove columns from columns array
    for (const String & empty_column_name : empty_columns)
    {
        auto find_func = [&empty_column_name](const auto & pair) -> bool
        {
            return pair.name == empty_column_name;
        };
        auto remove_it
            = std::find_if(columns.begin(), columns.end(), find_func);

        if (remove_it != columns.end())
            columns.erase(remove_it);
    }
    return remove_files;
}

}
