#include <Storages/DataLakes/HudiMetadataParser.h>
#include <Common/logger_useful.h>
#include <ranges>
#include <Poco/String.h>
#include "config.h"
#include <filesystem>
#include <IO/ReadHelpers.h>

#if USE_AWS_S3
#include <Storages/DataLakes/S3MetadataReader.h>
#include <Storages/StorageS3.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{
    /**
     * Documentation links:
     * - https://hudi.apache.org/tech-specs/
     */

    /**
      * Hudi tables store metadata files and data files.
      * Metadata files are stored in .hoodie/metadata directory.
      * There can be two types of data files
      * 1. base files (columnar file formats like Apache Parquet/Orc)
      * 2. log files
      * Currently we support reading only `base files`.
      * Data file name format:
      * [File Id]_[File Write Token]_[Transaction timestamp].[File Extension]
      *
      * To find needed parts we need to find out latest part file for every partition.
      * Part format is usually parquet, but can differ.
      */
    Strings processMetadataFiles(const std::vector<String> & keys, const std::string & base_directory)
    {
        auto * log = &Poco::Logger::get("HudiMetadataParser");

        struct FileInfo
        {
            String filename;
            UInt64 timestamp;
        };
        std::unordered_map<String, FileInfo> latest_parts; /// Partition path (directory) -> latest part file info.

        /// For each partition path take only latest file.
        for (const auto & key : keys)
        {
            const auto delim = key.find_last_of('_') + 1;
            if (delim == std::string::npos)
                throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected format for file: {}", key);

            const auto timestamp = parse<UInt64>(key.substr(delim + 1));
            const auto file_path = key.substr(base_directory.size());

            LOG_TEST(log, "Having path {}", file_path);

            const auto [it, inserted] = latest_parts.emplace(/* partition_path */fs::path(key).parent_path(), FileInfo{});
            if (inserted)
                it->second = FileInfo{file_path, timestamp};
            else if (it->second.timestamp < timestamp)
                it->second = {file_path, timestamp};
        }

        LOG_TRACE(log, "Having {} result partitions", latest_parts.size());

        Strings result;
        for (const auto & [_, file_info] : latest_parts)
            result.push_back(file_info.filename);
        return result;
    }
}

template <typename Configuration, typename MetadataReadHelper>
Strings HudiMetadataParser<Configuration, MetadataReadHelper>::getFiles(const Configuration & configuration, ContextPtr)
{
    const Strings files = MetadataReadHelper::listFiles(configuration, "", Poco::toLower(configuration.format));
    return processMetadataFiles(files, configuration.getPath());
}

#if USE_AWS_S3
template Strings HudiMetadataParser<StorageS3::Configuration, S3DataLakeMetadataReadHelper>::getFiles(
    const StorageS3::Configuration & configuration, ContextPtr);
#endif

}
