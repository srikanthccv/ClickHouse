#pragma once

#include <IO/WriteBufferFromFileDecorator.h>
#include <IO/WriteSettings.h>
#include <Common/IFileCache.h>


namespace DB
{

class FileSegmentRangeWriter;

class CachedWriteBufferFromFile final : public WriteBufferFromFileDecorator
{
public:
    CachedWriteBufferFromFile(
        std::unique_ptr<WriteBuffer> impl_,
        FileCachePtr cache_,
        const String & source_path_,
        const IFileCache::Key & key_,
        bool is_persistent_cache_file_,
        const String & query_id_,
        const WriteSettings & settings_);

    void nextImpl() override;

    void preFinalize() override;

private:
    void cacheData(char * data, size_t size);
    void appendFilesystemCacheLog(const FileSegmentPtr & file_segment);

    FileCachePtr cache;
    String source_path;
    IFileCache::Key key;

    bool is_persistent_cache_file;
    size_t current_download_offset = 0;
    const String query_id;
    bool enable_cache_log;

    ProfileEvents::Counters current_file_segment_counters;
    std::unique_ptr<FileSegmentRangeWriter> cache_writer;
};

}
