#pragma once

#if !defined(ARCADIA_BUILD)
#include <Common/config.h>
#endif

#include <Disks/IDiskRemote.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/ReadSettings.h>

namespace Aws
{
namespace S3
{
class S3Client;
}
}

namespace DB
{

class ReadBufferFromRemoteFSGather : public ReadBuffer
{
friend class ReadIndirectBufferFromRemoteFS;

public:
    explicit ReadBufferFromRemoteFSGather(const RemoteMetadata & metadata_, const String & path_);

    String getFileName() const;

    void reset();

    void seek(off_t offset); /// SEEK_SET only.

    void setReadUntilPosition(size_t position) override;

    size_t readInto(char * data, size_t size, size_t offset, size_t ignore = 0);

protected:
    virtual SeekableReadBufferPtr createImplementationBuffer(const String & path, size_t offset) const = 0;

    RemoteMetadata metadata;

private:
    bool nextImpl() override;

    void initialize();

    bool readImpl();

    SeekableReadBufferPtr current_buf;

    size_t current_buf_idx = 0;

    size_t absolute_position = 0;

    size_t buf_idx = 0;

    size_t bytes_to_ignore = 0;

    size_t last_offset = 0;

    String canonical_path;
};


#if USE_AWS_S3
/// Reads data from S3 using stored paths in metadata.
class ReadBufferFromS3Gather final : public ReadBufferFromRemoteFSGather
{
public:
    ReadBufferFromS3Gather(
        const String & path_,
        std::shared_ptr<Aws::S3::S3Client> client_ptr_,
        const String & bucket_,
        IDiskRemote::Metadata metadata_,
        size_t max_single_read_retries_,
        const ReadSettings & settings_,
        bool threadpool_read_ = false)
        : ReadBufferFromRemoteFSGather(metadata_, path_)
        , client_ptr(std::move(client_ptr_))
        , bucket(bucket_)
        , max_single_read_retries(max_single_read_retries_)
        , settings(settings_)
        , threadpool_read(threadpool_read_)
    {
    }

    SeekableReadBufferPtr createImplementationBuffer(const String & path, size_t last_offset) const override;

private:
    std::shared_ptr<Aws::S3::S3Client> client_ptr;
    String bucket;
    UInt64 max_single_read_retries;
    ReadSettings settings;
    bool threadpool_read;
};
#endif


class ReadBufferFromWebServerGather final : public ReadBufferFromRemoteFSGather
{
public:
    ReadBufferFromWebServerGather(
            const String & path_,
            const String & uri_,
            RemoteMetadata metadata_,
            ContextPtr context_,
            size_t threadpool_read_,
            const ReadSettings & settings_)
        : ReadBufferFromRemoteFSGather(metadata_, path_)
        , uri(uri_)
        , context(context_)
        , threadpool_read(threadpool_read_)
        , settings(settings_)
    {
    }

    SeekableReadBufferPtr createImplementationBuffer(const String & path, size_t last_offset) const override;

private:
    String uri;
    ContextPtr context;
    bool threadpool_read;
    ReadSettings settings;
};


#if USE_HDFS
/// Reads data from HDFS using stored paths in metadata.
class ReadBufferFromHDFSGather final : public ReadBufferFromRemoteFSGather
{
public:
    ReadBufferFromHDFSGather(
            const String & path_,
            const Poco::Util::AbstractConfiguration & config_,
            const String & hdfs_uri_,
            IDiskRemote::Metadata metadata_,
            size_t buf_size_)
        : ReadBufferFromRemoteFSGather(metadata_, path_)
        , config(config_)
        , buf_size(buf_size_)
    {
        const size_t begin_of_path = hdfs_uri_.find('/', hdfs_uri_.find("//") + 2);
        hdfs_directory = hdfs_uri_.substr(begin_of_path);
        hdfs_uri = hdfs_uri_.substr(0, begin_of_path);
    }

    SeekableReadBufferPtr createImplementationBuffer(const String & path, size_t last_offset) const override;

private:
    const Poco::Util::AbstractConfiguration & config;
    String hdfs_uri;
    String hdfs_directory;
    size_t buf_size;
};
#endif

}
