#pragma once

#include <Storages/StorageS3Settings.h>
#include "config.h"

#if USE_AWS_S3

#include <memory>

#include <IO/HTTPCommon.h>
#include <IO/ParallelReadBuffer.h>
#include <IO/ReadBuffer.h>
#include <IO/ReadSettings.h>
#include <IO/ReadBufferFromFileBase.h>
#include <IO/WithFileName.h>

#include <aws/s3/model/GetObjectResult.h>

namespace Aws::S3
{
class Client;
}

namespace DB
{
/**
 * Perform S3 HTTP GET request and provide response to read.
 */
class ReadBufferFromS3 : public ReadBufferFromFileBase
{
private:
    std::shared_ptr<const S3::Client> client_ptr;
    String bucket;
    String key;
    String version_id;
    const S3Settings::RequestSettings request_settings;

    /// These variables are atomic because they can be used for `logging only`
    /// (where it is not important to get consistent result)
    /// from separate thread other than the one which uses the buffer for s3 reading.
    std::atomic<off_t> offset = 0;
    std::atomic<off_t> read_until_position = 0;

    Aws::S3::Model::GetObjectResult read_result;
    std::unique_ptr<ReadBuffer> impl;

    Poco::Logger * log = &Poco::Logger::get("ReadBufferFromS3");

public:
    ReadBufferFromS3(
        std::shared_ptr<const S3::Client> client_ptr_,
        const String & bucket_,
        const String & key_,
        const String & version_id_,
        const S3Settings::RequestSettings & request_settings_,
        const ReadSettings & settings_,
        bool use_external_buffer = false,
        size_t offset_ = 0,
        size_t read_until_position_ = 0,
        bool restricted_seek_ = false);

    bool nextImpl() override;

    off_t seek(off_t off, int whence) override;

    off_t getPosition() override;

    size_t getFileSize() override;

    void setReadUntilPosition(size_t position) override;
    void setReadUntilEnd() override;

    size_t getFileOffsetOfBufferEnd() const override { return offset; }

    bool supportsRightBoundedReads() const override { return true; }

    String getFileName() const override { return bucket + "/" + key; }

private:
    std::unique_ptr<ReadBuffer> initialize();

    // If true, if we destroy impl now, no work was wasted. Just for metrics.
    bool atEndOfRequestedRangeGuess();

    ReadSettings read_settings;

    bool use_external_buffer;

    /// There is different seek policy for disk seek and for non-disk seek
    /// (non-disk seek is applied for seekable input formats: orc, arrow, parquet).
    bool restricted_seek;
};

/// Creates separate ReadBufferFromS3 for sequence of ranges of particular object
class ReadBufferS3Factory : public ParallelReadBuffer::ReadBufferFactory, public WithFileName
{
public:
    explicit ReadBufferS3Factory(
        std::shared_ptr<const S3::Client> client_ptr_,
        const String & bucket_,
        const String & key_,
        const String & version_id_,
        size_t object_size_,
        const S3Settings::RequestSettings & request_settings_,
        const ReadSettings & read_settings_)
        : client_ptr(client_ptr_)
        , bucket(bucket_)
        , key(key_)
        , version_id(version_id_)
        , read_settings(read_settings_)
        , object_size(object_size_)
        , request_settings(request_settings_)
    {
        assert(range_step > 0);
        assert(range_step < object_size);
    }

    std::unique_ptr<SeekableReadBuffer> getReader() override;

    size_t getFileSize() override;

    String getFileName() const override { return bucket + "/" + key; }

private:
    std::shared_ptr<const S3::Client> client_ptr;
    const String bucket;
    const String key;
    const String version_id;
    ReadSettings read_settings;
    size_t object_size;
    const S3Settings::RequestSettings request_settings;
};

}

#endif
