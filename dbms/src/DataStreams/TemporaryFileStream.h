#pragma once

#include <Common/ClickHouseRevision.h>
#include <DataStreams/IBlockInputStream.h>
#include <DataStreams/NativeBlockInputStream.h>
#include <DataStreams/NativeBlockOutputStream.h>
#include <DataStreams/copyData.h>
#include <Compression/CompressedReadBuffer.h>
#include <Compression/CompressedWriteBuffer.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/WriteBufferFromFile.h>

namespace DB
{

/// To read the data that was flushed into the temporary data file.
struct TemporaryFileStream
{
    ReadBufferFromFile file_in;
    CompressedReadBuffer compressed_in;
    BlockInputStreamPtr block_in;

    TemporaryFileStream(const std::string & path)
        : file_in(path)
        , compressed_in(file_in)
        , block_in(std::make_shared<NativeBlockInputStream>(compressed_in, ClickHouseRevision::get()))
    {}

    TemporaryFileStream(const std::string & path, const Block & header_)
        : file_in(path)
        , compressed_in(file_in)
        , block_in(std::make_shared<NativeBlockInputStream>(compressed_in, header_, 0))
    {}

    /// Flush data from input stream into file for future reading
    static void write(const std::string & path, const Block & header, IBlockInputStream & input, std::atomic<bool> * is_cancelled = nullptr)
    {
        WriteBufferFromFile file_buf(path);
        CompressedWriteBuffer compressed_buf(file_buf);
        NativeBlockOutputStream output(compressed_buf, 0, header);
        copyData(input, output, is_cancelled);
    }
};

class TemporaryFileLazyInputStream : public IBlockInputStream
{
public:
    TemporaryFileLazyInputStream(const std::string & path_, const Block & header_)
        : path(path_)
        , header(header_)
        , done(false)
    {}

    String getName() const override { return "TemporaryFile"; }
    Block getHeader() const override { return header; }
    void readSuffix() override {}

protected:
    Block readImpl() override
    {
        if (!done)
        {
            done = true;
            TemporaryFileStream stream(path, header);
            return stream.block_in->read();
        }
        return {};
    }

private:
    const std::string path;
    Block header;
    bool done;
};

}
