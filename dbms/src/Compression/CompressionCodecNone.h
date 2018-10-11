#pragma once

#include <IO/WriteBuffer.h>
#include <Compression/ICompressionCodec.h>
#include <IO/BufferWithOwnMemory.h>
#include <Parsers/StringRange.h>

namespace DB
{

class CompressionCodecNone : public ICompressionCodec
{
public:
    char getMethodByte() override;

    void getCodecDesc(String & codec_desc) override;

    size_t compress(char * source, size_t source_size, char * compressed_buf) override;

    size_t decompress(char *source, size_t source_size, char *dest, size_t decompressed_size) override;
};

}