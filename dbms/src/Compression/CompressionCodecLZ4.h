#pragma once

#include <IO/WriteBuffer.h>
#include <Compression/ICompressionCodec.h>
#include <IO/BufferWithOwnMemory.h>
#include <Parsers/StringRange.h>
#include <IO/LZ4_decompress_faster.h>

namespace DB
{

class CompressionCodecLZ4 : public ICompressionCodec
{
public:
    UInt8 getMethodByte() const override;

    String getCodecDesc() const override;

private:

    UInt32 doCompressData(const char * source, UInt32 source_size, char * dest) const override;

    void doDecompressData(const char * source, UInt32 source_size, char * dest, UInt32 uncompressed_size) const override;

    UInt32 getCompressedDataSize(UInt32 uncompressed_size) const override;

    mutable LZ4::PerformanceStatistics lz4_stat;
};

}
