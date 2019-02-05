#include "BrotliReadBuffer.h"

namespace DB
{
BrotliReadBuffer::BrotliReadBuffer(ReadBuffer &in_, size_t buf_size, char *existing_memory, size_t alignment)
        : BufferWithOwnMemory<ReadBuffer>(buf_size, existing_memory, alignment)
        , in(in_)
        , bstate(BrotliDecoderCreateInstance(nullptr, nullptr, nullptr))
        , bresult(BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT)
        , in_available(0)
        , in_data(nullptr)
        , out_capacity(0)
        , out_data(nullptr)
        , eof(false)
{
}

BrotliReadBuffer::~BrotliReadBuffer()
{
    BrotliDecoderDestroyInstance(bstate);
}

bool BrotliReadBuffer::nextImpl()
{
    if (eof)
        return false;

    if (!in_available)
    {
        in.nextIfAtEnd();
        in_available = in.buffer().end() - in.position();
        in_data = reinterpret_cast<uint8_t *>(in.position());
    }

    if (bresult == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT && (!in_available || in.eof()))
    {
        throw Exception(std::string("brotli decode error"), ErrorCodes::CANNOT_READ_ALL_DATA);
    }

    out_capacity = internal_buffer.size();
    out_data = reinterpret_cast<uint8_t *>(internal_buffer.begin());

    bresult = BrotliDecoderDecompressStream(bstate, &in_available, &in_data, &out_capacity, &out_data, nullptr);

    in.position() = in.buffer().end() - in_available;
    working_buffer.resize(internal_buffer.size() - out_capacity);

    if (bresult == BROTLI_DECODER_RESULT_SUCCESS)
    {
        if (in.eof())
        {
            eof = true;
            return working_buffer.size() != 0;
        }
        else
        {
            throw Exception(std::string("brotli decode error"), ErrorCodes::CANNOT_READ_ALL_DATA);
        }
    }

    if (bresult == BROTLI_DECODER_RESULT_ERROR)
    {
        throw Exception(std::string("brotli decode error"), ErrorCodes::CANNOT_READ_ALL_DATA);
    }

    return true;
}
}