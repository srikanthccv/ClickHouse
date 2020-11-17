#include <IO/MySQLBinlogEventReadBuffer.h>


namespace DB
{

MySQLBinlogEventReadBuffer::MySQLBinlogEventReadBuffer(ReadBuffer & in_, size_t checksum_signature_length_)
    : ReadBuffer(nullptr, 0, 0), in(in_), checksum_signature_length(checksum_signature_length_)
{
    if (checksum_signature_length)
        checksum_buf = new char[checksum_signature_length];

    nextIfAtEnd();
}

bool MySQLBinlogEventReadBuffer::nextImpl()
{
    if (hasPendingData())
        return true;

    if (in.eof())
        return false;

    if (checksum_buff_size == checksum_buff_limit)
    {
        if (likely(in.available() > checksum_signature_length))
        {
            working_buffer = ReadBuffer::Buffer(in.position(), in.buffer().end() - checksum_signature_length);
            in.ignore(working_buffer.size());
            return true;
        }

        in.readStrict(checksum_buf, checksum_signature_length);
        checksum_buff_size = checksum_buff_limit = checksum_signature_length;
    }
    else
    {
        for (size_t index = 0; index < checksum_buff_size - checksum_buff_limit; ++index)
            checksum_buf[index] = checksum_buf[checksum_buff_limit + index];

        checksum_buff_size -= checksum_buff_limit;
        size_t read_bytes = checksum_signature_length - checksum_buff_size;
        in.readStrict(checksum_buf + checksum_buff_size, read_bytes);   /// Minimum checksum_signature_length bytes
        checksum_buff_size = checksum_buff_limit = checksum_signature_length;
    }

    if (in.eof())
        return false;

    if (in.available() < checksum_signature_length)
    {
        size_t left_move_size = checksum_signature_length - in.available();
        checksum_buff_limit = checksum_buff_size - left_move_size;
    }

    working_buffer = ReadBuffer::Buffer(checksum_buf, checksum_buf + checksum_buff_limit);
    return true;
}

MySQLBinlogEventReadBuffer::~MySQLBinlogEventReadBuffer()
{
    try
    {
        /// ignore last 4 bytes
        nextIfAtEnd();

        if (checksum_signature_length)
            delete[] checksum_buf;
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

}
