#include <IO/ReadBufferFromIStream.h>
#include <Common/Exception.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_READ_FROM_ISTREAM;
}

bool ReadBufferFromIStream::nextImpl()
{
    istr.read(internal_buffer.begin(), internal_buffer.size());
    size_t gcount = istr.gcount();

    if (!gcount)
    {
        if (istr.eof())
            return false;
        else
            throw Exception("Cannot read from istream", ErrorCodes::CANNOT_READ_FROM_ISTREAM);
    }
    else
        working_buffer.resize(gcount);

    return true;
}

ReadBufferFromIStream::ReadBufferFromIStream(std::istream & istr_, size_t size)
    : BufferWithOwnMemory<ReadBuffer>(size), istr(istr_)
{
}

}
