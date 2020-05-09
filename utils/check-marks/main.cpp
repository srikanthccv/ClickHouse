#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <Compression/CompressedWriteBuffer.h>
#include <Compression/CompressedReadBuffer.h>
#include <IO/WriteHelpers.h>
#include <IO/Operators.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteBufferFromFileDescriptor.h>
#include <Compression/CompressedReadBufferFromFile.h>


/** This program checks correctness of .mrk (marks) file for corresponding compressed .bin file.
  */


namespace DB
{
    namespace ErrorCodes
    {
        extern const int TOO_LARGE_SIZE_COMPRESSED;
    }
}


static void checkByCompressedReadBuffer(const std::string & mrk_path, const std::string & bin_path)
{
    DB::ReadBufferFromFile mrk_in(mrk_path);
    DB::CompressedReadBufferFromFile bin_in(bin_path, 0, 0, 0);

    DB::WriteBufferFromFileDescriptor out(STDOUT_FILENO);
    bool mrk2_format = boost::algorithm::ends_with(mrk_path, ".mrk2");

    for (size_t mark_num = 0; !mrk_in.eof(); ++mark_num)
    {
        UInt64 offset_in_compressed_file = 0;
        UInt64 offset_in_decompressed_block = 0;
        UInt64 index_granularity_rows = 0;

        DB::readBinary(offset_in_compressed_file, mrk_in);
        DB::readBinary(offset_in_decompressed_block, mrk_in);

        out << "Mark " << mark_num << ", points to " << offset_in_compressed_file << ", " << offset_in_decompressed_block;

        if (mrk2_format)
        {
            DB::readBinary(index_granularity_rows, mrk_in);

            out << ", has rows after " << index_granularity_rows;
        }

        out << ".\n" << DB::flush;

        bin_in.seek(offset_in_compressed_file, offset_in_decompressed_block);
    }
}


int main(int argc, char ** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
    ;

    boost::program_options::variables_map options;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), options);

    if (options.count("help") || argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " file.mrk file.bin" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    try
    {
        checkByCompressedReadBuffer(argv[1], argv[2]);
    }
    catch (const DB::Exception & e)
    {
        std::cerr << e.what() << ", " << e.message() << std::endl
            << std::endl
            << "Stack trace:" << std::endl
            << e.getStackTraceString()
            << std::endl;
        throw;
    }

    return 0;
}
