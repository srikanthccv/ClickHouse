#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <time.h>
#include <filesystem>

#include <Common/Config/YAMLParser.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size)
{
    /// How to test:
    /// build ClickHouse with YAML_fuzzer.cpp
    /// ./YAML_fuzzer YAML_CORPUS
    /// where YAML_CORPUS is a directory with different YAML configs for libfuzzer
    char file_name[L_tmpnam];
    if (!std::tmpnam(file_name))
    {
        std::cerr << "Cannot create temp file!\n";
        return 1;
    }
    std::string input = std::string(reinterpret_cast<const char*>(data), size);

    {
        std::ofstream temp_file(file_name);
        temp_file << input;
    }

    try
    {
        DB::YAMLParserImpl::parse(std::string(file_name));
    }
    catch (...)
    {
        std::cerr << "YAML_fuzzer failed: " << DB::getCurrentExceptionMessage(__PRETTY_FUNCTION__) << std::endl;
        return 1;
    }
    return 0;
}
