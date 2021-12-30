#pragma once
#include <Common/StringUtils/StringUtils.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/ReadBufferFromString.h>
#include <IO/ReadHelpers.h>
#include <IO/readFloatText.h>
#include <IO/Operators.h>
#include <IO/ZstdInflatingReadBuffer.h>

#include <Common/Arena.h>
#include <base/StringRef.h>
#include <Common/HashTable/HashMap.h>

#include <string_view>
#include <string>
#include <cstring>
#include <unordered_map>
#include <base/logger_useful.h>
#include <Common/getResource.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int FILE_DOESNT_EXIST;
}

class FrequencyHolder
{

public:
    struct Language
    {
        String name;
        HashMap<StringRef, Float64> map;
    };

    struct Encoding
    {
        String name;
        HashMap<UInt16, Float64> map;
    };

public:
    using Map = HashMap<StringRef, Float64>;
    using Container = std::vector<Language>;
    using EncodingMap = HashMap<UInt16, Float64>;
    using EncodingContainer = std::vector<Encoding>;

    static FrequencyHolder & getInstance()
    {
        static FrequencyHolder instance;
        return instance;
    }


    void loadEncodingsFrequency()
    {
        Poco::Logger * log = &Poco::Logger::get("EncodingsFrequency");

        LOG_TRACE(log, "Loading embedded charset frequencies");

        auto resource = getResource("charset_freq.txt.zst");
            if (resource.empty())
                throw Exception(ErrorCodes::FILE_DOESNT_EXIST, "There is no embedded charset frequencies");

        String line;
        UInt16 bigram;
        Float64 frequency;
        String charset_name;

        auto buf = std::make_unique<ReadBufferFromMemory>(resource.data(), resource.size());
        std::unique_ptr<ReadBuffer> in = std::make_unique<ZstdInflatingReadBuffer>(std::move(buf));

        while (!in->eof())
        {
            readString(line, *in);
            ++in->position();

            if (line.empty())
                continue;

            ReadBufferFromString buf_line(line);

            // Start loading a new charset
            if (line.starts_with("//"))
            {
                buf_line.ignore(3);
                readString(charset_name, buf_line);

                Encoding enc;
                enc.name = charset_name;
                encodings_freq.push_back(std::move(enc));
            }
            else
            {
                readIntText(bigram, buf_line);
                buf_line.ignore();
                readFloatText(frequency, buf_line);

                encodings_freq.back().map[bigram] = frequency;
            }
        }
        LOG_TRACE(log, "Charset frequencies was added, charsets count: {}", encodings_freq.size());
    }


    void loadEmotionalDict()
    {
        Poco::Logger * log = &Poco::Logger::get("EmotionalDict");
        LOG_TRACE(log, "Loading embedded emotional dictionary (RU)");

        auto resource = getResource("emotional_dictionary_rus.txt.zst");
            if (resource.empty())
                throw Exception(ErrorCodes::FILE_DOESNT_EXIST, "There is no embedded emotional dictionary");

        String line;
        String word;
        Float64 tonality;
        size_t count = 0;

        auto buf = std::make_unique<ReadBufferFromMemory>(resource.data(), resource.size());
        std::unique_ptr<ReadBuffer> in = std::make_unique<ZstdInflatingReadBuffer>(std::move(buf));

        while (!in->eof())
        {
            readString(line, *in);
            ++in->position();

            if (line.empty())
                continue;

            ReadBufferFromString buf_line(line);

            readStringUntilWhitespace(word, buf_line);
            buf_line.ignore();
            readFloatText(tonality, buf_line);

            StringRef ref{string_pool.insert(word.data(), word.size()), word.size()};
            emotional_dict[ref] = tonality;
            ++count;
        }
        LOG_TRACE(log, "Emotional dictionary was added. Word count: {}", std::to_string(count));
    }


    void loadProgrammingFrequency()
    {
        Poco::Logger * log = &Poco::Logger::get("ProgrammingFrequency");

        LOG_TRACE(log, "Loading embedded programming languages frequencies loading");

        auto resource = getResource("prog_freq.txt.zst");
            if (resource.empty())
                throw Exception(ErrorCodes::FILE_DOESNT_EXIST, "There is no embedded programming languages frequencies");

        String line;
        String bigram;
        Float64 frequency;
        String programming_language;

        auto buf = std::make_unique<ReadBufferFromMemory>(resource.data(), resource.size());
        std::unique_ptr<ReadBuffer> in = std::make_unique<ZstdInflatingReadBuffer>(std::move(buf));

        while (!in->eof())
        {
            readString(line, *in);
            ++in->position();

            if (line.empty())
                continue;

            ReadBufferFromString buf_line(line);

            // Start loading a new language
            if (line.starts_with("//"))
            {
                buf_line.ignore(3);
                readString(programming_language, buf_line);

                Language lang;
                lang.name = programming_language;
                programming_freq.push_back(std::move(lang));
            }
            else
            {
                readStringUntilWhitespace(bigram, buf_line);
                buf_line.ignore();
                readFloatText(frequency, buf_line);

                StringRef ref{string_pool.insert(bigram.data(), bigram.size()), bigram.size()};
                programming_freq.back().map[ref] = frequency;
            }
        }
        LOG_TRACE(log, "Programming languages frequencies was added");
    }

    const Map & getEmotionalDict()
    {
        std::lock_guard lock(mutex);
        if (emotional_dict.empty())
            loadEmotionalDict();

        return emotional_dict;
    }


    const EncodingContainer & getEncodingsFrequency()
    {
        std::lock_guard lock(mutex);
        if (encodings_freq.empty())
            loadEncodingsFrequency();

        return encodings_freq;
    }

    const Container & getProgrammingFrequency()
    {
        std::lock_guard lock(mutex);
        if (programming_freq.empty())
            loadProgrammingFrequency();

        return programming_freq;
    }


private:
    Arena string_pool;

    Map emotional_dict;
    Container programming_freq;
    EncodingContainer encodings_freq;

    std::mutex mutex;
};
}
