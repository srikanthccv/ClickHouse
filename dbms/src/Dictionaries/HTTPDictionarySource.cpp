#include <Dictionaries/HTTPDictionarySource.h>

#include <Poco/Net/HTTPRequest.h>
#include <Interpreters/Context.h>
#include <DataStreams/OwningBlockInputStream.h>
#include <IO/ReadWriteBufferFromHTTP.h>
#include <DataStreams/IBlockOutputStream.h>
#include <IO/WriteBufferFromOStream.h>
#include <Dictionaries/DictionarySourceHelpers.h>
#include <common/logger_useful.h>
#include <IO/ConnectionTimeouts.h>

namespace DB
{

static const size_t max_block_size = 8192;


HTTPDictionarySource::HTTPDictionarySource(const DictionaryStructure & dict_struct_,
    const Poco::Util::AbstractConfiguration & config, const std::string & config_prefix,
    Block & sample_block, const Context & context)
    : log(&Logger::get("HTTPDictionarySource")),
    update_time{std::chrono::system_clock::now()},
    dict_struct{dict_struct_},
    url{config.getString(config_prefix + ".url", "")},
    update_field{config.getString(config_prefix + ".update_field", "")},
    date{"0000-00-00%2000:00:00"},
    format{config.getString(config_prefix + ".format")},
    sample_block{sample_block},
    context(context),
    timeouts(ConnectionTimeouts::getHTTPTimeouts(context.getSettingsRef()))
{
}

HTTPDictionarySource::HTTPDictionarySource(const HTTPDictionarySource & other)
    : log(&Logger::get("HTTPDictionarySource")),
    update_time{other.update_time},
    dict_struct{other.dict_struct},
    url{other.url},
    update_field{other.update_field},
    date{other.date},
    format{other.format},
    sample_block{other.sample_block},
    context(other.context),
    timeouts(ConnectionTimeouts::getHTTPTimeouts(context.getSettingsRef()))
{
}

void HTTPDictionarySource::setDate()
{
    if (!hasUpdateField())
        return;
    else if ((hasUpdateField() && date == "0000-00-00%2000:00:00"))
    {
        auto tmp_time = update_time;
        update_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(update_time - tmp_time);
        time_t hr_time = std::chrono::system_clock::to_time_t(update_time) - duration.count() - 1;
        char buffer [80];
        struct tm * timeinfo;
        timeinfo = localtime (&hr_time);
        strftime(buffer, 80, "%Y-%m-%d%%20%H:%M:%S", timeinfo);
        std::string str_time(buffer);
        date = str_time;
    }
    else
    {
        auto tmp_time = update_time;
        update_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(update_time - tmp_time);
        time_t hr_time = std::chrono::system_clock::to_time_t(update_time) - duration.count() - 1;
        char buffer [80];
        struct tm * timeinfo;
        timeinfo = localtime (&hr_time);
        strftime(buffer, 80, "%Y-%m-%d%%20%H:%M:%S", timeinfo);
        std::string str_time(buffer);
        date = str_time;
        url_update = url + update_field + date;
    }
}

BlockInputStreamPtr HTTPDictionarySource::loadAll()
{
    LOG_TRACE(log, "loadAll " + toString());
    setDate();

    Poco::URI uri;
    if (!url_update.empty())
        uri = url_update;
    else
        uri = url;
    auto in_ptr = std::make_unique<ReadWriteBufferFromHTTP>(uri, Poco::Net::HTTPRequest::HTTP_GET,
                                                            ReadWriteBufferFromHTTP::OutStreamCallback(), timeouts);
    auto input_stream = context.getInputFormat(format, *in_ptr, sample_block, max_block_size);

    return std::make_shared<OwningBlockInputStream<ReadWriteBufferFromHTTP>>(input_stream, std::move(in_ptr));
}

BlockInputStreamPtr HTTPDictionarySource::loadIds(const std::vector<UInt64> & ids)
{
    LOG_TRACE(log, "loadIds " << toString() << " size = " << ids.size());

    ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = [&](std::ostream & ostr)
    {
        WriteBufferFromOStream out_buffer(ostr);
        auto output_stream = context.getOutputFormat(format, out_buffer, sample_block);
        formatIDs(output_stream, ids);
    };

    Poco::URI uri(url);
    auto in_ptr = std::make_unique<ReadWriteBufferFromHTTP>(uri, Poco::Net::HTTPRequest::HTTP_POST,
                                                            out_stream_callback, timeouts);
    auto input_stream = context.getInputFormat(format, *in_ptr, sample_block, max_block_size);
    return std::make_shared<OwningBlockInputStream<ReadWriteBufferFromHTTP>>(input_stream, std::move(in_ptr));
}

BlockInputStreamPtr HTTPDictionarySource::loadKeys(
    const Columns & key_columns, const std::vector<size_t> & requested_rows)
{
    LOG_TRACE(log, "loadKeys " << toString() << " size = " << requested_rows.size());

    ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = [&](std::ostream & ostr)
    {
        WriteBufferFromOStream out_buffer(ostr);
        auto output_stream = context.getOutputFormat(format, out_buffer, sample_block);
        formatKeys(dict_struct, output_stream, key_columns, requested_rows);
    };

    Poco::URI uri(url);
    auto in_ptr = std::make_unique<ReadWriteBufferFromHTTP>(uri, Poco::Net::HTTPRequest::HTTP_POST,
                                                            out_stream_callback, timeouts);
    auto input_stream = context.getInputFormat(format, *in_ptr, sample_block, max_block_size);
    return std::make_shared<OwningBlockInputStream<ReadWriteBufferFromHTTP>>(input_stream, std::move(in_ptr));
}

bool HTTPDictionarySource::isModified() const
{
    return true;
}

bool HTTPDictionarySource::supportsSelectiveLoad() const
{
    return true;
}

bool HTTPDictionarySource::hasUpdateField() const
{
    if (update_field.empty())
        return false;
    else
        return true;
}

DictionarySourcePtr HTTPDictionarySource::clone() const
{
    return std::make_unique<HTTPDictionarySource>(*this);
}

std::string HTTPDictionarySource::toString() const
{
    Poco::URI uri(url);
    return uri.toString();
}

}
