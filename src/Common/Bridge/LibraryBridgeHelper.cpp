#include "LibraryBridgeHelper.h"

#include <sstream>
#include <IO/ReadHelpers.h>
#include <DataStreams/OneBlockInputStream.h>
#include <DataStreams/OwningBlockInputStream.h>
#include <Dictionaries/DictionarySourceHelpers.h>
#include <Processors/Formats/InputStreamFromInputFormat.h>
#include <IO/WriteBufferFromOStream.h>
#include <Formats/FormatFactory.h>
#include <Poco/Path.h>
#include <Poco/Util/AbstractConfiguration.h>
#include <Common/ShellCommand.h>
#include <common/logger_useful.h>
#include <ext/range.h>


namespace DB
{

LibraryBridgeHelper::LibraryBridgeHelper(
        const Context & context_,
        const std::string & dictionary_id_)
    : log(&Poco::Logger::get("LibraryBridgeHelper"))
    , context(context_)
    , config(context.getConfigRef())
    , http_timeout(context.getSettingsRef().http_receive_timeout.value.totalSeconds())
    , dictionary_id(dictionary_id_)
{
    bridge_port = config.getUInt("library_bridge.port", DEFAULT_PORT);
    bridge_host = config.getString("library_bridge.host", DEFAULT_HOST);
}


Poco::URI LibraryBridgeHelper::getDictionaryURI() const
{
    auto uri = getMainURI();
    uri.addQueryParameter("dictionary_id", dictionary_id);
    return uri;
}


Poco::URI LibraryBridgeHelper::createBaseURI() const
{
    Poco::URI uri;
    uri.setHost(bridge_host);
    uri.setPort(bridge_port);
    uri.setScheme("http");
    return uri;
}


void LibraryBridgeHelper::startBridge(std::unique_ptr<ShellCommand> cmd) const
{
    cmd->wait();
}


bool LibraryBridgeHelper::initLibrary(const std::string & library_path, const std::string library_settings)
{
    startBridgeSync();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", LIB_NEW_METHOD);
    uri.addQueryParameter("library_path", library_path);
    uri.addQueryParameter("library_settings", library_settings);

    return executeRequest(uri);
}


bool LibraryBridgeHelper::cloneLibrary(const std::string & other_dictionary_id)
{
    startBridgeSync();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", LIB_CLONE_METHOD);
    uri.addQueryParameter("from_dictionary_id", other_dictionary_id);

    return executeRequest(uri);
}


bool LibraryBridgeHelper::removeLibrary()
{
    startBridgeSync();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", LIB_DELETE_METHOD);

    return executeRequest(uri);
}


bool LibraryBridgeHelper::isModified()
{
    startBridgeSync();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", IS_MODIFIED_METHOD);

    return executeRequest(uri);
}


bool LibraryBridgeHelper::supportsSelectiveLoad()
{
    startBridgeSync();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", SUPPORTS_SELECTIVE_LOAD_METHOD);

    return executeRequest(uri);
}


BlockInputStreamPtr LibraryBridgeHelper::loadAll(const std::string attributes_string, const Block & sample_block)
{
    startBridgeSync();

    auto uri = getDictionaryURI();

    uri.addQueryParameter("method", LOAD_ALL_METHOD);
    uri.addQueryParameter("attributes", attributes_string);
    uri.addQueryParameter("sample_block", sample_block.getNamesAndTypesList().toString());

    return loadBase(uri, sample_block);
}


BlockInputStreamPtr LibraryBridgeHelper::loadIds(const std::string attributes_string, const std::string ids_string, const Block & sample_block)
{
    startBridgeSync();

    auto uri = getDictionaryURI();

    uri.addQueryParameter("method", LOAD_IDS_METHOD);
    uri.addQueryParameter("attributes", attributes_string);
    uri.addQueryParameter("ids", ids_string);
    uri.addQueryParameter("sample_block", sample_block.getNamesAndTypesList().toString());

    return loadBase(uri, sample_block);
}


BlockInputStreamPtr LibraryBridgeHelper::loadKeys(const Block & key_columns, const Block & sample_block)
{
    startBridgeSync();

    auto columns = key_columns.getColumns();
    auto keys_sample_block = key_columns.cloneEmpty();

    auto uri = getDictionaryURI();
    uri.addQueryParameter("method", LOAD_KEYS_METHOD);
    uri.addQueryParameter("sample_block", sample_block.getNamesAndTypesList().toString());
    uri.addQueryParameter("requested_block", keys_sample_block.getNamesAndTypesList().toString());

    ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = [key_columns, sample_block, this](std::ostream & ostr)
    {
        WriteBufferFromOStream out_buffer(ostr);
        auto output_stream = context.getOutputStream(
                LibraryBridgeHelper::DEFAULT_FORMAT, out_buffer, sample_block);
        formatBlock(output_stream, key_columns);
    };

    return loadBase(uri, sample_block, out_stream_callback);
}


bool LibraryBridgeHelper::executeRequest(const Poco::URI & uri)
{
    ReadWriteBufferFromHTTP buf(
        uri,
        Poco::Net::HTTPRequest::HTTP_POST,
        {},
        ConnectionTimeouts::getHTTPTimeouts(context));

    bool res;
    readBoolText(res, buf);
    return res;
}


BlockInputStreamPtr LibraryBridgeHelper::loadBase(const Poco::URI & uri, const Block & sample_block, ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback)
{
    auto read_buf_ptr = std::make_unique<ReadWriteBufferFromHTTP>(
        uri,
        Poco::Net::HTTPRequest::HTTP_POST,
        std::move(out_stream_callback),
        ConnectionTimeouts::getHTTPTimeouts(context),
        0,
        Poco::Net::HTTPBasicCredentials{},
        DBMS_DEFAULT_BUFFER_SIZE,
        ReadWriteBufferFromHTTP::HTTPHeaderEntries{});

    auto input_stream = context.getInputFormat(LibraryBridgeHelper::DEFAULT_FORMAT, *read_buf_ptr, sample_block, DEFAULT_BLOCK_SIZE);
    return std::make_shared<OwningBlockInputStream<ReadWriteBufferFromHTTP>>(input_stream, std::move(read_buf_ptr));
}

}
