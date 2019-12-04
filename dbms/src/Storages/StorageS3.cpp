#include <Storages/StorageFactory.h>
#include <Storages/StorageS3.h>

#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTLiteral.h>

#include <IO/ReadBufferFromS3.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteBufferFromS3.h>
#include <IO/WriteHelpers.h>

#include <Formats/FormatFactory.h>

#include <DataStreams/IBlockOutputStream.h>
#include <DataStreams/IBlockInputStream.h>
#include <DataStreams/AddingDefaultsBlockInputStream.h>

#include <Poco/Net/HTTPRequest.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace
{
    class StorageS3BlockInputStream : public IBlockInputStream
    {
    public:
        StorageS3BlockInputStream(const Poco::URI & uri,
            const String & format,
            const String & name_,
            const Block & sample_block,
            const Context & context,
            UInt64 max_block_size,
            const ConnectionTimeouts & timeouts,
            const CompressionMethod compression_method)
            : name(name_)
        {
            read_buf = getReadBuffer<ReadBufferFromS3>(compression_method, uri, timeouts);
            reader = FormatFactory::instance().getInput(format, *read_buf, sample_block, context, max_block_size);
        }

        String getName() const override
        {
            return name;
        }

        Block readImpl() override
        {
            return reader->read();
        }

        Block getHeader() const override
        {
            return reader->getHeader();
        }

        void readPrefixImpl() override
        {
            reader->readPrefix();
        }

        void readSuffixImpl() override
        {
            reader->readSuffix();
        }

    private:
        String name;
        std::unique_ptr<ReadBuffer> read_buf;
        BlockInputStreamPtr reader;
    };

    class StorageS3BlockOutputStream : public IBlockOutputStream
    {
    public:
        StorageS3BlockOutputStream(const Poco::URI & uri,
            const String & format,
            UInt64 min_upload_part_size,
            const Block & sample_block_,
            const Context & context,
            const ConnectionTimeouts & timeouts,
            const CompressionMethod compression_method)
            : sample_block(sample_block_)
        {
            write_buf = getWriteBuffer<WriteBufferFromS3>(compression_method, uri, min_upload_part_size, timeouts);
            writer = FormatFactory::instance().getOutput(format, *write_buf, sample_block, context);
        }

        Block getHeader() const override
        {
            return sample_block;
        }

        void write(const Block & block) override
        {
            writer->write(block);
        }

        void writePrefix() override
        {
            writer->writePrefix();
        }

        void writeSuffix() override
        {
            writer->writeSuffix();
            writer->flush();
            write_buf->finalize();
        }

    private:
        Block sample_block;
        std::unique_ptr<WriteBuffer> write_buf;
        BlockOutputStreamPtr writer;
    };
}


StorageS3::StorageS3(
    const Poco::URI & uri_,
    const StorageID & table_id_,
    const String & format_name_,
    UInt64 min_upload_part_size_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    Context & context_,
    const String & compression_method_ = "")
    : IStorage({table_id_}, columns_)
    , uri(uri_)
    , context_global(context_)
    , format_name(format_name_)
    , min_upload_part_size(min_upload_part_size_)
    , compression_method(compression_method_)
{
    setColumns(columns_);
    setConstraints(constraints_);
}


BlockInputStreams StorageS3::read(
    const Names & column_names,
    const SelectQueryInfo & /*query_info*/,
    const Context & context,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    unsigned /*num_streams*/)
{
    BlockInputStreamPtr block_input = std::make_shared<StorageS3BlockInputStream>(
        uri,
        format_name,
        getName(),
        getHeaderBlock(column_names),
        context,
        max_block_size,
        ConnectionTimeouts::getHTTPTimeouts(context),
        IStorage::chooseCompressionMethod(uri.toString(), compression_method));

    auto column_defaults = getColumns().getDefaults();
    if (column_defaults.empty())
        return {block_input};
    return {std::make_shared<AddingDefaultsBlockInputStream>(block_input, column_defaults, context)};
}

BlockOutputStreamPtr StorageS3::write(const ASTPtr & /*query*/, const Context & /*context*/)
{
    return std::make_shared<StorageS3BlockOutputStream>(
        uri, format_name, min_upload_part_size, getSampleBlock(), context_global,
        ConnectionTimeouts::getHTTPTimeouts(context_global),
        IStorage::chooseCompressionMethod(uri.toString(), compression_method));
}

void registerStorageS3(StorageFactory & factory)
{
    factory.registerStorage("S3", [](const StorageFactory::Arguments & args)
    {
        ASTs & engine_args = args.engine_args;

        if (engine_args.size() != 2 && engine_args.size() != 3)
            throw Exception(
                "Storage S3 requires 2 or 3 arguments: url, name of used format and compression_method.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        engine_args[0] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args[0], args.local_context);

        String url = engine_args[0]->as<ASTLiteral &>().value.safeGet<String>();
        Poco::URI uri(url);

        engine_args[1] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args[1], args.local_context);

        String format_name = engine_args[1]->as<ASTLiteral &>().value.safeGet<String>();

        UInt64 min_upload_part_size = args.local_context.getSettingsRef().s3_min_upload_part_size;

        String compression_method;
        if (engine_args.size() == 3)
        {
            engine_args[2] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args[2], args.local_context);
            compression_method = engine_args[2]->as<ASTLiteral &>().value.safeGet<String>();
        } else compression_method = "auto";

        return StorageS3::create(uri, args.table_id, format_name, min_upload_part_size, args.columns, args.constraints, args.context);
    });
}
}
