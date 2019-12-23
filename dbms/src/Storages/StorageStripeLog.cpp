#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include <map>
#include <optional>

#include <Common/escapeForFileName.h>
#include <Common/Exception.h>

#include <Compression/CompressedReadBuffer.h>
#include <Compression/CompressedReadBufferFromFile.h>
#include <Compression/CompressedWriteBuffer.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>

#include <DataStreams/IBlockInputStream.h>
#include <DataStreams/IBlockOutputStream.h>
#include <DataStreams/NativeBlockInputStream.h>
#include <DataStreams/NativeBlockOutputStream.h>
#include <DataStreams/NullBlockInputStream.h>

#include <DataTypes/DataTypeFactory.h>

#include <Columns/ColumnArray.h>

#include <Interpreters/Context.h>

#include <Storages/StorageStripeLog.h>
#include <Storages/StorageFactory.h>


namespace DB
{

#define INDEX_BUFFER_SIZE 4096

namespace ErrorCodes
{
    extern const int EMPTY_LIST_OF_COLUMNS_PASSED;
    extern const int CANNOT_CREATE_DIRECTORY;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int INCORRECT_FILE_NAME;
    extern const int LOGICAL_ERROR;
}


class StripeLogBlockInputStream final : public IBlockInputStream
{
public:
    StripeLogBlockInputStream(StorageStripeLog & storage_, size_t max_read_buffer_size_,
        std::shared_ptr<const IndexForNativeFormat> & index_,
        IndexForNativeFormat::Blocks::const_iterator index_begin_,
        IndexForNativeFormat::Blocks::const_iterator index_end_)
        : storage(storage_), max_read_buffer_size(max_read_buffer_size_),
        index(index_), index_begin(index_begin_), index_end(index_end_)
    {
        if (index_begin != index_end)
        {
            for (const auto & column : index_begin->columns)
            {
                auto type = DataTypeFactory::instance().get(column.type);
                header.insert(ColumnWithTypeAndName{ type, column.name });
            }
        }
    }

    String getName() const override { return "StripeLog"; }

    Block getHeader() const override
    {
        return header;
    }

protected:
    Block readImpl() override
    {
        Block res;
        start();

        if (block_in)
        {
            res = block_in->read();

            /// Freeing memory before destroying the object.
            if (!res)
            {
                block_in.reset();
                data_in.reset();
                index.reset();
            }
        }

        return res;
    }

private:
    StorageStripeLog & storage;
    size_t max_read_buffer_size;

    std::shared_ptr<const IndexForNativeFormat> index;
    IndexForNativeFormat::Blocks::const_iterator index_begin;
    IndexForNativeFormat::Blocks::const_iterator index_end;
    Block header;

    /** optional - to create objects only on first reading
      *  and delete objects (release buffers) after the source is exhausted
      * - to save RAM when using a large number of sources.
      */
    bool started = false;
    std::optional<CompressedReadBufferFromFile> data_in;
    std::optional<NativeBlockInputStream> block_in;

    void start()
    {
        if (!started)
        {
            started = true;

            String data_file = storage.table_path + "data.bin";
            size_t buffer_size = std::min(max_read_buffer_size, storage.disk->getFileSize(data_file));

            data_in.emplace(fullPath(storage.disk, data_file), 0, 0, buffer_size);
            block_in.emplace(*data_in, 0, index_begin, index_end);
        }
    }
};


class StripeLogBlockOutputStream final : public IBlockOutputStream
{
public:
    explicit StripeLogBlockOutputStream(StorageStripeLog & storage_)
        : storage(storage_), lock(storage.rwlock),
        data_out_file(storage.table_path + "data.bin"),
        data_out_compressed(storage.disk->append(data_out_file)),
        data_out(*data_out_compressed, CompressionCodecFactory::instance().getDefaultCodec(), storage.max_compress_block_size),
        index_out_file(storage.table_path + "index.mrk"),
        index_out_compressed(storage.disk->append(index_out_file)),
        index_out(*index_out_compressed),
        block_out(data_out, 0, storage.getSampleBlock(), false, &index_out, storage.disk->getFileSize(data_out_file))
    {
    }

    ~StripeLogBlockOutputStream() override
    {
        try
        {
            writeSuffix();
        }
        catch (...)
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }

    Block getHeader() const override { return storage.getSampleBlock(); }

    void write(const Block & block) override
    {
        block_out.write(block);
    }

    void writeSuffix() override
    {
        if (done)
            return;

        block_out.writeSuffix();
        data_out.next();
        data_out_compressed->next();
        index_out.next();
        index_out_compressed->next();

        storage.file_checker.update(data_out_file);
        storage.file_checker.update(index_out_file);

        done = true;
    }

private:
    StorageStripeLog & storage;
    std::unique_lock<std::shared_mutex> lock;

    String data_out_file;
    std::unique_ptr<WriteBuffer> data_out_compressed;
    CompressedWriteBuffer data_out;
    String index_out_file;
    std::unique_ptr<WriteBuffer> index_out_compressed;
    CompressedWriteBuffer index_out;
    NativeBlockOutputStream block_out;

    bool done = false;
};


StorageStripeLog::StorageStripeLog(
    DiskPtr disk_,
    const String & database_name_,
    const String & table_name_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    bool attach,
    size_t max_compress_block_size_)
    : disk(disk_), database_name(database_name_), table_name(table_name_),
      table_path("data/" + escapeForFileName(database_name_) + '/' + escapeForFileName(table_name_) + '/'),
    max_compress_block_size(max_compress_block_size_),
    file_checker(disk, table_path + "sizes.json"),
    log(&Logger::get("StorageStripeLog"))
{
    setColumns(columns_);
    setConstraints(constraints_);

    if (!attach)
    {
        /// create directories if they do not exist
        disk->createDirectories(table_path);
    }
}


void StorageStripeLog::rename(const String & /*new_path_to_db*/, const String & new_database_name, const String & new_table_name, TableStructureWriteLockHolder &)
{
    std::unique_lock<std::shared_mutex> lock(rwlock);

    String new_table_path = "data/" + escapeForFileName(new_database_name) + '/' + escapeForFileName(new_table_name) + '/';

    disk->moveDirectory(table_path, new_table_path);

    database_name = new_database_name;
    table_name = new_table_name;
    table_path = new_table_path;
    file_checker.setPath(table_path + "sizes.json");
}


BlockInputStreams StorageStripeLog::read(
    const Names & column_names,
    const SelectQueryInfo & /*query_info*/,
    const Context & context,
    QueryProcessingStage::Enum /*processed_stage*/,
    const size_t /*max_block_size*/,
    unsigned num_streams)
{
    std::shared_lock<std::shared_mutex> lock(rwlock);

    check(column_names);

    NameSet column_names_set(column_names.begin(), column_names.end());

    String index_file = table_path + "index.mrk";
    if (!disk->exists(index_file))
        return { std::make_shared<NullBlockInputStream>(getSampleBlockForColumns(column_names)) };

    CompressedReadBufferFromFile index_in(fullPath(disk, index_file), 0, 0, INDEX_BUFFER_SIZE);

    std::shared_ptr<const IndexForNativeFormat> index{std::make_shared<IndexForNativeFormat>(index_in, column_names_set)};

    BlockInputStreams res;

    size_t size = index->blocks.size();
    if (num_streams > size)
        num_streams = size;

    for (size_t stream = 0; stream < num_streams; ++stream)
    {
        IndexForNativeFormat::Blocks::const_iterator begin = index->blocks.begin();
        IndexForNativeFormat::Blocks::const_iterator end = index->blocks.begin();

        std::advance(begin, stream * size / num_streams);
        std::advance(end, (stream + 1) * size / num_streams);

        res.emplace_back(std::make_shared<StripeLogBlockInputStream>(
            *this, context.getSettingsRef().max_read_buffer_size, index, begin, end));
    }

    /// We do not keep read lock directly at the time of reading, because we read ranges of data that do not change.

    return res;
}


BlockOutputStreamPtr StorageStripeLog::write(
    const ASTPtr & /*query*/, const Context & /*context*/)
{
    return std::make_shared<StripeLogBlockOutputStream>(*this);
}


CheckResults StorageStripeLog::checkData(const ASTPtr & /* query */, const Context & /* context */)
{
    std::shared_lock<std::shared_mutex> lock(rwlock);
    return file_checker.check();
}

void StorageStripeLog::truncate(const ASTPtr &, const Context &, TableStructureWriteLockHolder &)
{
    if (table_name.empty())
        throw Exception("Logical error: table name is empty", ErrorCodes::LOGICAL_ERROR);

    std::shared_lock<std::shared_mutex> lock(rwlock);

    disk->clearDirectory(table_path);

    file_checker = FileChecker{disk, table_path + "sizes.json"};
}


void registerStorageStripeLog(StorageFactory & factory)
{
    factory.registerStorage("StripeLog", [](const StorageFactory::Arguments & args)
    {
        if (!args.engine_args.empty())
            throw Exception(
                "Engine " + args.engine_name + " doesn't support any arguments (" + toString(args.engine_args.size()) + " given)",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        return StorageStripeLog::create(
            args.context.getDefaultDisk(), args.database_name, args.table_name, args.columns, args.constraints,
            args.attach, args.context.getSettings().max_compress_block_size);
    });
}

}
