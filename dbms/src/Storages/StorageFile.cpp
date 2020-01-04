#include <Storages/StorageFile.h>
#include <Storages/StorageFactory.h>

#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>

#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTIdentifier.h>

#include <IO/ReadBufferFromFile.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteBufferFromFile.h>
#include <IO/WriteHelpers.h>

#include <Formats/FormatFactory.h>
#include <DataStreams/IBlockInputStream.h>
#include <DataStreams/IBlockOutputStream.h>
#include <DataStreams/AddingDefaultsBlockInputStream.h>
#include <DataStreams/narrowBlockInputStreams.h>

#include <Common/escapeForFileName.h>
#include <Common/typeid_cast.h>
#include <Common/parseGlobs.h>

#include <fcntl.h>

#include <Poco/Path.h>
#include <Poco/File.h>

#include <re2/re2.h>
#include <filesystem>
#include <Storages/Distributed/DirectoryMonitor.h>

namespace fs = std::filesystem;

namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_WRITE_TO_FILE_DESCRIPTOR;
    extern const int CANNOT_SEEK_THROUGH_FILE;
    extern const int DATABASE_ACCESS_DENIED;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int UNKNOWN_IDENTIFIER;
    extern const int INCORRECT_FILE_NAME;
    extern const int FILE_DOESNT_EXIST;
    extern const int EMPTY_LIST_OF_COLUMNS_PASSED;
}

namespace
{

/* Recursive directory listing with matched paths as a result.
 * Have the same method in StorageHDFS.
 */
static std::vector<std::string> listFilesWithRegexpMatching(const std::string & path_for_ls, const std::string & for_match)
{
    const size_t first_glob = for_match.find_first_of("*?{");

    const size_t end_of_path_without_globs = for_match.substr(0, first_glob).rfind('/');
    const std::string suffix_with_globs = for_match.substr(end_of_path_without_globs);   /// begin with '/'

    const size_t next_slash = suffix_with_globs.find('/', 1);
    re2::RE2 matcher(makeRegexpPatternFromGlobs(suffix_with_globs.substr(0, next_slash)));

    std::vector<std::string> result;
    const std::string prefix_without_globs = path_for_ls + for_match.substr(1, end_of_path_without_globs);
    if (!fs::exists(fs::path(prefix_without_globs.data())))
    {
        return result;
    }
    const fs::directory_iterator end;
    for (fs::directory_iterator it(prefix_without_globs); it != end; ++it)
    {
        const std::string full_path = it->path().string();
        const size_t last_slash = full_path.rfind('/');
        const String file_name = full_path.substr(last_slash);
        const bool looking_for_directory = next_slash != std::string::npos;
        /// Condition is_directory means what kind of path is it in current iteration of ls
        if (!fs::is_directory(it->path()) && !looking_for_directory)
        {
            if (re2::RE2::FullMatch(file_name, matcher))
            {
                result.push_back(it->path().string());
            }
        }
        else if (fs::is_directory(it->path()) && looking_for_directory)
        {
            if (re2::RE2::FullMatch(file_name, matcher))
            {
                /// Recursion depth is limited by pattern. '*' works only for depth = 1, for depth = 2 pattern path is '*/*'. So we do not need additional check.
                Strings result_part = listFilesWithRegexpMatching(full_path + "/", suffix_with_globs.substr(next_slash));
                std::move(result_part.begin(), result_part.end(), std::back_inserter(result));
            }
        }
    }
    return result;
}

static std::string getTablePath(const std::string & table_dir_path, const std::string & format_name)
{
    return table_dir_path + "/data." + escapeForFileName(format_name);
}

/// Both db_dir_path and table_path must be converted to absolute paths (in particular, path cannot contain '..').
static void checkCreationIsAllowed(const Context & context_global, const std::string & db_dir_path, const std::string & table_path)
{
    if (context_global.getApplicationType() != Context::ApplicationType::SERVER)
        return;

    /// "/dev/null" is allowed for perf testing
    if (!startsWith(table_path, db_dir_path) && table_path != "/dev/null")
        throw Exception("Part path " + table_path + " is not inside " + db_dir_path, ErrorCodes::DATABASE_ACCESS_DENIED);

    Poco::File table_path_poco_file = Poco::File(table_path);
    if (table_path_poco_file.exists() && table_path_poco_file.isDirectory())
        throw Exception("File " + table_path + " must not be a directory", ErrorCodes::INCORRECT_FILE_NAME);
}
}


StorageFile::StorageFile(int table_fd_, CommonArguments args)
    : StorageFile(args)
{
    if (args.context.getApplicationType() == Context::ApplicationType::SERVER)
        throw Exception("Using file descriptor as source of storage isn't allowed for server daemons", ErrorCodes::DATABASE_ACCESS_DENIED);

    is_db_table = false;
    use_table_fd = true;
    table_fd = table_fd_;

    /// Save initial offset, it will be used for repeating SELECTs
    /// If FD isn't seekable (lseek returns -1), then the second and subsequent SELECTs will fail.
    table_fd_init_offset = lseek(table_fd, 0, SEEK_CUR);
}

StorageFile::StorageFile(const std::string & table_path_, const std::string & user_files_path, CommonArguments args)
    : StorageFile(args)
{
    is_db_table = false;
    std::string user_files_absolute_path = Poco::Path(user_files_path).makeAbsolute().makeDirectory().toString();
    Poco::Path poco_path = Poco::Path(table_path_);
    if (poco_path.isRelative())
        poco_path = Poco::Path(user_files_absolute_path, poco_path);

    const std::string path = poco_path.absolute().toString();
    if (path.find_first_of("*?{") == std::string::npos)
    {
        paths.push_back(path);
    }
    else
        paths = listFilesWithRegexpMatching("/", path);
    for (const auto & cur_path : paths)
        checkCreationIsAllowed(args.context, user_files_absolute_path, cur_path);

    if (args.format_name == "Distributed")
    {
        if (!paths.empty())
        {
            auto & first_path = paths[0];
            Block header = StorageDistributedDirectoryMonitor::createStreamFromFile(first_path)->getHeader();

            setColumns(ColumnsDescription(header.getNamesAndTypesList()));
        }
    }
}

StorageFile::StorageFile(const std::string & relative_table_dir_path, CommonArguments args)
    : StorageFile(args)
{
    if (relative_table_dir_path.empty())
        throw Exception("Storage " + getName() + " requires data path", ErrorCodes::INCORRECT_FILE_NAME);

    String table_dir_path = base_path + relative_table_dir_path + "/";
    Poco::File(table_dir_path).createDirectories();
    paths = {getTablePath(table_dir_path, format_name)};
}

StorageFile::StorageFile(CommonArguments args)
    : table_name(args.table_name), database_name(args.database_name), format_name(args.format_name)
    , compression_method(args.compression_method), base_path(args.context.getPath())
{
    setColumns(args.columns);
    setConstraints(args.constraints);
}

class StorageFileBlockInputStream : public IBlockInputStream
{
public:
    StorageFileBlockInputStream(std::shared_ptr<StorageFile> storage_,
        const Context & context, UInt64 max_block_size,
        std::string file_path,
        const CompressionMethod compression_method,
        BlockInputStreamPtr prepared_reader = nullptr)
        : storage(std::move(storage_)), reader(std::move(prepared_reader))
    {
        if (storage->use_table_fd)
        {
            unique_lock = std::unique_lock(storage->rwlock);

            /// We could use common ReadBuffer and WriteBuffer in storage to leverage cache
            ///  and add ability to seek unseekable files, but cache sync isn't supported.

            if (storage->table_fd_was_used) /// We need seek to initial position
            {
                if (storage->table_fd_init_offset < 0)
                    throw Exception("File descriptor isn't seekable, inside " + storage->getName(), ErrorCodes::CANNOT_SEEK_THROUGH_FILE);

                /// ReadBuffer's seek() doesn't make sense, since cache is empty
                if (lseek(storage->table_fd, storage->table_fd_init_offset, SEEK_SET) < 0)
                    throwFromErrno("Cannot seek file descriptor, inside " + storage->getName(), ErrorCodes::CANNOT_SEEK_THROUGH_FILE);
            }

            storage->table_fd_was_used = true;
            read_buf = getReadBuffer<ReadBufferFromFileDescriptor>(compression_method, storage->table_fd);
        }
        else
        {
            shared_lock = std::shared_lock(storage->rwlock);
            read_buf = getReadBuffer<ReadBufferFromFile>(compression_method, file_path);
        }

        if (!reader)
            reader = FormatFactory::instance().getInput(storage->format_name, *read_buf, storage->getSampleBlock(), context, max_block_size);
    }

    String getName() const override
    {
        return storage->getName();
    }

    Block readImpl() override
    {
        return reader->read();
    }

    Block getHeader() const override { return reader->getHeader(); }

    void readPrefixImpl() override
    {
        reader->readPrefix();
    }

    void readSuffixImpl() override
    {
        reader->readSuffix();
    }

private:
    std::shared_ptr<StorageFile> storage;
    Block sample_block;
    std::unique_ptr<ReadBuffer> read_buf;
    BlockInputStreamPtr reader;

    std::shared_lock<std::shared_mutex> shared_lock;
    std::unique_lock<std::shared_mutex> unique_lock;
};


BlockInputStreams StorageFile::read(
    const Names & /*column_names*/,
    const SelectQueryInfo & /*query_info*/,
    const Context & context,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    unsigned num_streams)
{
    const ColumnsDescription & columns_ = getColumns();
    auto column_defaults = columns_.getDefaults();
    BlockInputStreams blocks_input;
    if (use_table_fd)   /// need to call ctr BlockInputStream
        paths = {""};   /// when use fd, paths are empty
    else
    {
        if (paths.size() == 1 && !Poco::File(paths[0]).exists())
            throw Exception("File " + paths[0] + " doesn't exist", ErrorCodes::FILE_DOESNT_EXIST);
    }
    blocks_input.reserve(paths.size());
    for (const auto & file_path : paths)
    {
        BlockInputStreamPtr cur_block;

        if (format_name == "Distributed")
            cur_block = StorageDistributedDirectoryMonitor::createStreamFromFile(file_path);
        else
            cur_block = std::make_shared<StorageFileBlockInputStream>(
                std::static_pointer_cast<StorageFile>(shared_from_this()), context, max_block_size, file_path, IStorage::chooseCompressionMethod(file_path, compression_method));

        blocks_input.push_back(column_defaults.empty() ? cur_block : std::make_shared<AddingDefaultsBlockInputStream>(cur_block, column_defaults, context));
    }
    return narrowBlockInputStreams(blocks_input, num_streams);
}


class StorageFileBlockOutputStream : public IBlockOutputStream
{
public:
    explicit StorageFileBlockOutputStream(StorageFile & storage_,
        const CompressionMethod compression_method,
        const Context & context)
        : storage(storage_), lock(storage.rwlock)
    {
        if (storage.use_table_fd)
        {
            /** NOTE: Using real file binded to FD may be misleading:
              * SELECT *; INSERT insert_data; SELECT *; last SELECT returns initil_fd_data + insert_data
              * INSERT data; SELECT *; last SELECT returns only insert_data
              */
            storage.table_fd_was_used = true;
            write_buf = getWriteBuffer<WriteBufferFromFileDescriptor>(compression_method, storage.table_fd);
        }
        else
        {
            if (storage.paths.size() != 1)
                throw Exception("Table '" + storage.table_name + "' is in readonly mode because of globs in filepath", ErrorCodes::DATABASE_ACCESS_DENIED);
            write_buf = getWriteBuffer<WriteBufferFromFile>(compression_method, storage.paths[0], DBMS_DEFAULT_BUFFER_SIZE, O_WRONLY | O_APPEND | O_CREAT);
        }

        writer = FormatFactory::instance().getOutput(storage.format_name, *write_buf, storage.getSampleBlock(), context);
    }

    Block getHeader() const override { return storage.getSampleBlock(); }

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
    }

    void flush() override
    {
        writer->flush();
    }

private:
    StorageFile & storage;
    std::unique_lock<std::shared_mutex> lock;
    std::unique_ptr<WriteBuffer> write_buf;
    BlockOutputStreamPtr writer;
};

BlockOutputStreamPtr StorageFile::write(
    const ASTPtr & /*query*/,
    const Context & context)
{
    if (format_name == "Distributed")
        throw Exception("Method write is not implemented for Distributed format", ErrorCodes::NOT_IMPLEMENTED);

    return std::make_shared<StorageFileBlockOutputStream>(*this,
        IStorage::chooseCompressionMethod(paths[0], compression_method), context);
}

Strings StorageFile::getDataPaths() const
{
    if (paths.empty())
        throw Exception("Table '" + table_name + "' is in readonly mode", ErrorCodes::DATABASE_ACCESS_DENIED);
    return paths;
}

void StorageFile::rename(const String & new_path_to_table_data, const String & new_database_name, const String & new_table_name, TableStructureWriteLockHolder &)
{
    if (!is_db_table)
        throw Exception("Can't rename table '" + table_name + "' binded to user-defined file (or FD)", ErrorCodes::DATABASE_ACCESS_DENIED);

    if (paths.size() != 1)
        throw Exception("Can't rename table '" + table_name + "' in readonly mode", ErrorCodes::DATABASE_ACCESS_DENIED);

    std::unique_lock<std::shared_mutex> lock(rwlock);

    std::string path_new = getTablePath(base_path + new_path_to_table_data, format_name);
    Poco::File(Poco::Path(path_new).parent()).createDirectories();
    Poco::File(paths[0]).renameTo(path_new);

    paths[0] = std::move(path_new);
    table_name = new_table_name;
    database_name = new_database_name;
}


void registerStorageFile(StorageFactory & factory)
{
    factory.registerStorage("File", [](const StorageFactory::Arguments & args)
    {
        ASTs & engine_args = args.engine_args;

        if (!(engine_args.size() >= 1 && engine_args.size() <= 3))
            throw Exception(
                "Storage File requires from 1 to 3 arguments: name of used format, source and compression_method.",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        engine_args[0] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args[0], args.local_context);
        String format_name = engine_args[0]->as<ASTLiteral &>().value.safeGet<String>();

        String compression_method;
        StorageFile::CommonArguments common_args{args.database_name, args.table_name, format_name, compression_method,
                                                 args.columns, args.constraints, args.context};

        if (engine_args.size() == 1)    /// Table in database
            return StorageFile::create(args.relative_data_path, common_args);

        /// Will use FD if engine_args[1] is int literal or identifier with std* name
        int source_fd = -1;
        String source_path;

        if (auto opt_name = tryGetIdentifierName(engine_args[1]))
        {
            if (*opt_name == "stdin")
                source_fd = STDIN_FILENO;
            else if (*opt_name == "stdout")
                source_fd = STDOUT_FILENO;
            else if (*opt_name == "stderr")
                source_fd = STDERR_FILENO;
            else
                throw Exception("Unknown identifier '" + *opt_name + "' in second arg of File storage constructor",
                                ErrorCodes::UNKNOWN_IDENTIFIER);
        }
        else if (const auto * literal = engine_args[1]->as<ASTLiteral>())
        {
            auto type = literal->value.getType();
            if (type == Field::Types::Int64)
                source_fd = static_cast<int>(literal->value.get<Int64>());
            else if (type == Field::Types::UInt64)
                source_fd = static_cast<int>(literal->value.get<UInt64>());
            else if (type == Field::Types::String)
                source_path = literal->value.get<String>();
            else
                throw Exception("Second argument must be path or file descriptor", ErrorCodes::BAD_ARGUMENTS);
        }

        if (engine_args.size() == 3)
        {
            engine_args[2] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args[2], args.local_context);
            compression_method = engine_args[2]->as<ASTLiteral &>().value.safeGet<String>();
        }
        else
            compression_method = "auto";

        if (0 <= source_fd)     /// File descriptor
            return StorageFile::create(source_fd, common_args);
        else                    /// User's file
            return StorageFile::create(source_path, args.context.getUserFilesPath(), common_args);
    });
}

}
