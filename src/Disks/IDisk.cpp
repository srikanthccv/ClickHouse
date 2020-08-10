#include "IDisk.h"
#include "Disks/Executor.h"
#include <IO/ReadBufferFromFileBase.h>
#include <IO/WriteBufferFromFileBase.h>
#include <IO/copyData.h>
#include <Poco/Logger.h>
#include <common/logger_useful.h>
#include <Common/setThreadName.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
}

bool IDisk::isDirectoryEmpty(const String & path)
{
    return !iterateDirectory(path)->isValid();
}

void copyFile(IDisk & from_disk, const String & from_path, IDisk & to_disk, const String & to_path)
{
    LOG_DEBUG(&Poco::Logger::get("IDisk"), "Copying from {} {} to {} {}.", from_disk.getName(), from_path, to_disk.getName(), to_path);

    auto in = from_disk.readFile(from_path);
    auto out = to_disk.writeFile(to_path);
    copyData(*in, *out);
}


using ResultsCollector = std::vector<std::future<void>>;

void asyncCopy(IDisk & from_disk, String from_path, IDisk & to_disk, String to_path, Executor & exec, ResultsCollector & results)
{
    if (from_disk.isFile(from_path))
    {
        auto result = exec.execute(
            [&from_disk, from_path, &to_disk, to_path]()
            {
                setThreadName("DiskCopier");
                DB::copyFile(from_disk, from_path, to_disk, to_path + fileName(from_path));
            });

        results.push_back(std::move(result));
    }
    else
    {
        Poco::Path path(from_path);
        const String & dir_name = path.directory(path.depth() - 1);
        const String dest = to_path + dir_name + "/";
        to_disk.createDirectories(dest);

        for (auto it = from_disk.iterateDirectory(from_path); it->isValid(); it->next())
            asyncCopy(from_disk, it->path(), to_disk, dest, exec, results);
    }
}

void IDisk::copy(const String & from_path, const std::shared_ptr<IDisk> & to_disk, const String & to_path)
{
    auto exec = to_disk->getExecutor();
    ResultsCollector results;

    asyncCopy(*this, from_path, *to_disk, to_path, *exec, results);

    for (auto & result : results)
        result.wait();
    for (auto & result : results)
        result.get();
}

/// Executes task synchronously in case when disk doesn't support async operations.
class SyncExecutor : public Executor
{
public:
    SyncExecutor() = default;
    std::future<void> execute(std::function<void()> task) override
    {
        auto promise = std::make_shared<std::promise<void>>();
        try
        {
            task();
            promise->set_value();
        }
        catch (...)
        {
            try
            {
                promise->set_exception(std::current_exception());
            }
            catch (...) { }
        }
        return promise->get_future();
    }
};

std::unique_ptr<Executor> IDisk::getExecutor()
{
    return std::make_unique<SyncExecutor>();
}

void IDisk::truncateFile(const String &, size_t)
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Truncate operation is not implemented for disk of type {}", getType());
}

}
