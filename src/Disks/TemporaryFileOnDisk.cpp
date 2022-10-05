#include <Disks/TemporaryFileOnDisk.h>
#include <Poco/TemporaryFile.h>
#include <Common/CurrentMetrics.h>
#include <Common/logger_useful.h>

#include <filesystem>

namespace ProfileEvents
{
    extern const Event ExternalProcessingFilesTotal;
}

namespace DB
{

TemporaryFileOnDisk::TemporaryFileOnDisk(const DiskPtr & disk_)
    : TemporaryFileOnDisk(disk_, "")
{}

TemporaryFileOnDisk::TemporaryFileOnDisk(const DiskPtr & disk_, CurrentMetrics::Value metric_scope)
    : TemporaryFileOnDisk(disk_)
{
    sub_metric_increment.emplace(metric_scope);
}

TemporaryFileOnDisk::TemporaryFileOnDisk(const DiskPtr & disk_, const String & prefix)
    : disk(disk_)
{
    if (!disk)
        throw Exception("Disk is not specified", ErrorCodes::LOGICAL_ERROR);

    if (fs::path prefix_path(prefix); prefix_path.has_parent_path())
        disk->createDirectories(prefix_path.parent_path());

    ProfileEvents::increment(ProfileEvents::ExternalProcessingFilesTotal);

    /// Do not use default temporaty root path `/tmp/tmpXXXXXX`.
    /// The `dummy_prefix` is used to know what to replace with the real prefix.
    String dummy_prefix = "a/";
    relative_path = Poco::TemporaryFile::tempName(dummy_prefix);
    dummy_prefix += "tmp";
    /// a/tmpXXXXX -> <prefix>XXXXX
    assert(relative_path.starts_with(dummy_prefix));
    relative_path.replace(0, dummy_prefix.length(), prefix);

    if (relative_path.empty())
        throw Exception("Temporary file name is empty", ErrorCodes::LOGICAL_ERROR);
}

String TemporaryFileOnDisk::getPath() const
{
    return std::filesystem::path(disk->getPath()) / relative_path;
}

TemporaryFileOnDisk::~TemporaryFileOnDisk()
{
    try
    {
        if (!disk || relative_path.empty())
            return;

        if (!disk->exists(relative_path))
        {
            LOG_WARNING(&Poco::Logger::get("TemporaryFileOnDisk"), "Temorary path '{}' does not exist in '{}'", relative_path, disk->getPath());
            return;
        }

        disk->removeRecursive(relative_path);
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

}
