#include <Common/FileSyncGuard.h>
#include <Common/Exception.h>
#include <Disks/IDisk.h>
#include <fcntl.h> // O_RDWR

/// OSX does not have O_DIRECTORY
#ifndef O_DIRECTORY
#define O_DIRECTORY O_RDWR
#endif

namespace DB
{

FileSyncGuard::FileSyncGuard(const DiskPtr & disk_, const String & path)
    : disk(disk_)
    , fd(disk_->open(path, O_DIRECTORY))
{}

FileSyncGuard::~FileSyncGuard()
{
    try
    {
#if defined(OS_DARWIN)
        if (fcntl(fd, F_FULLFSYNC, 0))
            throwFromErrno("Cannot fcntl(F_FULLFSYNC)");
#endif
        disk->sync(fd);
        disk->close(fd);
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

}
