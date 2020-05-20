#include "registerDisks.h"

#include "DiskFactory.h"

#if !defined(ARCADIA_BUILD)
#    include <Common/config.h>
#endif

namespace DB
{
void registerDiskLocal(DiskFactory & factory);
void registerDiskMemory(DiskFactory & factory);
#if USE_AWS_S3
void registerDiskS3(DiskFactory & factory);
#endif
void registerDiskHDFS(DiskFactory & factory);


void registerDisks()
{
    auto & factory = DiskFactory::instance();

    registerDiskLocal(factory);
    registerDiskMemory(factory);
    registerDiskHDFS(factory);

#if USE_AWS_S3
    registerDiskS3(factory);
#endif
}

}
