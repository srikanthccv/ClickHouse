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

#if USE_SSL
void registerDiskEncrypted(DiskFactory & factory);
#endif

#if USE_HDFS
void registerDiskHDFS(DiskFactory & factory);
#endif


void registerDisks()
{
    auto & factory = DiskFactory::instance();

    registerDiskLocal(factory);
    registerDiskMemory(factory);

#if USE_AWS_S3
    registerDiskS3(factory);
#endif

#if USE_SSL
    registerDiskEncrypted(factory);
#endif

#if USE_HDFS
    registerDiskHDFS(factory);
#endif
}

}
