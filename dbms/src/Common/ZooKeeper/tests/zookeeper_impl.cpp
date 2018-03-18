#include <Common/ZooKeeper/ZooKeeperImpl.h>
#include <iostream>


int main()
try
{
    ZooKeeperImpl::ZooKeeper zookeeper({Poco::Net::SocketAddress{"localhost:2181"}}, "", "", "", {30, 0}, {0, 50000});

    zookeeper.create("/test", "hello", false, false, {}, [](const ZooKeeperImpl::ZooKeeper::CreateResponse & response)
    {
        if (response.error)
            std::cerr << "Error " << response.error << ": " << ZooKeeperImpl::ZooKeeper::errorMessage(response.error) << "\n";
        else
            std::cerr << "Path created: " << response.path_created << "\n";
    });

    sleep(100);

    return 0;
}
catch (...)
{
    DB::tryLogCurrentException(__PRETTY_FUNCTION__);
    return 1;
}
