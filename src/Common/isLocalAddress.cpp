#include <Common/isLocalAddress.h>

#include <ifaddrs.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <base/types.h>
#include <boost/core/noncopyable.hpp>
#include <Common/Exception.h>
#include <Poco/Net/IPAddress.h>
#include <Poco/Net/SocketAddress.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int SYSTEM_ERROR;
}

namespace
{

struct NetworkInterfaces : public boost::noncopyable
{
    ifaddrs * ifaddr;
    NetworkInterfaces()
    {
        if (getifaddrs(&ifaddr) == -1)
        {
            throwFromErrno("Cannot getifaddrs", ErrorCodes::SYSTEM_ERROR);
        }
    }

    void swap(NetworkInterfaces && other)
    {
        auto * tmp = ifaddr;
        ifaddr = other.ifaddr;
        other.ifaddr = tmp;
    }

    bool hasAddress(const Poco::Net::IPAddress & address) const
    {
        ifaddrs * iface;
        for (iface = ifaddr; iface != nullptr; iface = iface->ifa_next)
        {
            /// Point-to-point (VPN) addresses may have NULL ifa_addr
            if (!iface->ifa_addr)
                continue;

            auto family = iface->ifa_addr->sa_family;
            std::optional<Poco::Net::IPAddress> interface_address;
            switch (family)
            {
                /// We interested only in IP-addresses
                case AF_INET:
                {
                    interface_address.emplace(*(iface->ifa_addr));
                    break;
                }
                case AF_INET6:
                {
                    interface_address.emplace(&reinterpret_cast<const struct sockaddr_in6*>(iface->ifa_addr)->sin6_addr, sizeof(struct in6_addr));
                    break;
                }
                default:
                    continue;
            }

            /** Compare the addresses without taking into account `scope`.
              * Theoretically, this may not be correct - depends on `route` setting
              *  - through which interface we will actually access the specified address.
              */
            if (interface_address->length() == address.length()
                && 0 == memcmp(interface_address->addr(), address.addr(), address.length()))
                return true;
        }
        return false;
    }

    ~NetworkInterfaces()
    {
        freeifaddrs(ifaddr);
    }

    static const NetworkInterfaces & instance()
    {
        static constexpr int NET_INTERFACE_VALID_PERIOD_MS = 30000;
        static NetworkInterfaces nf;
        static std::atomic<std::chrono::steady_clock::time_point> last_updated_time = std::chrono::steady_clock::now();
        static std::shared_mutex nf_mtx;

        auto now = std::chrono::steady_clock::now();
        auto last_updated_time_snapshot = last_updated_time.load();

        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_updated_time_snapshot).count() > NET_INTERFACE_VALID_PERIOD_MS)
        {
            std::unique_lock lock(nf_mtx);
            if (last_updated_time.load() != last_updated_time_snapshot) /// it's possible that last_updated_time after we get the snapshot
                return nf;
            nf.swap(NetworkInterfaces());
            last_updated_time.store(now);
            return nf;
        }
        else
        {
            std::shared_lock lock(nf_mtx);
            return nf;
        }
    }
};

}


bool isLocalAddress(const Poco::Net::IPAddress & address)
{
    /** 127.0.0.1 is treat as local address unconditionally.
      * ::1 is also treat as local address unconditionally.
      *
      * 127.0.0.{2..255} are not treat as local addresses, because they are used in tests
      *  to emulate distributed queries across localhost.
      *
      * But 127.{0,1}.{0,1}.{0,1} are treat as local addresses,
      *  because they are used in Debian for localhost.
      */
    if (address.isLoopback())
    {
        if (address.family() == Poco::Net::AddressFamily::IPv4)
        {
            /// The address is located in memory in big endian form (network byte order).
            const unsigned char * digits = static_cast<const unsigned char *>(address.addr());

            if (digits[0] == 127
                && digits[1] <= 1
                && digits[2] <= 1
                && digits[3] <= 1)
            {
                return true;
            }
        }
        else if (address.family() == Poco::Net::AddressFamily::IPv6)
        {
            return true;
        }
    }

    return NetworkInterfaces::instance().hasAddress(address);
}


bool isLocalAddress(const Poco::Net::SocketAddress & address, UInt16 clickhouse_port)
{
    return clickhouse_port == address.port() && isLocalAddress(address.host());
}


size_t getHostNameDifference(const std::string & local_hostname, const std::string & host)
{
    /// FIXME should we replace it with Levenstein distance? (we already have it in NamePrompter)
    size_t hostname_difference = 0;
    for (size_t i = 0; i < std::min(local_hostname.length(), host.length()); ++i)
        if (local_hostname[i] != host[i])
            ++hostname_difference;
    return hostname_difference;
}

}
