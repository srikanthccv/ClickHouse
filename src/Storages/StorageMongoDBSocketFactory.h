#pragma once

#include <Poco/MongoDB/Connection.h>


namespace DB
{

class StorageMongoDBSocketFactory : public Poco::MongoDB::Connection::SocketFactory
{
public:
    virtual Poco::Net::StreamSocket createSocket(const std::string & host, int port, Poco::Timespan connectTimeout, bool secure) override;

private:
    Poco::Net::StreamSocket createPlainSocket(const std::string & host, int port, Poco::Timespan connectTimeout);
    Poco::Net::StreamSocket createSecureSocket(const std::string & host, int port, Poco::Timespan connectTimeout);
};

}
