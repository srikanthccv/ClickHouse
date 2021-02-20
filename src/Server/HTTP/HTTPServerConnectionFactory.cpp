#include <Server/HTTP/HTTPServerConnectionFactory.h>

#include <Server/HTTP/HTTPServerConnection.h>

namespace DB
{
HTTPServerConnectionFactory::HTTPServerConnectionFactory(
    const Context & context_, Poco::Net::HTTPServerParams::Ptr params_, HTTPRequestHandlerFactoryPtr factory_)
    : context(context_), params(params_), factory(factory_)
{
    poco_check_ptr(factory);
}

Poco::Net::TCPServerConnection * HTTPServerConnectionFactory::createConnection(const Poco::Net::StreamSocket & socket)
{
    return new HTTPServerConnection(context, socket, params, factory);
}

}
