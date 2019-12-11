#pragma once
#include <Common/config.h>
#include <Poco/Net/TCPServerConnection.h>
#include <Common/getFQDNOrHostName.h>
#include <Core/MySQLProtocol.h>
#include "IServer.h"

#if USE_POCO_NETSSL
#include <Poco/Net/SecureStreamSocket.h>
#endif

namespace DB
{
/// Handler for MySQL wire protocol connections. Allows to connect to ClickHouse using MySQL client.
class MySQLHandler : public Poco::Net::TCPServerConnection
{
public:
    MySQLHandler(IServer & server_, const Poco::Net::StreamSocket & socket_, bool ssl_enabled, size_t connection_id_);

    void run() final;

private:
    /// Enables SSL, if client requested.
    void finishHandshake(MySQLProtocol::HandshakeResponse &);

    void comQuery(ReadBuffer & payload);

    void comFieldList(ReadBuffer & payload);

    void comPing();

    void comInitDB(ReadBuffer & payload);

    void authenticate(const String & user_name, const String & auth_plugin_name, const String & auth_response);

    virtual void authPluginSSL();
    virtual void finishHandshakeSSL(size_t packet_size, char * buf, size_t pos, std::function<void(size_t)> read_bytes, MySQLProtocol::HandshakeResponse & packet);

    IServer & server;

protected:
    Poco::Logger * log;

    Context connection_context;

    std::shared_ptr<MySQLProtocol::PacketSender> packet_sender;

private:
    size_t connection_id = 0;

    size_t server_capability_flags = 0;
    size_t client_capability_flags = 0;

protected:
    std::unique_ptr<MySQLProtocol::Authentication::IPlugin> auth_plugin;

    std::shared_ptr<ReadBuffer> in;
    std::shared_ptr<WriteBuffer> out;

    bool secure_connection = false;

private:
    static const String show_table_status_replacement_query;
};

#if USE_SSL && USE_POCO_NETSSL
class MySQLHandlerSSL : public MySQLHandler
{
public:
    MySQLHandlerSSL(IServer & server_, const Poco::Net::StreamSocket & socket_, bool ssl_enabled, size_t connection_id_, RSA & public_key_, RSA & private_key_);

private:
    void authPluginSSL() override;
    void finishHandshakeSSL(size_t packet_size, char * buf, size_t pos, std::function<void(size_t)> read_bytes, MySQLProtocol::HandshakeResponse & packet) override;

    RSA & public_key;
    RSA & private_key;
    std::shared_ptr<Poco::Net::SecureStreamSocket> ss;
};
#endif

}
