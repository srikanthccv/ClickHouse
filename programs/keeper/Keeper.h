#pragma once

#include <Server/IServer.h>
#include <daemon/BaseDaemon.h>

namespace Poco
{
    namespace Net
    {
        class ServerSocket;
    }
}

namespace DB
{

class Keeper : public BaseDaemon, public IServer
{
public:
    using ServerApplication::run;

    Poco::Util::LayeredConfiguration & config() const override
    {
        return BaseDaemon::config();
    }

    Poco::Logger & logger() const override
    {
        return BaseDaemon::logger();
    }

    ContextPtr context() const override
    {
        return global_context;
    }

    bool isCancelled() const override
    {
        return BaseDaemon::isCancelled();
    }

    void defineOptions(Poco::Util::OptionSet & _options) override;

protected:
    int run() override;

    void initialize(Application & self) override;

    void uninitialize() override;

    int main(const std::vector<std::string> & args) override;

    std::string getDefaultCorePath() const override;

private:
    ContextPtr global_context;

    Poco::Net::SocketAddress socketBindListen(Poco::Net::ServerSocket & socket, const std::string & host, UInt16 port, [[maybe_unused]] bool secure = false) const;

    using CreateServerFunc = std::function<void(UInt16)>;
    void createServer(const std::string & listen_host, const char * port_name, bool listen_try, CreateServerFunc && func) const;
};

}
