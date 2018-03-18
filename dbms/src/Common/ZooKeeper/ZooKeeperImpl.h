#pragma once

#include <Core/Types.h>
#include <Common/ConcurrentBoundedQueue.h>

#include <IO/ReadBuffer.h>
#include <IO/WriteBuffer.h>
#include <IO/ReadBufferFromPocoSocket.h>
#include <IO/WriteBufferFromPocoSocket.h>

#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>

#include <map>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>


namespace ZooKeeperImpl
{

using namespace DB;


/** Usage scenario:
  * - create an object and issue commands;
  * - you provide callbacks for your commands; callbacks are invoked in internal thread and must be cheap:
  *   for example, just signal a condvar / fulfull a promise.
  * - you also may provide callbacks for watches; they are also invoked in internal thread and must be cheap.
  * - whenever you receive SessionExpired exception of method isValid returns false,
  *   the ZooKeeper instance is no longer usable - you may only destroy it and probably create another.
  * - whenever session is expired or ZooKeeper instance is destroying, all callbacks are notified with special event.
  * - data for callbacks must be alive when ZooKeeper instance is alive.
  */
class ZooKeeper
{
public:
    using Addresses = std::vector<Poco::Net::SocketAddress>;

    struct ACL
    {
        static constexpr int32_t Read = 1;
        static constexpr int32_t Write = 2;
        static constexpr int32_t Create = 4;
        static constexpr int32_t Delete = 8;
        static constexpr int32_t Admin = 16;
        static constexpr int32_t All = 0x1F;

        int32_t permissions;
        String scheme;
        String id;

        void write(WriteBuffer & out) const;
    };
    using ACLs = std::vector<ACL>;

    struct Stat
    {
        int64_t czxid;
        int64_t mzxid;
        int64_t ctime;
        int64_t mtime;
        int32_t version;
        int32_t cversion;
        int32_t aversion;
        int64_t ephemeralOwner;
        int32_t dataLength;
        int32_t numChildren;
        int64_t pzxid;

        void read(ReadBuffer & in);
    };

    using XID = int32_t;
    using OpNum = int32_t;

    struct Response
    {
        int32_t error = 0;
        virtual ~Response() {}
        virtual void readImpl(ReadBuffer &) = 0;

        virtual void removeRootPath(const String & /* root_path */) {}
    };

    using ResponsePtr = std::shared_ptr<Response>;
    using ResponseCallback = std::function<void(const Response &)>;

    struct Request
    {
        XID xid;

        virtual ~Request() {};
        virtual OpNum getOpNum() const = 0;

        /// Writes length, xid, op_num, then the rest.
        void write(WriteBuffer & out) const;
        virtual void writeImpl(WriteBuffer &) const = 0;

        virtual ResponsePtr makeResponse() const = 0;

        virtual void addRootPath(const String & /* root_path */) {};
        virtual String getPath() const = 0;
    };

    using RequestPtr = std::shared_ptr<Request>;
    using Requests = std::vector<RequestPtr>;

    struct HeartbeatRequest final : Request
    {
        OpNum getOpNum() const override { return 11; }
        void writeImpl(WriteBuffer &) const override {}
        ResponsePtr makeResponse() const override;
        String getPath() const override { return {}; }
    };

    struct HeartbeatResponse final : Response
    {
        void readImpl(ReadBuffer &) override {}
    };

    struct WatchResponse final : Response
    {
        int32_t type;
        int32_t state;
        String path;

        void readImpl(ReadBuffer &) override;
        void removeRootPath(const String & root_path) override;
    };

    using WatchCallback = std::function<void(const WatchResponse &)>;

    struct CloseRequest final : Request
    {
        OpNum getOpNum() const override { return -11; }
        void writeImpl(WriteBuffer &) const override {}
        ResponsePtr makeResponse() const override;
        String getPath() const override { return {}; }
    };

    struct CreateRequest final : Request
    {
        String path;
        String data;
        bool is_ephemeral;
        bool is_sequential;
        ACLs acls;

        OpNum getOpNum() const override { return 1; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct CreateResponse final : Response
    {
        String path_created;

        void readImpl(ReadBuffer &) override;
        void removeRootPath(const String & root_path) override;
    };

    using CreateCallback = std::function<void(const CreateResponse &)>;

    struct RemoveRequest final : Request
    {
        String path;
        int32_t version;

        OpNum getOpNum() const override { return 2; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct RemoveResponse final : Response
    {
        void readImpl(ReadBuffer &) override {}
    };

    using RemoveCallback = std::function<void(const RemoveResponse &)>;

    struct ExistsRequest final : Request
    {
        String path;

        OpNum getOpNum() const override { return 3; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct ExistsResponse final : Response
    {
        Stat stat;

        void readImpl(ReadBuffer &) override;
    };

    using ExistsCallback = std::function<void(const ExistsResponse &)>;

    struct GetRequest final : Request
    {
        String path;

        OpNum getOpNum() const override { return 4; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct GetResponse final : Response
    {
        String data;
        Stat stat;

        void readImpl(ReadBuffer &) override;
    };

    using GetCallback = std::function<void(const GetResponse &)>;

    struct SetRequest final : Request
    {
        String path;
        String data;
        int32_t version;

        OpNum getOpNum() const override { return 5; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct SetResponse final : Response
    {
        Stat stat;

        void readImpl(ReadBuffer &) override;
    };

    using SetCallback = std::function<void(const SetResponse &)>;

    struct ListRequest final : Request
    {
        String path;

        OpNum getOpNum() const override { return 12; }
        void writeImpl(WriteBuffer &) const override;
        ResponsePtr makeResponse() const override;
        void addRootPath(const String & root_path) override;
        String getPath() const override { return path; }
    };

    struct ListResponse final : Response
    {
        Stat stat;
        std::vector<String> names;

        void readImpl(ReadBuffer &) override;
    };

    using ListCallback = std::function<void(const ListResponse &)>;

    /// Connection to addresses is performed in order. If you want, shuffle them manually.
    ZooKeeper(
        const Addresses & addresses,
        const String & root_path,
        const String & auth_scheme,
        const String & auth_data,
        Poco::Timespan session_timeout,
        Poco::Timespan connection_timeout);

    ~ZooKeeper();

    /// If not valid, you can only destroy the object. All other methods will throw exception.
    bool isValid() const { return !expired; }

    void create(
        const String & path,
        const String & data,
        bool is_ephemeral,
        bool is_sequential,
        const ACLs & acls,
        CreateCallback callback);

    void remove(
        const String & path,
        int32_t version,
        RemoveCallback callback);

    void exists(
        const String & path,
        ExistsCallback callback,
        WatchCallback watch);

    void get(
        const String & path,
        GetCallback callback,
        WatchCallback watch);

    void set(
        const String & path,
        const String & data,
        int32_t version,
        SetCallback callback);

    void list(
        const String & path,
        ListCallback callback,
        WatchCallback watch);

    void multi();

    void close();


    enum ZOO_ERRORS
    {
        ZOK = 0,

        /** System and server-side errors.
          * This is never thrown by the server, it shouldn't be used other than
          * to indicate a range. Specifically error codes greater than this
          * value, but lesser than ZAPIERROR, are system errors.
          */
        ZSYSTEMERROR = -1,

        ZRUNTIMEINCONSISTENCY = -2, /// A runtime inconsistency was found
        ZDATAINCONSISTENCY = -3,    /// A data inconsistency was found
        ZCONNECTIONLOSS = -4,       /// Connection to the server has been lost
        ZMARSHALLINGERROR = -5,     /// Error while marshalling or unmarshalling data
        ZUNIMPLEMENTED = -6,        /// Operation is unimplemented
        ZOPERATIONTIMEOUT = -7,     /// Operation timeout
        ZBADARGUMENTS = -8,         /// Invalid arguments
        ZINVALIDSTATE = -9,         /// Invliad zhandle state

        /** API errors.
          * This is never thrown by the server, it shouldn't be used other than
          * to indicate a range. Specifically error codes greater than this
          * value are API errors.
          */
        ZAPIERROR = -100,

        ZNONODE = -101,                     /// Node does not exist
        ZNOAUTH = -102,                     /// Not authenticated
        ZBADVERSION = -103,                 /// Version conflict
        ZNOCHILDRENFOREPHEMERALS = -108,    /// Ephemeral nodes may not have children
        ZNODEEXISTS = -110,                 /// The node already exists
        ZNOTEMPTY = -111,                   /// The node has children
        ZSESSIONEXPIRED = -112,             /// The session has been expired by the server
        ZINVALIDCALLBACK = -113,            /// Invalid callback specified
        ZINVALIDACL = -114,                 /// Invalid ACL specified
        ZAUTHFAILED = -115,                 /// Client authentication failed
        ZCLOSING = -116,                    /// ZooKeeper is closing
        ZNOTHING = -117,                    /// (not error) no server responses to process
        ZSESSIONMOVED = -118                /// Session moved to another server, so operation is ignored
    };

    static const char * errorMessage(int32_t code);

private:
    String root_path;
    ACLs default_acls;

    Poco::Timespan session_timeout;

    Poco::Net::StreamSocket socket;
    std::optional<ReadBufferFromPocoSocket> in;
    std::optional<WriteBufferFromPocoSocket> out;

    struct RequestInfo
    {
        RequestPtr request;
        ResponseCallback callback;
        WatchCallback watch;
    };

    using RequestsQueue = ConcurrentBoundedQueue<RequestInfo>;

    RequestsQueue requests{1};
    void pushRequest(RequestInfo && info);

    using Operations = std::map<XID, RequestInfo>;

    Operations operations;
    std::mutex operations_mutex;

    using WatchCallbacks = std::vector<WatchCallback>;
    using Watches = std::map<String /* path */, WatchCallbacks>;

    Watches watches;
    std::mutex watches_mutex;

    std::thread send_thread;
    std::thread receive_thread;

    std::atomic<bool> stop {false};
    std::atomic<bool> expired {false};

    void connect(
        const Addresses & addresses,
        Poco::Timespan connection_timeout);

    void sendHandshake();
    void receiveHandshake();

    void sendAuth(const String & auth_scheme, const String & auth_data);

    void receiveEvent();

    void sendThread();
    void receiveThread();

    template <typename T>
    void write(const T &);

    template <typename T>
    void read(T &);
};

};
