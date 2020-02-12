#pragma once

#include <Access/AccessRights.h>
#include <Interpreters/ClientInfo.h>
#include <Core/UUID.h>
#include <ext/scope_guard.h>
#include <ext/shared_ptr_helper.h>
#include <boost/smart_ptr/atomic_shared_ptr.hpp>
#include <mutex>


namespace Poco { class Logger; }

namespace DB
{
struct User;
using UserPtr = std::shared_ptr<const User>;
struct RowPolicyContext;
using RowPolicyContextPtr = std::shared_ptr<const RowPolicyContext>;
struct QuotaContext;
using QuotaContextPtr = std::shared_ptr<const QuotaContext>;
struct Settings;
class AccessControlManager;


class AccessRightsContext
{
public:
    struct Params
    {
        std::optional<UUID> user_id;
        UInt64 readonly = 0;
        bool allow_ddl = false;
        bool allow_introspection = false;
        String current_database;
        ClientInfo::Interface interface = ClientInfo::Interface::TCP;
        ClientInfo::HTTPMethod http_method = ClientInfo::HTTPMethod::UNKNOWN;
        Poco::Net::IPAddress address;
        String quota_key;

        friend bool operator ==(const Params & lhs, const Params & rhs);
        friend bool operator !=(const Params & lhs, const Params & rhs) { return !(lhs == rhs); }
        friend bool operator <(const Params & lhs, const Params & rhs);
        friend bool operator >(const Params & lhs, const Params & rhs) { return rhs < lhs; }
        friend bool operator <=(const Params & lhs, const Params & rhs) { return !(rhs < lhs); }
        friend bool operator >=(const Params & lhs, const Params & rhs) { return !(lhs < rhs); }
    };

    /// Default constructor creates access rights' context which allows everything.
    AccessRightsContext();

    const Params & getParams() const { return params; }
    UserPtr getUser() const;
    String getUserName() const;

    void checkPassword(const String & password) const;
    void checkHostIsAllowed() const;

    RowPolicyContextPtr getRowPolicy() const;
    QuotaContextPtr getQuota() const;

    /// Checks if a specified access granted, and throws an exception if not.
    /// Empty database means the current database.
    void checkAccess(const AccessFlags & access) const;
    void checkAccess(const AccessFlags & access, const std::string_view & database) const;
    void checkAccess(const AccessFlags & access, const std::string_view & database, const std::string_view & table) const;
    void checkAccess(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::string_view & column) const;
    void checkAccess(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::vector<std::string_view> & columns) const;
    void checkAccess(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const Strings & columns) const;
    void checkAccess(const AccessRightsElement & access) const;
    void checkAccess(const AccessRightsElements & access) const;

    /// Checks if a specified access granted.
    bool isGranted(const AccessFlags & access) const;
    bool isGranted(const AccessFlags & access, const std::string_view & database) const;
    bool isGranted(const AccessFlags & access, const std::string_view & database, const std::string_view & table) const;
    bool isGranted(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::string_view & column) const;
    bool isGranted(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::vector<std::string_view> & columns) const;
    bool isGranted(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const Strings & columns) const;
    bool isGranted(const AccessRightsElement & access) const;
    bool isGranted(const AccessRightsElements & access) const;

    /// Checks if a specified access granted, and logs a warning if not.
    bool isGranted(Poco::Logger * log_, const AccessFlags & access) const;
    bool isGranted(Poco::Logger * log_, const AccessFlags & access, const std::string_view & database) const;
    bool isGranted(Poco::Logger * log_, const AccessFlags & access, const std::string_view & database, const std::string_view & table) const;
    bool isGranted(Poco::Logger * log_, const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::string_view & column) const;
    bool isGranted(Poco::Logger * log_, const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::vector<std::string_view> & columns) const;
    bool isGranted(Poco::Logger * log_, const AccessFlags & access, const std::string_view & database, const std::string_view & table, const Strings & columns) const;
    bool isGranted(Poco::Logger * log_, const AccessRightsElement & access) const;
    bool isGranted(Poco::Logger * log_, const AccessRightsElements & access) const;

    /// Checks if a specified access granted with grant option, and throws an exception if not.
    void checkGrantOption(const AccessFlags & access) const;
    void checkGrantOption(const AccessFlags & access, const std::string_view & database) const;
    void checkGrantOption(const AccessFlags & access, const std::string_view & database, const std::string_view & table) const;
    void checkGrantOption(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::string_view & column) const;
    void checkGrantOption(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const std::vector<std::string_view> & columns) const;
    void checkGrantOption(const AccessFlags & access, const std::string_view & database, const std::string_view & table, const Strings & columns) const;
    void checkGrantOption(const AccessRightsElement & access) const;
    void checkGrantOption(const AccessRightsElements & access) const;

private:
    friend class AccessRightsContextFactory;
    friend struct ext::shared_ptr_helper<AccessRightsContext>;
    AccessRightsContext(const AccessControlManager & manager_, const Params & params_); /// AccessRightsContext should be created by AccessRightsContextFactory.

    void setUser(const UserPtr & user_) const;

    template <int mode, bool grant_option, typename... Args>
    bool checkAccessImpl(Poco::Logger * log_, const AccessFlags & access, const Args &... args) const;

    template <int mode, bool grant_option>
    bool checkAccessImpl(Poco::Logger * log_, const AccessRightsElement & access) const;

    template <int mode, bool grant_option>
    bool checkAccessImpl(Poco::Logger * log_, const AccessRightsElements & access) const;

    boost::shared_ptr<const AccessRights> calculateResultAccess(bool grant_option) const;
    boost::shared_ptr<const AccessRights> calculateResultAccess(bool grant_option, UInt64 readonly_, bool allow_ddl_, bool allow_introspection_) const;

    const AccessControlManager * manager = nullptr;
    const Params params;
    mutable Poco::Logger * trace_log = nullptr;
    mutable UserPtr user;
    mutable String user_name;
    mutable ext::scope_guard subscription_for_user_change;
    mutable boost::atomic_shared_ptr<const AccessRights> result_access_cache[7];
    mutable RowPolicyContextPtr row_policy_context;
    mutable QuotaContextPtr quota_context;
    mutable std::mutex mutex;
};

using AccessRightsContextPtr = std::shared_ptr<const AccessRightsContext>;

}
