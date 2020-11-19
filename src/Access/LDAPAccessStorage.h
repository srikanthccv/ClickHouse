#pragma once

#include <Access/MemoryAccessStorage.h>
#include <Core/Types.h>
#include <ext/scope_guard.h>
#include <map>
#include <mutex>
#include <set>
#include <vector>


namespace Poco
{
    namespace Util
    {
        class AbstractConfiguration;
    }
}


namespace DB
{
class AccessControlManager;
struct LDAPSearchParams;
using LDAPSearchParamsList = std::vector<LDAPSearchParams>;
using LDAPSearchResults = std::set<String>;
using LDAPSearchResultsList = std::vector<LDAPSearchResults>;

/// Implementation of IAccessStorage which allows attaching users from a remote LDAP server.
/// Currently, any user name will be treated as a name of an existing remote user,
/// a user info entity will be created, with LDAP_SERVER authentication type.
class LDAPAccessStorage : public IAccessStorage
{
public:
    static constexpr char STORAGE_TYPE[] = "ldap";

    explicit LDAPAccessStorage(const String & storage_name_, AccessControlManager * access_control_manager_, const Poco::Util::AbstractConfiguration & config, const String & prefix);
    virtual ~LDAPAccessStorage() override = default;

    String getLDAPServerName() const;

public: // IAccessStorage implementations.
    virtual const char * getStorageType() const override;
    virtual String getStorageParamsJSON() const override;

private: // IAccessStorage implementations.
    virtual std::optional<UUID> findImpl(EntityType type, const String & name) const override;
    virtual std::vector<UUID> findAllImpl(EntityType type) const override;
    virtual bool existsImpl(const UUID & id) const override;
    virtual AccessEntityPtr readImpl(const UUID & id) const override;
    virtual String readNameImpl(const UUID & id) const override;
    virtual bool canInsertImpl(const AccessEntityPtr &) const override;
    virtual UUID insertImpl(const AccessEntityPtr & entity, bool replace_if_exists) override;
    virtual void removeImpl(const UUID & id) override;
    virtual void updateImpl(const UUID & id, const UpdateFunc & update_func) override;
    virtual ext::scope_guard subscribeForChangesImpl(const UUID & id, const OnChangedHandler & handler) const override;
    virtual ext::scope_guard subscribeForChangesImpl(EntityType type, const OnChangedHandler & handler) const override;
    virtual bool hasSubscriptionImpl(const UUID & id) const override;
    virtual bool hasSubscriptionImpl(EntityType type) const override;
    virtual UUID loginImpl(const String & user_name, const String & password, const Poco::Net::IPAddress & address, const ExternalAuthenticators & external_authenticators) const override;
    virtual UUID getIDOfLoggedUserImpl(const String & user_name) const override;

private:
    void setConfiguration(AccessControlManager * access_control_manager_, const Poco::Util::AbstractConfiguration & config, const String & prefix);
    void processRoleChange(const UUID & id, const AccessEntityPtr & entity);

    void applyRoleChangeNoLock(bool grant, const UUID & role_id, const String & role_name);
    void grantRolesNoLock(User & user, const LDAPSearchResultsList & external_roles) const;
    void updateRolesNoLock(const UUID & id, const String & user_name, const LDAPSearchResultsList & external_roles) const;
    std::set<String> mapExternalRolesNoLock(const String & user_name, const LDAPSearchResultsList & external_roles) const;
    bool isPasswordCorrectLDAPNoLock(const User & user, const String & password, const ExternalAuthenticators & external_authenticators, LDAPSearchResultsList & search_results) const;

    mutable std::recursive_mutex mutex;
    AccessControlManager * access_control_manager = nullptr;
    String ldap_server;
    LDAPSearchParamsList role_search_params;
    std::set<String> common_role_names;
    mutable std::map<String, std::set<String>> users_per_roles; // per-user roles: role name -> user names
    mutable std::map<UUID, String> granted_role_names;          // currently granted roles: role id -> role name
    mutable std::map<String, UUID> granted_role_ids;            // currently granted roles: role name -> role id
    mutable std::map<String, std::size_t> external_role_hashes; // user name -> LDAPSearchResultsList hash
    ext::scope_guard role_change_subscription;
    mutable MemoryAccessStorage memory_storage;
};
}
