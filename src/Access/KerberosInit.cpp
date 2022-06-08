#include <Access/KerberosInit.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>
#include <Poco/Logger.h>
#include <Loggers/Loggers.h>
#include <filesystem>

using namespace DB;

std::mutex KerberosInit::kinit_mtx;

int KerberosInit::init(const String & keytab_file, const String & principal, const String & cache_name)
{
    // Using mutex to prevent cache file corruptions
    std::unique_lock<std::mutex> lck(kinit_mtx);

    auto log = &Poco::Logger::get("ADQM");
    LOG_DEBUG(log,"Trying to authenticate to Kerberos v5");

    krb5_error_code ret;

    const char *deftype = nullptr;
    int flags = 0;

    if (!std::filesystem::exists(keytab_file))
        throw Exception("Error keytab file does not exist", ErrorCodes::KERBEROS_ERROR);

    memset(&k5, 0, sizeof(k5));
    ret = krb5_init_context(&k5.ctx);
    if (ret)
        throw Exception("Error while initializing Kerberos 5 library", ErrorCodes::KERBEROS_ERROR);

    if (!cache_name.empty())
    {
        ret = krb5_cc_resolve(k5.ctx, cache_name.c_str(), &k5.out_cc);
        if (ret)
            throw Exception("Error in resolving cache", ErrorCodes::KERBEROS_ERROR);
        LOG_DEBUG(log,"Resolved cache");
    }
    else
    {
        // Resolve the default ccache and get its type and default principal (if it is initialized).
        ret = krb5_cc_default(k5.ctx, &defcache);
        if (ret)
            throw Exception("Error while getting default ccache", ErrorCodes::KERBEROS_ERROR);
        LOG_DEBUG(log,"Resolved default cache");
        deftype = krb5_cc_get_type(k5.ctx, defcache);
        if (krb5_cc_get_principal(k5.ctx, defcache, &defcache_princ) != 0)
            defcache_princ = nullptr;
    }

    // Use the specified principal name.
    ret = krb5_parse_name_flags(k5.ctx, principal.c_str(), flags, &k5.me);
    if (ret)
        throw Exception("Error when parsing principal name " + principal, ErrorCodes::KERBEROS_ERROR);

    // Cache related commands
    if (k5.out_cc == nullptr && krb5_cc_support_switch(k5.ctx, deftype))
    {
        // Use an existing cache for the client principal if we can.
        ret = krb5_cc_cache_match(k5.ctx, k5.me, &k5.out_cc);
        if (ret && ret != KRB5_CC_NOTFOUND)
            throw Exception("Error while searching for cache for " + principal, ErrorCodes::KERBEROS_ERROR);
        if (!ret)
        {
            LOG_DEBUG(log,"Using default cache: {}", krb5_cc_get_name(k5.ctx, k5.out_cc));
            k5.switch_to_cache = 1;
        }
        else if (defcache_princ != nullptr)
        {
            // Create a new cache to avoid overwriting the initialized default cache.
            ret = krb5_cc_new_unique(k5.ctx, deftype, nullptr, &k5.out_cc);
            if (ret)
                throw Exception("Error while generating new cache", ErrorCodes::KERBEROS_ERROR);
            LOG_DEBUG(log,"Using default cache: {}", krb5_cc_get_name(k5.ctx, k5.out_cc));
            k5.switch_to_cache = 1;
        }
    }

    // Use the default cache if we haven't picked one yet.
    if (k5.out_cc == nullptr)
    {
        k5.out_cc = defcache;
        defcache = nullptr;
        LOG_DEBUG(log,"Using default cache: {}", krb5_cc_get_name(k5.ctx, k5.out_cc));
    }

    ret = krb5_unparse_name(k5.ctx, k5.me, &k5.name);
    if (ret)
        throw Exception("Error when unparsing name", ErrorCodes::KERBEROS_ERROR);
    LOG_DEBUG(log,"Using principal: {}", k5.name);

    memset(&my_creds, 0, sizeof(my_creds));
    ret = krb5_get_init_creds_opt_alloc(k5.ctx, &options);
    if (ret)
        throw Exception("Error in options allocation", ErrorCodes::KERBEROS_ERROR);

    // Resolve keytab
    ret = krb5_kt_resolve(k5.ctx, keytab_file.c_str(), &keytab);
    if (ret)
        throw Exception("Error in resolving keytab "+keytab_file, ErrorCodes::KERBEROS_ERROR);
    LOG_DEBUG(log,"Using keytab: {}", keytab_file);

    if (k5.in_cc)
    {
        ret = krb5_get_init_creds_opt_set_in_ccache(k5.ctx, options, k5.in_cc);
        if (ret)
            throw Exception("Error in setting input credential cache", ErrorCodes::KERBEROS_ERROR);
    }
    ret = krb5_get_init_creds_opt_set_out_ccache(k5.ctx, options, k5.out_cc);
    if (ret)
        throw Exception("Error in setting output credential cache", ErrorCodes::KERBEROS_ERROR);

    // Action: init or renew
    LOG_DEBUG(log,"Trying to renew credentials");
    ret = krb5_get_renewed_creds(k5.ctx, &my_creds, k5.me, k5.out_cc, nullptr);
    if (ret)
    {
        LOG_DEBUG(log,"Renew failed, trying to get initial credentials");
        ret = krb5_get_init_creds_keytab(k5.ctx, &my_creds, k5.me, keytab, 0, nullptr, options);
        if (ret)
            throw Exception("Error in getting initial credentials", ErrorCodes::KERBEROS_ERROR);
        else
            LOG_DEBUG(log,"Got initial credentials");
    }
    else
    {
        LOG_DEBUG(log,"Successfull renewal");
        ret = krb5_cc_initialize(k5.ctx, k5.out_cc, k5.me);
        if (ret)
            throw Exception("Error when initializing cache", ErrorCodes::KERBEROS_ERROR);
        LOG_DEBUG(log,"Initialized cache");
        ret = krb5_cc_store_cred(k5.ctx, k5.out_cc, &my_creds);
        if (ret)
            LOG_DEBUG(log,"Error while storing credentials");
        LOG_DEBUG(log,"Stored credentials");
    }

    if (k5.switch_to_cache) {
        ret = krb5_cc_switch(k5.ctx, k5.out_cc);
        if (ret)
            throw Exception("Error while switching to new ccache", ErrorCodes::KERBEROS_ERROR);
    }

    LOG_DEBUG(log,"Authenticated to Kerberos v5");
    return 0;
}

KerberosInit::~KerberosInit()
{
    std::unique_lock<std::mutex> lck(kinit_mtx);
    if (k5.ctx)
    {
        if (defcache)
            krb5_cc_close(k5.ctx, defcache);
        krb5_free_principal(k5.ctx, defcache_princ);

        if (options)
            krb5_get_init_creds_opt_free(k5.ctx, options);
        if (my_creds.client == k5.me)
            my_creds.client = nullptr;
        krb5_free_cred_contents(k5.ctx, &my_creds);
        if (keytab)
            krb5_kt_close(k5.ctx, keytab);

        krb5_free_unparsed_name(k5.ctx, k5.name);
        krb5_free_principal(k5.ctx, k5.me);
        if (k5.in_cc != nullptr)
            krb5_cc_close(k5.ctx, k5.in_cc);
        if (k5.out_cc != nullptr)
            krb5_cc_close(k5.ctx, k5.out_cc);
        krb5_free_context(k5.ctx);
    }
}
