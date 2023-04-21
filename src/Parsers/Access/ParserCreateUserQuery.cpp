#include <Parsers/Access/ParserCreateUserQuery.h>
#include <Parsers/Access/ASTCreateUserQuery.h>
#include <Parsers/Access/ASTRolesOrUsersSet.h>
#include <Parsers/Access/ASTSettingsProfileElement.h>
#include <Parsers/Access/ASTUserNameWithHost.h>
#include <Parsers/Access/ASTAuthenticationData.h>
#include <Parsers/Access/ParserRolesOrUsersSet.h>
#include <Parsers/Access/ParserSettingsProfileElement.h>
#include <Parsers/Access/ParserUserNameWithHost.h>
#include <Parsers/Access/parseUserName.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/CommonParsers.h>
#include <Parsers/ExpressionElementParsers.h>
#include <Parsers/ExpressionListParsers.h>
#include <Parsers/ParserDatabaseOrNone.h>
#include <Parsers/parseIdentifierOrStringLiteral.h>
#include <base/range.h>
#include <boost/algorithm/string/predicate.hpp>
#include <base/insertAtEnd.h>
#include "config.h"


namespace DB
{

namespace
{
    bool parseRenameTo(IParserBase::Pos & pos, Expected & expected, std::optional<String> & new_name)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!ParserKeyword{"RENAME TO"}.ignore(pos, expected))
                return false;

            String maybe_new_name;
            if (!parseUserName(pos, expected, maybe_new_name))
                return false;

            new_name.emplace(std::move(maybe_new_name));
            return true;
        });
    }

    class ParserStringAndSubstitution : public IParserBase
    {
    private:
        const char * getName() const override { return "ParserStringAndSubstitution"; }
        bool parseImpl(Pos & pos, ASTPtr & node, Expected & expected) override
        {
            return ParserStringLiteral{}.parse(pos, node, expected) || ParserSubstitution{}.parse(pos, node, expected);
        }

    public:
        explicit ParserStringAndSubstitution() = default;
    };


    bool parseAuthenticationData(IParserBase::Pos & pos, Expected & expected, std::shared_ptr<ASTAuthenticationData> & auth_data)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (ParserKeyword{"NOT IDENTIFIED"}.ignore(pos, expected))
            {
                auth_data = std::make_shared<ASTAuthenticationData>();
                auth_data->type = AuthenticationType::NO_PASSWORD;

                return true;
            }

            if (!ParserKeyword{"IDENTIFIED"}.ignore(pos, expected))
                return false;

            std::optional<AuthenticationType> type;

            bool expect_password = false;
            bool expect_hash = false;
            bool expect_ldap_server_name = false;
            bool expect_kerberos_realm = false;
            bool expect_common_names = false;

            if (ParserKeyword{"WITH"}.ignore(pos, expected))
            {
                for (auto check_type : collections::range(AuthenticationType::MAX))
                {
                    if (ParserKeyword{AuthenticationTypeInfo::get(check_type).raw_name}.ignore(pos, expected))
                    {
                        type = check_type;

                        if (check_type == AuthenticationType::LDAP)
                            expect_ldap_server_name = true;
                        else if (check_type == AuthenticationType::KERBEROS)
                            expect_kerberos_realm = true;
                        else if (check_type == AuthenticationType::SSL_CERTIFICATE)
                            expect_common_names = true;
                        else if (check_type != AuthenticationType::NO_PASSWORD)
                            expect_password = true;

                        break;
                    }
                }

                if (!type)
                {
                    if (ParserKeyword{"SHA256_HASH"}.ignore(pos, expected))
                    {
                        type = AuthenticationType::SHA256_PASSWORD;
                        expect_hash = true;
                    }
                    else if (ParserKeyword{"DOUBLE_SHA1_HASH"}.ignore(pos, expected))
                    {
                        type = AuthenticationType::DOUBLE_SHA1_PASSWORD;
                        expect_hash = true;
                    }
                    else
                        return false;
                }
            }

            /// If authentication type is not specified, then the default password type is used
            if (!type)
                expect_password = true;

            ASTPtr value;
            ASTPtr parsed_salt;
            ASTPtr common_names;
            if (expect_password || expect_hash)
            {
                if (!ParserKeyword{"BY"}.ignore(pos, expected) || !ParserStringAndSubstitution{}.parse(pos, value, expected))
                    return false;

                if (expect_hash && type == AuthenticationType::SHA256_PASSWORD)
                {
                    if (ParserKeyword{"SALT"}.ignore(pos, expected))
                    {
                        if (!ParserStringAndSubstitution{}.parse(pos, parsed_salt, expected))
                            return false;
                    }
                }
            }
            else if (expect_ldap_server_name)
            {
                if (!ParserKeyword{"SERVER"}.ignore(pos, expected) || !ParserStringAndSubstitution{}.parse(pos, value, expected))
                    return false;
            }
            else if (expect_kerberos_realm)
            {
                if (ParserKeyword{"REALM"}.ignore(pos, expected))
                {
                    if (!ParserStringAndSubstitution{}.parse(pos, value, expected))
                        return false;
                }
            }
            else if (expect_common_names)
            {
                if (!ParserKeyword{"CN"}.ignore(pos, expected))
                    return false;

                if (!ParserList{std::make_unique<ParserStringAndSubstitution>(), std::make_unique<ParserToken>(TokenType::Comma), false}.parse(pos, common_names, expected))
                    return false;
            }

            auth_data = std::make_shared<ASTAuthenticationData>();

            auth_data->type = type;
            auth_data->is_password = expect_password;
            auth_data->is_hash = expect_hash;

            if (value)
                auth_data->children.push_back(std::move(value));

            if (parsed_salt)
                auth_data->children.push_back(std::move(parsed_salt));

            if (common_names)
                auth_data->children = std::move(common_names->children);

            return true;
        });
    }


    bool parseHostsWithoutPrefix(IParserBase::Pos & pos, Expected & expected, AllowedClientHosts & hosts)
    {
        AllowedClientHosts res_hosts;

        auto parse_host = [&]
        {
            if (ParserKeyword{"NONE"}.ignore(pos, expected))
                return true;

            if (ParserKeyword{"ANY"}.ignore(pos, expected))
            {
                res_hosts.addAnyHost();
                return true;
            }

            if (ParserKeyword{"LOCAL"}.ignore(pos, expected))
            {
                res_hosts.addLocalHost();
                return true;
            }

            if (ParserKeyword{"REGEXP"}.ignore(pos, expected))
            {
                ASTPtr ast;
                if (!ParserList{std::make_unique<ParserStringLiteral>(), std::make_unique<ParserToken>(TokenType::Comma), false}.parse(pos, ast, expected))
                    return false;

                for (const auto & name_regexp_ast : ast->children)
                    res_hosts.addNameRegexp(name_regexp_ast->as<const ASTLiteral &>().value.safeGet<String>());
                return true;
            }

            if (ParserKeyword{"NAME"}.ignore(pos, expected))
            {
                ASTPtr ast;
                if (!ParserList{std::make_unique<ParserStringLiteral>(), std::make_unique<ParserToken>(TokenType::Comma), false}.parse(pos, ast, expected))
                    return false;

                for (const auto & name_ast : ast->children)
                    res_hosts.addName(name_ast->as<const ASTLiteral &>().value.safeGet<String>());

                return true;
            }

            if (ParserKeyword{"IP"}.ignore(pos, expected))
            {
                ASTPtr ast;
                if (!ParserList{std::make_unique<ParserStringLiteral>(), std::make_unique<ParserToken>(TokenType::Comma), false}.parse(pos, ast, expected))
                    return false;

                for (const auto & subnet_ast : ast->children)
                    res_hosts.addSubnet(subnet_ast->as<const ASTLiteral &>().value.safeGet<String>());

                return true;
            }

            if (ParserKeyword{"LIKE"}.ignore(pos, expected))
            {
                ASTPtr ast;
                if (!ParserList{std::make_unique<ParserStringLiteral>(), std::make_unique<ParserToken>(TokenType::Comma), false}.parse(pos, ast, expected))
                    return false;

                for (const auto & pattern_ast : ast->children)
                    res_hosts.addLikePattern(pattern_ast->as<const ASTLiteral &>().value.safeGet<String>());

                return true;
            }

            return false;
        };

        if (!ParserList::parseUtil(pos, expected, parse_host, false))
            return false;

        hosts = std::move(res_hosts);
        return true;
    }


    bool parseHosts(IParserBase::Pos & pos, Expected & expected, std::string_view prefix, AllowedClientHosts & hosts)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!prefix.empty() && !ParserKeyword{prefix}.ignore(pos, expected))
                return false;

            if (!ParserKeyword{"HOST"}.ignore(pos, expected))
                return false;

            AllowedClientHosts res_hosts;
            if (!parseHostsWithoutPrefix(pos, expected, res_hosts))
                return false;

            hosts.add(res_hosts);
            return true;
        });
    }


    bool parseDefaultRoles(IParserBase::Pos & pos, Expected & expected, bool id_mode, std::shared_ptr<ASTRolesOrUsersSet> & default_roles)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!ParserKeyword{"DEFAULT ROLE"}.ignore(pos, expected))
                return false;

            ASTPtr ast;
            ParserRolesOrUsersSet default_roles_p;
            default_roles_p.allowAll().allowRoles().useIDMode(id_mode);
            if (!default_roles_p.parse(pos, ast, expected))
                return false;

            default_roles = typeid_cast<std::shared_ptr<ASTRolesOrUsersSet>>(ast);
            default_roles->allow_users = false;
            return true;
        });
    }


    bool parseSettings(IParserBase::Pos & pos, Expected & expected, bool id_mode, std::vector<std::shared_ptr<ASTSettingsProfileElement>> & settings)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!ParserKeyword{"SETTINGS"}.ignore(pos, expected))
                return false;

            ASTPtr new_settings_ast;
            ParserSettingsProfileElements elements_p;
            elements_p.useIDMode(id_mode);
            if (!elements_p.parse(pos, new_settings_ast, expected))
                return false;

            settings = std::move(new_settings_ast->as<ASTSettingsProfileElements &>().elements);
            return true;
        });
    }

    bool parseGrantees(IParserBase::Pos & pos, Expected & expected, bool id_mode, std::shared_ptr<ASTRolesOrUsersSet> & grantees)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!ParserKeyword{"GRANTEES"}.ignore(pos, expected))
                return false;

            ASTPtr ast;
            ParserRolesOrUsersSet grantees_p;
            grantees_p.allowAny().allowUsers().allowCurrentUser().allowRoles().useIDMode(id_mode);
            if (!grantees_p.parse(pos, ast, expected))
                return false;

            grantees = typeid_cast<std::shared_ptr<ASTRolesOrUsersSet>>(ast);
            return true;
        });
    }

    bool parseOnCluster(IParserBase::Pos & pos, Expected & expected, String & cluster)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            return ParserKeyword{"ON"}.ignore(pos, expected) && ASTQueryWithOnCluster::parse(pos, cluster, expected);
        });
    }

    bool parseDefaultDatabase(IParserBase::Pos & pos, Expected & expected, std::shared_ptr<ASTDatabaseOrNone> & default_database)
    {
        return IParserBase::wrapParseImpl(pos, [&]
        {
            if (!ParserKeyword{"DEFAULT DATABASE"}.ignore(pos, expected))
                return false;

            ASTPtr ast;
            ParserDatabaseOrNone database_p;
            if (!database_p.parse(pos, ast, expected))
                return false;

            default_database = typeid_cast<std::shared_ptr<ASTDatabaseOrNone>>(ast);
            return true;
        });
    }
}


bool ParserCreateUserQuery::parseImpl(Pos & pos, ASTPtr & node, Expected & expected)
{
    bool alter = false;
    if (attach_mode)
    {
        if (!ParserKeyword{"ATTACH USER"}.ignore(pos, expected))
            return false;
    }
    else
    {
        if (ParserKeyword{"ALTER USER"}.ignore(pos, expected))
            alter = true;
        else if (!ParserKeyword{"CREATE USER"}.ignore(pos, expected))
            return false;
    }

    bool if_exists = false;
    bool if_not_exists = false;
    bool or_replace = false;
    if (alter)
    {
        if (ParserKeyword{"IF EXISTS"}.ignore(pos, expected))
            if_exists = true;
    }
    else
    {
        if (ParserKeyword{"IF NOT EXISTS"}.ignore(pos, expected))
            if_not_exists = true;
        else if (ParserKeyword{"OR REPLACE"}.ignore(pos, expected))
            or_replace = true;
    }

    ASTPtr names_ast;
    if (!ParserUserNamesWithHost{}.parse(pos, names_ast, expected))
        return false;
    auto names = typeid_cast<std::shared_ptr<ASTUserNamesWithHost>>(names_ast);
    auto names_ref = names->names;

    std::optional<String> new_name;
    std::optional<String> temporary_password;
    std::optional<AllowedClientHosts> hosts;
    std::optional<AllowedClientHosts> add_hosts;
    std::optional<AllowedClientHosts> remove_hosts;
    std::shared_ptr<ASTAuthenticationData> auth_data;
    std::shared_ptr<ASTRolesOrUsersSet> default_roles;
    std::shared_ptr<ASTSettingsProfileElements> settings;
    std::shared_ptr<ASTRolesOrUsersSet> grantees;
    std::shared_ptr<ASTDatabaseOrNone> default_database;
    String cluster;

    while (true)
    {
        if (!auth_data)
        {
            std::shared_ptr<ASTAuthenticationData> new_auth_data;
            if (parseAuthenticationData(pos, expected, new_auth_data))
            {
                auth_data = std::move(new_auth_data);
                continue;
            }
        }

        AllowedClientHosts new_hosts;
        if (parseHosts(pos, expected, "", new_hosts))
        {
            if (!hosts)
                hosts.emplace();
            hosts->add(new_hosts);
            continue;
        }

        std::vector<std::shared_ptr<ASTSettingsProfileElement>> new_settings;
        if (parseSettings(pos, expected, attach_mode, new_settings))
        {
            if (!settings)
                settings = std::make_shared<ASTSettingsProfileElements>();

            insertAtEnd(settings->elements, std::move(new_settings));
            continue;
        }

        if (!default_roles && parseDefaultRoles(pos, expected, attach_mode, default_roles))
            continue;

        if (cluster.empty() && parseOnCluster(pos, expected, cluster))
            continue;

        if (!grantees && parseGrantees(pos, expected, attach_mode, grantees))
            continue;

        if (!default_database && parseDefaultDatabase(pos, expected, default_database))
            continue;

        if (alter)
        {
            if (!new_name && (names->size() == 1) && parseRenameTo(pos, expected, new_name))
                continue;

            if (parseHosts(pos, expected, "ADD", new_hosts))
            {
                if (!add_hosts)
                    add_hosts.emplace();
                add_hosts->add(new_hosts);
                continue;
            }

            if (parseHosts(pos, expected, "DROP", new_hosts))
            {
                if (!remove_hosts)
                    remove_hosts.emplace();
                remove_hosts->add(new_hosts);
                continue;
            }
        }

        break;
    }

    if (!alter && !hosts)
    {
        String common_host_pattern;
        if (names->getHostPatternIfCommon(common_host_pattern) && !common_host_pattern.empty())
        {
            hosts.emplace().addLikePattern(common_host_pattern);
            names->concatParts();
        }
    }
    else if (alter)
        names->concatParts();

    auto query = std::make_shared<ASTCreateUserQuery>();
    node = query;

    query->alter = alter;
    query->attach = attach_mode;
    query->if_exists = if_exists;
    query->if_not_exists = if_not_exists;
    query->or_replace = or_replace;
    query->cluster = std::move(cluster);
    query->names = std::move(names);
    query->new_name = std::move(new_name);
    query->auth_data = std::move(auth_data);
    query->hosts = std::move(hosts);
    query->add_hosts = std::move(add_hosts);
    query->remove_hosts = std::move(remove_hosts);
    query->default_roles = std::move(default_roles);
    query->settings = std::move(settings);
    query->grantees = std::move(grantees);
    query->default_database = std::move(default_database);

    if (query->auth_data)
        query->children.push_back(query->auth_data);

    return true;
}
}
