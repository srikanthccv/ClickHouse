#include <Interpreters/ExternalDictionariesLoader.h>
#include <Interpreters/Context.h>
#include <Dictionaries/DictionaryFactory.h>
#include <Dictionaries/getDictionaryConfigurationFromAST.h>

namespace DB
{

/// Must not acquire Context lock in constructor to avoid possibility of deadlocks.
ExternalDictionariesLoader::ExternalDictionariesLoader(Context & context_)
    : ExternalLoader("external dictionary", &Logger::get("ExternalDictionariesLoader"))
    , context(context_)
{
    enableAsyncLoading(true);
    enablePeriodicUpdates(true);
}


ExternalLoader::LoadablePtr ExternalDictionariesLoader::create(
        const std::string & name, const Poco::Util::AbstractConfiguration & config, const std::string & key_in_config) const
{
    /// For dictionaries from databases (created with DDL qureies) we have to perform
    /// additional checks, so we identify them here.
    bool dictionary_from_database = !key_in_config.empty();
    return DictionaryFactory::instance().create(name, config, key_in_config, context, dictionary_from_database);
}

void ExternalDictionariesLoader::addConfigRepository(
    const std::string & repository_name, std::unique_ptr<IExternalLoaderConfigRepository> config_repository)
{
    ExternalLoader::addConfigRepository(repository_name, std::move(config_repository), {"dictionary", "name"});
}


void ExternalDictionariesLoader::addDictionaryWithConfig(
    const String & dictionary_name, const String & repo_name, const ASTCreateQuery & query, bool load_never_loading) const
{
    ExternalLoader::addObjectAndLoad(
        dictionary_name, /// names are equal
        dictionary_name,
        repo_name,
        getDictionaryConfigurationFromAST(query),
        "dictionary", load_never_loading);
}
}
