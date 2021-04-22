#include <Storages/StorageDictionary.h>
#include <Storages/StorageFactory.h>
#include <DataTypes/DataTypesNumber.h>
#include <Dictionaries/DictionaryStructure.h>
#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Interpreters/ExternalDictionariesLoader.h>
#include <Interpreters/ExternalLoaderDictionaryStorageConfigRepository.h>
#include <Parsers/ASTLiteral.h>
#include <Common/quoteString.h>
#include <Processors/Sources/SourceFromInputStream.h>
#include <Processors/Pipe.h>
#include <IO/Operators.h>
#include <Dictionaries/getDictionaryConfigurationFromAST.h>
#include <Interpreters/Context.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int THERE_IS_NO_COLUMN;
    extern const int CANNOT_DETACH_DICTIONARY_AS_TABLE;
}

namespace
{
    void checkNamesAndTypesCompatibleWithDictionary(const String & dictionary_name, const ColumnsDescription & columns, const DictionaryStructure & dictionary_structure)
    {
        auto dictionary_names_and_types = StorageDictionary::getNamesAndTypes(dictionary_structure);
        std::set<NameAndTypePair> names_and_types_set(dictionary_names_and_types.begin(), dictionary_names_and_types.end());

        for (const auto & column : columns.getOrdinary())
        {
            if (names_and_types_set.find(column) == names_and_types_set.end())
            {
                throw Exception(ErrorCodes::THERE_IS_NO_COLUMN, "Not found column {} {} in dictionary {}. There are only columns {}",
                                column.name, column.type->getName(), backQuote(dictionary_name),
                                StorageDictionary::generateNamesAndTypesDescription(dictionary_names_and_types));
            }
        }
    }
}


NamesAndTypesList StorageDictionary::getNamesAndTypes(const DictionaryStructure & dictionary_structure)
{
    NamesAndTypesList dictionary_names_and_types;

    if (dictionary_structure.id)
        dictionary_names_and_types.emplace_back(dictionary_structure.id->name, std::make_shared<DataTypeUInt64>());

    /// In old-style (XML) configuration we don't have this attributes in the
    /// main attribute list, so we have to add them to columns list explicitly.
    /// In the new configuration (DDL) we have them both in range_* nodes and
    /// main attribute list, but for compatibility we add them before main
    /// attributes list.
    if (dictionary_structure.range_min)
        dictionary_names_and_types.emplace_back(dictionary_structure.range_min->name, dictionary_structure.range_min->type);

    if (dictionary_structure.range_max)
        dictionary_names_and_types.emplace_back(dictionary_structure.range_max->name, dictionary_structure.range_max->type);

    if (dictionary_structure.key)
    {
        for (const auto & attribute : *dictionary_structure.key)
            dictionary_names_and_types.emplace_back(attribute.name, attribute.type);
    }

    for (const auto & attribute : dictionary_structure.attributes)
    {
        /// Some attributes can be already added (range_min and range_max)
        if (!dictionary_names_and_types.contains(attribute.name))
            dictionary_names_and_types.emplace_back(attribute.name, attribute.type);
    }

    return dictionary_names_and_types;
}


String StorageDictionary::generateNamesAndTypesDescription(const NamesAndTypesList & list)
{
    WriteBufferFromOwnString ss;
    bool first = true;
    for (const auto & name_and_type : list)
    {
        if (!std::exchange(first, false))
            ss << ", ";
        ss << name_and_type.name << ' ' << name_and_type.type->getName();
    }
    return ss.str();
}

StorageDictionary::StorageDictionary(
    const StorageID & table_id_,
    const String & dictionary_name_,
    const ColumnsDescription & columns_,
    Location location_,
    ContextPtr context_)
    : IStorage(table_id_)
    , WithContext(context_->getGlobalContext())
    , dictionary_name(dictionary_name_)
    , location(location_)
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    setInMemoryMetadata(storage_metadata);
}


StorageDictionary::StorageDictionary(
    const StorageID & table_id_,
    const String & dictionary_name_,
    const DictionaryStructure & dictionary_structure_,
    Location location_,
    ContextPtr context_)
    : StorageDictionary(
        table_id_,
        dictionary_name_,
        ColumnsDescription{getNamesAndTypes(dictionary_structure_)},
        location_,
        context_)
{
}

StorageDictionary::StorageDictionary(
    const StorageID & table_id,
    LoadablesConfigurationPtr dictionary_configuration,
    ContextPtr context_)
    : StorageDictionary(
        table_id,
        table_id.getInternalDictionaryName(),
        context_->getExternalDictionariesLoader().getDictionaryStructure(*dictionary_configuration),
        Location::SameDatabaseAndNameAsDictionary,
        context_)
{
    update_time = Poco::Timestamp(time(nullptr));
    configuration = dictionary_configuration;

    /// TODO: Check if it is safe
    auto repository = std::make_unique<ExternalLoaderDictionaryStorageConfigRepository>(*this);
    remove_repository_callback = context_->getExternalDictionariesLoader().addConfigRepository(std::move(repository));
}

StorageDictionary::~StorageDictionary()
{
    drop();
}

void StorageDictionary::checkTableCanBeDropped() const
{
    if (location == Location::SameDatabaseAndNameAsDictionary)
        throw Exception("Cannot drop/detach dictionary " + backQuote(dictionary_name) + " as table, use DROP DICTIONARY or DETACH DICTIONARY query instead", ErrorCodes::CANNOT_DETACH_DICTIONARY_AS_TABLE);
    if (location == Location::DictionaryDatabase)
        throw Exception("Cannot drop/detach table " + getStorageID().getFullTableName() + " from a database with DICTIONARY engine", ErrorCodes::CANNOT_DETACH_DICTIONARY_AS_TABLE);
}

void StorageDictionary::checkTableCanBeDetached() const
{
    checkTableCanBeDropped();
}

Pipe StorageDictionary::read(
    const Names & column_names,
    const StorageMetadataPtr & /*metadata_snapshot*/,
    SelectQueryInfo & /*query_info*/,
    ContextPtr local_context,
    QueryProcessingStage::Enum /*processed_stage*/,
    const size_t max_block_size,
    const unsigned /*threads*/)
{
    auto dictionary = getContext()->getExternalDictionariesLoader().getDictionary(dictionary_name, local_context);
    auto stream = dictionary->getBlockInputStream(column_names, max_block_size);
    /// TODO: update dictionary interface for processors.
    return Pipe(std::make_shared<SourceFromInputStream>(stream));
}

void StorageDictionary::drop()
{
    std::lock_guard<std::mutex> lock(dictionary_config_mutex);
    remove_repository_callback.reset();
}

void StorageDictionary::shutdown()
{
    drop();
}

Poco::Timestamp StorageDictionary::getUpdateTime() const
{
    std::lock_guard<std::mutex> lock(dictionary_config_mutex);
    return update_time;
}

LoadablesConfigurationPtr StorageDictionary::getConfiguration() const
{
    std::lock_guard<std::mutex> lock(dictionary_config_mutex);
    return configuration;
}

void StorageDictionary::renameInMemory(const StorageID & new_table_id)
{
    if (configuration)
    {
        configuration->setString("dictionary.database", new_table_id.database_name);
        configuration->setString("dictionary.name", new_table_id.table_name);

        auto & external_dictionaries_loader = getContext()->getExternalDictionariesLoader();
        external_dictionaries_loader.reloadConfig(getStorageID().getInternalDictionaryName());

        auto result = external_dictionaries_loader.getLoadResult(getStorageID().getInternalDictionaryName());
        if (!result.object)
            return;

        const auto dictionary = std::static_pointer_cast<const IDictionary>(result.object);
        dictionary->updateDictionaryName(new_table_id);
    }

    IStorage::renameInMemory(new_table_id);
}

void registerStorageDictionary(StorageFactory & factory)
{
    factory.registerStorage("Dictionary", [](const StorageFactory::Arguments & args)
    {
        auto query = args.query;

        auto local_context = args.getLocalContext();

        if (query.is_dictionary)
        {
            /// Create dictionary storage that owns underlying dictionary
            auto abstract_dictionary_configuration = getDictionaryConfigurationFromAST(args.query, local_context, args.table_id.database_name);
            return StorageDictionary::create(args.table_id, abstract_dictionary_configuration, local_context);
        }
        else
        {
            /// Create dictionary storage that is view of underlying dictionary

            if (args.engine_args.size() != 1)
                throw Exception("Storage Dictionary requires single parameter: name of dictionary",
                    ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

            args.engine_args[0] = evaluateConstantExpressionOrIdentifierAsLiteral(args.engine_args[0], local_context);
            String dictionary_name = args.engine_args[0]->as<ASTLiteral &>().value.safeGet<String>();

            if (!args.attach)
            {
                const auto & dictionary = args.getContext()->getExternalDictionariesLoader().getDictionary(dictionary_name, args.getContext());
                const DictionaryStructure & dictionary_structure = dictionary->getStructure();
                checkNamesAndTypesCompatibleWithDictionary(dictionary_name, args.columns, dictionary_structure);
            }

            return StorageDictionary::create(args.table_id, dictionary_name, args.columns, StorageDictionary::Location::Custom, local_context);
        }
    });
}

}
