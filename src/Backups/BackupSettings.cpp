#include <Backups/BackupSettings.h>
#include <Backups/BackupInfo.h>
#include <Parsers/ASTBackupQuery.h>
#include <Parsers/ASTSetQuery.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int UNKNOWN_SETTING;
}

BackupSettings BackupSettings::fromBackupQuery(const ASTBackupQuery & query)
{
    BackupSettings res;

    if (query.base_backup_name)
        res.base_backup_info = std::make_shared<BackupInfo>(BackupInfo::fromAST(*query.base_backup_name));

    if (query.settings)
    {
        const auto & settings = query.settings->as<const ASTSetQuery &>().changes;
        for (const auto & setting : settings)
        {
            if (setting.name == "structure_only")
                res.structure_only = setting.value.safeGet<bool>();
            else
                throw Exception(ErrorCodes::UNKNOWN_SETTING, "Unknown setting {}", setting.name);
        }
    }

    return res;
}

}
