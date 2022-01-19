#include <Backups/BackupSettings.h>
#include <Backups/BackupInfo.h>
#include <Parsers/ASTBackupQuery.h>
#include <Parsers/ASTSetQuery.h>


namespace DB
{

BackupSettings BackupSettings::fromBackupQuery(const ASTBackupQuery & query)
{
    BackupSettings res;

    if (query.base_backup_name)
        res.base_backup_info = std::make_shared<BackupInfo>(BackupInfo::fromAST(*query.base_backup_name));

    return res;
}

}
