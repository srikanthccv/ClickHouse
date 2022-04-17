#include <Backups/DistributedBackupCoordination.h>
#include <IO/ReadBufferFromString.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteBufferFromString.h>
#include <IO/WriteHelpers.h>
#include <Common/ZooKeeper/KeeperException.h>
#include <Common/escapeForFileName.h>
#include <Common/hex.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int UNEXPECTED_NODE_IN_ZOOKEEPER;
}

/// zookeeper_path/file_names/file_name->checksum
/// zookeeper_path/file_infos/checksum->info
/// zookeeper_path/archive_suffixes
/// zookeeper_path/current_archive_suffix

namespace
{
    using FileInfo = IBackupCoordination::FileInfo;

    String serializeFileInfo(const FileInfo & info)
    {
        WriteBufferFromOwnString out;
        writeBinary(info.file_name, out);
        writeBinary(info.size, out);
        writeBinary(info.checksum, out);
        writeBinary(info.base_size, out);
        writeBinary(info.base_checksum, out);
        writeBinary(info.archive_suffix, out);
        writeBinary(info.pos_in_archive, out);
        return out.str();
    }

    FileInfo deserializeFileInfo(const String & str)
    {
        FileInfo info;
        ReadBufferFromString in{str};
        readBinary(info.file_name, in);
        readBinary(info.size, in);
        readBinary(info.checksum, in);
        readBinary(info.base_size, in);
        readBinary(info.base_checksum, in);
        readBinary(info.archive_suffix, in);
        readBinary(info.pos_in_archive, in);
        return info;
    }

    String hexChecksum(UInt128 checksum)
    {
        return getHexUIntLowercase(checksum);
    }

    UInt128 unhexChecksum(const String & checksum)
    {
        if (checksum.size() != sizeof(UInt128) * 2)
            throw Exception(
                ErrorCodes::UNEXPECTED_NODE_IN_ZOOKEEPER,
                "Unexpected size of checksum: {}, must be {}",
                checksum.size(),
                sizeof(UInt128) * 2);
        return unhexUInt<UInt128>(checksum.data());
    }

    constexpr size_t NUM_ATTEMPTS = 10;
}

DistributedBackupCoordination::DistributedBackupCoordination(const String & zookeeper_path_, zkutil::GetZooKeeper get_zookeeper_)
    : zookeeper_path(zookeeper_path_), get_zookeeper(get_zookeeper_)
{
    createRootNodes();
}

DistributedBackupCoordination::~DistributedBackupCoordination() = default;

void DistributedBackupCoordination::createRootNodes()
{
    auto zookeeper = get_zookeeper();
    zookeeper->createAncestors(zookeeper_path);
    zookeeper->createIfNotExists(zookeeper_path, "");
    zookeeper->createIfNotExists(zookeeper_path + "/file_names", "");
    zookeeper->createIfNotExists(zookeeper_path + "/file_infos", "");
    zookeeper->createIfNotExists(zookeeper_path + "/archive_suffixes", "");
    zookeeper->createIfNotExists(zookeeper_path + "/current_archive_suffix", "0");
}

void DistributedBackupCoordination::removeAllNodes()
{
    auto zookeeper = get_zookeeper();
    zookeeper->removeRecursive(zookeeper_path);
}

void DistributedBackupCoordination::addFileInfo(const FileInfo & file_info, bool & is_new_checksum)
{
    auto zookeeper = get_zookeeper();

    String full_path = zookeeper_path + "/file_names/" + escapeForFileName(file_info.file_name);
    String checksum_str = hexChecksum(file_info.checksum);
    zookeeper->create(full_path, checksum_str, zkutil::CreateMode::Persistent);

    full_path = zookeeper_path + "/file_infos/" + checksum_str;
    auto code = zookeeper->tryCreate(full_path, serializeFileInfo(file_info), zkutil::CreateMode::Persistent);
    if ((code != Coordination::Error::ZOK) && (code != Coordination::Error::ZNODEEXISTS))
        throw zkutil::KeeperException(code, full_path);

    is_new_checksum = (code == Coordination::Error::ZOK);
}

void DistributedBackupCoordination::updateFileInfo(const FileInfo & file_info)
{
    auto zookeeper = get_zookeeper();
    String checksum_str = hexChecksum(file_info.checksum);
    String full_path = zookeeper_path + "/file_infos/" + checksum_str;
    for (size_t attempt = 0; attempt < NUM_ATTEMPTS; ++attempt)
    {
        Coordination::Stat stat;
        auto new_info = deserializeFileInfo(zookeeper->get(full_path, &stat));
        new_info.archive_suffix = file_info.archive_suffix;
        auto code = zookeeper->trySet(full_path, serializeFileInfo(new_info), stat.version);
        if (code == Coordination::Error::ZOK)
            return;
        bool is_last_attempt = (attempt == NUM_ATTEMPTS - 1);
        if ((code != Coordination::Error::ZBADVERSION) || is_last_attempt)
            throw zkutil::KeeperException(code, full_path);
    }
}

std::vector<FileInfo> DistributedBackupCoordination::getAllFileInfos()
{
    auto zookeeper = get_zookeeper();
    std::vector<FileInfo> file_infos;
    Strings escaped_names = zookeeper->getChildren(zookeeper_path + "/file_names");
    for (const String & escaped_name : escaped_names)
    {
        String checksum = zookeeper->get(zookeeper_path + "/file_names/" + escaped_name);
        FileInfo file_info = deserializeFileInfo(zookeeper->get(zookeeper_path + "/file_infos/" + checksum));
        file_info.file_name = unescapeForFileName(escaped_name);
        file_infos.emplace_back(std::move(file_info));
    }
    return file_infos;
}

Strings DistributedBackupCoordination::listFiles(const String & prefix, const String & terminator)
{
    auto zookeeper = get_zookeeper();
    Strings escaped_names = zookeeper->getChildren(zookeeper_path + "/file_names");

    Strings elements;
    for (const String & escaped_name : escaped_names)
    {
        String name = unescapeForFileName(escaped_name);
        if (!name.starts_with(prefix))
            continue;
        size_t start_pos = prefix.length();
        size_t end_pos = String::npos;
        if (!terminator.empty())
            end_pos = name.find(terminator, start_pos);
        std::string_view new_element = std::string_view{name}.substr(start_pos, end_pos - start_pos);
        if (!elements.empty() && (elements.back() == new_element))
            continue;
        elements.push_back(String{new_element});
    }

    std::sort(elements.begin(), elements.end());
    return elements;
}

std::optional<UInt128> DistributedBackupCoordination::getChecksumByFileName(const String & file_name)
{
    auto zookeeper = get_zookeeper();
    String checksum;
    if (!zookeeper->tryGet(zookeeper_path + "/file_names/" + escapeForFileName(file_name), checksum))
        return std::nullopt;
    return unhexChecksum(checksum);
}

std::optional<FileInfo> DistributedBackupCoordination::getFileInfoByChecksum(const UInt128 & checksum)
{
    auto zookeeper = get_zookeeper();
    String file_info_str;
    if (!zookeeper->tryGet(zookeeper_path + "/file_infos/" + hexChecksum(checksum), file_info_str))
        return std::nullopt;
    return deserializeFileInfo(file_info_str);
}

std::optional<FileInfo> DistributedBackupCoordination::getFileInfoByFileName(const String & file_name)
{
    auto zookeeper = get_zookeeper();
    String checksum;
    if (!zookeeper->tryGet(zookeeper_path + "/file_names/" + escapeForFileName(file_name), checksum))
        return std::nullopt;
    FileInfo file_info = deserializeFileInfo(zookeeper->get(zookeeper_path + "/file_infos/" + checksum));
    file_info.file_name = file_name;
    return file_info;
}

String DistributedBackupCoordination::getNextArchiveSuffix()
{
    auto zookeeper = get_zookeeper();
    for (size_t attempt = 0; attempt != NUM_ATTEMPTS; ++attempt)
    {
        Coordination::Stat stat;
        String current_suffix_str = zookeeper->get(zookeeper_path + "/current_archive_suffix", &stat);
        UInt64 current_suffix = parseFromString<UInt64>(current_suffix_str);
        current_suffix_str = fmt::format("{:03}", ++current_suffix); /// Outputs 001, 002, 003, ...
        Coordination::Requests ops;
        ops.emplace_back(zkutil::makeSetRequest(zookeeper_path + "/current_archive_suffix", current_suffix_str, stat.version));
        ops.emplace_back(zkutil::makeCreateRequest(zookeeper_path + "/archive_suffixes/" + current_suffix_str, "", zkutil::CreateMode::Persistent));
        Coordination::Responses responses;
        auto code = zookeeper->tryMulti(ops, responses);
        if (code == Coordination::Error::ZOK)
            return current_suffix_str;
        bool is_last_attempt = (attempt == NUM_ATTEMPTS - 1);
        if ((responses[0]->error != Coordination::Error::ZBADVERSION) || is_last_attempt)
            throw zkutil::KeeperMultiException(code, ops, responses);
    }
    __builtin_unreachable();
}

Strings DistributedBackupCoordination::getAllArchiveSuffixes()
{
    auto zookeeper = get_zookeeper();
    return zookeeper->getChildren(zookeeper_path + "/archive_suffixes");
}

void DistributedBackupCoordination::drop()
{
    removeAllNodes();
}

}
