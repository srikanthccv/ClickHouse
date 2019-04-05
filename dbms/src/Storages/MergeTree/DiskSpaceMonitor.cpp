#include <Storages/MergeTree/DiskSpaceMonitor.h>

#include <Common/escapeForFileName.h>

namespace DB
{

std::map<String, DiskSpaceMonitor::DiskReserve> DiskSpaceMonitor::reserved;
std::mutex DiskSpaceMonitor::mutex;

DiskSelector::DiskSelector(const Poco::Util::AbstractConfiguration & config, const std::string & config_prefix) {
    Poco::Util::AbstractConfiguration::Keys keys;
    config.keys(config_prefix, keys);

    constexpr auto default_disk_name = "default";
    for (const auto & disk_name : keys)
    {
        UInt64 keep_free_space_bytes = config.getUInt64(config_prefix + "." + disk_name + ".keep_free_space_bytes", 0);
        String path;
        if (config.has(config_prefix + "." + disk_name + ".path")) {
            path = config.getString(config_prefix + "." + disk_name + ".path");
        }

        if (disk_name == default_disk_name) {
            if (!path.empty()) {
                ///@TODO_IGR ASK Rename Default disk to smth? ClickHouse disk? DB disk?
                throw Exception("It is not possible to specify default disk path", ErrorCodes::UNKNOWN_ELEMENT_IN_CONFIG);
            }
        } else {
            if (path.empty()) {
                throw Exception("Disk path can not be empty. Disk " + disk_name, ErrorCodes::UNKNOWN_ELEMENT_IN_CONFIG);
            }
        }
        disks.emplace(disk_name, Disk(disk_name, path, keep_free_space_bytes));
    }
}

const Disk & DiskSelector::operator[](const String & name) const {
    auto it = disks.find(name);
    if (it == disks.end()) {
        throw Exception("Unknown disk " + name, ErrorCodes::UNKNOWN_ELEMENT_IN_CONFIG);
    }
    return it->second;
}

bool DiskSelector::has(const String & name) const {
    auto it = disks.find(name);
    return it != disks.end();
}

Schema::Volume::Volume(const Poco::Util::AbstractConfiguration & config, const std::string & config_prefix, const DiskSelector & disk_selector) {
    Poco::Util::AbstractConfiguration::Keys keys;
    config.keys(config_prefix, keys);

    /// Disk's names
    Strings disks_names;

    for (const auto & name : keys)
    {
        if (startsWith(name.data(), "disk")) {
            disks_names.push_back(config.getString(config_prefix + "." + name));
        } else if (name == "part_size_threshold_bytes") {
            max_data_part_size = config.getUInt64(config_prefix + "." + name, 0);
        }
        ///@TODO_IGR part_size_threshold_ratio which set max_data_part_size by total disk size
    }

    if (max_data_part_size == 0) {
        --max_data_part_size;
    }

    /// Get paths from disk's names
    for (const auto & disk_name : disks_names) {
        /// Disks operator [] may throw exception
        disks.push_back(disk_selector[disk_name]);
    }
}

Schema::Volume::Volume(const Volume & other, const String & default_path, const String & enclosed_dir)
    : max_data_part_size(other.max_data_part_size),
      disks(other.disks),
      last_used(0)
{
    auto dir = escapeForFileName(enclosed_dir);
    for (auto & disk : disks) {
        if (disk.getName() == "default") {
            disk.SetPath(default_path + dir + '/');
        } else {
            disk.addEnclosedDirToPath(dir);
        }
    }
}

DiskSpaceMonitor::ReservationPtr Schema::Volume::reserve(UInt64 expected_size) const {
    /// This volume can not store files which size greater than max_data_part_size
    if (expected_size > max_data_part_size) {
        return {};
    }
    /// Real order is not necessary
    size_t start_from = last_used.fetch_add(1u, std::memory_order_relaxed);
    for (size_t i = 0; i != disks.size(); ++i) {
        size_t index = (start_from + i) % disks.size();
        auto reservation = DiskSpaceMonitor::tryToReserve(disks[index], expected_size);
        if (reservation) {
            return reservation;
        }
    }
    return {};
}

UInt64 Schema::Volume::getMaxUnreservedFreeSpace() const {
    UInt64 res = 0;
    for (const auto & disk : disks) {
        ///@TODO_IGR ASK There is cycle with mutex locking inside(((
        res = std::max(res, DiskSpaceMonitor::getUnreservedFreeSpace(disk));
    }
    return res;
}

Schema::Schema(const Poco::Util::AbstractConfiguration & config, const std::string & config_prefix, const DiskSelector & disks) {
    Poco::Util::AbstractConfiguration::Keys keys;
    config.keys(config_prefix, keys);

    for (const auto & name : keys)
    {
        if (!startsWith(name.data(), "volume"))
            throw Exception("Unknown element in config: " + config_prefix + "." + name + ", must be 'volume'",\
                            ErrorCodes::UNKNOWN_ELEMENT_IN_CONFIG);
        volumes.emplace_back(config, config_prefix + "." + name, disks);
    }
}

///@TODO_IGR ASK maybe iterator without copy?
Strings Schema::getFullPaths() const {
    Strings res;
    for (const auto & volume : volumes) {
        for (const auto & disk : volume.disks) {
            res.push_back(disk.getPath());
        }
    }
    return res;
}

UInt64 Schema::getMaxUnreservedFreeSpace() const {
    UInt64 res = 0;
    for (const auto & volume : volumes) {
        res = std::max(res, volume.getMaxUnreservedFreeSpace());
    }
    return res;
}

DiskSpaceMonitor::ReservationPtr Schema::reserve(UInt64 expected_size) const {
    for (auto & volume : volumes) {
        auto reservation = volume.reserve(expected_size);
        if (reservation) {
            return reservation;
        }
    }
    return {};
}

SchemaSelector::SchemaSelector(const Poco::Util::AbstractConfiguration & config, String config_prefix) {
    DiskSelector disks(config, config_prefix + ".disks");

    constexpr auto default_disk_name = "default";
    if (!disks.has(default_disk_name)) {
        std::cerr << "No default disk settings" << std::endl;
        disks.add(Disk(default_disk_name, "", 0));
    }

    config_prefix += ".schemes";

    Poco::Util::AbstractConfiguration::Keys keys;
    config.keys(config_prefix, keys);

    for (const auto & name : keys)
    {
        ///@TODO_IGR ASK What if same names?
        std::cerr << "Schema " + name << std::endl;
        schemes.emplace(name, Schema{config, config_prefix + "." + name, disks});
    }

    constexpr auto default_schema_name = "default";
    if (schemes.find(default_schema_name) == schemes.end()) {
        schemes.emplace(default_schema_name, Schema(Schema::Volumes{Schema::Volume::Disks{disks[default_disk_name]}}));
    }

    std::cerr << schemes.size() << " schemes loaded" << std::endl; ///@TODO_IGR ASK logs?
}

const Schema & SchemaSelector::operator[](const String & name) const {
    auto it = schemes.find(name);
    if (it == schemes.end()) {
        throw Exception("Unknown schema " + name, ErrorCodes::UNKNOWN_ELEMENT_IN_CONFIG); ///@TODO_IGR Choose error code
    }
    return it->second;
}

}
