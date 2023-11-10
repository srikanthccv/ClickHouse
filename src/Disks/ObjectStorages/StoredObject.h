#pragma once

#include <base/types.h>

#include <Disks/ObjectStorages/IObjectStorage_fwd.h>

#include <functional>
#include <string>


namespace DB
{

/// Object metadata: path, size, path_key_for_cache.
struct StoredObject
{
    String remote_path; /// abs path
    String local_path; /// or equivalent "metadata_path"

    uint64_t bytes_size = 0;

    StoredObject() = default;

    explicit StoredObject(String remote_path_)
        : remote_path(std::move(remote_path_))
    {}

    StoredObject(
        String remote_path_,
        uint64_t bytes_size_)
        : remote_path(std::move(remote_path_))
        , bytes_size(bytes_size_)
    {}

    StoredObject(
        String remote_path_,
        uint64_t bytes_size_,
        String local_path_)
        : remote_path(std::move(remote_path_))
        , local_path(std::move(local_path_))
        , bytes_size(bytes_size_)
    {}
};

using StoredObjects = std::vector<StoredObject>;

size_t getTotalSize(const StoredObjects & objects);

}
