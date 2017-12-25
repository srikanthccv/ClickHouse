#pragma once

#include <Core/NamesAndTypes.h>


namespace DB
{

class Cluster;
class Context;

/// Find the names and types of the table columns on any server in the cluster.
/// Used to implement the `remote` table function and others.
NamesAndTypesList getStructureOfRemoteTable(
    const Cluster & cluster,
    const std::string & database,
    const std::string & table,
    const Context & context);

}
