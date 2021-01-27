#pragma once

#include <common/logger_useful.h>
#include "PostgreSQLConnection.h"
#include "PostgreSQLReplicaConsumer.h"
#include <Interpreters/Context.h>
#include "pqxx/pqxx"


/* Implementation of logical streaming replication protocol: https://www.postgresql.org/docs/10/protocol-logical-replication.html.
 */

namespace DB
{

class PostgreSQLReplicationHandler
{
public:
    friend class PGReplicaLSN;
    PostgreSQLReplicationHandler(
            const std::string & database_name_,
            const std::string & table_name_,
            const std::string & conn_str_,
            const std::string & replication_slot_name_,
            const std::string & publication_name_);

    void startup(StoragePtr storage_, std::shared_ptr<Context> context_);
    void shutdown();
    void checkAndDropReplicationSlot();

private:
    using NontransactionPtr = std::shared_ptr<pqxx::nontransaction>;

    bool isPublicationExist();
    void createPublication();

    bool isReplicationSlotExist(NontransactionPtr ntx, std::string & slot_name);
    void createTempReplicationSlot(NontransactionPtr ntx, LSNPosition & start_lsn, std::string & snapshot_name);
    void createReplicationSlot(NontransactionPtr ntx);
    void dropReplicationSlot(NontransactionPtr tx, std::string & slot_name, bool use_replication_api);

    void startReplication();
    void loadFromSnapshot(std::string & snapshot_name);
    Context createQueryContext();
    void getTableOutput(const Context & query_context);

    Poco::Logger * log;
    const std::string database_name, table_name;

    std::string replication_slot, publication_name;
    std::string temp_replication_slot;

    PostgreSQLConnectionPtr connection;
    PostgreSQLConnectionPtr replication_connection;
    std::shared_ptr<pqxx::work> tx;

    std::shared_ptr<PostgreSQLReplicaConsumer> consumer;
    std::shared_ptr<Context> context;
    StoragePtr helper_table;
    //LSNPosition start_lsn, final_lsn;
};


}

