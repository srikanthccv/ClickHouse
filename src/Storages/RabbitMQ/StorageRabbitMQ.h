#pragma once

#include <Core/BackgroundSchedulePool.h>
#include <Storages/IStorage.h>
#include <Poco/Semaphore.h>
#include <ext/shared_ptr_helper.h>
#include <mutex>
#include <atomic>
#include <Storages/RabbitMQ/Buffer_fwd.h>
#include <Storages/RabbitMQ/RabbitMQHandler.h>
#include <Storages/RabbitMQ/RabbitMQSettings.h>
#include <Storages/RabbitMQ/UVLoop.h>
#include <Common/thread_local_rng.h>
#include <amqpcpp/libuv.h>
#include <uv.h>
#include <random>


namespace DB
{

using ChannelPtr = std::shared_ptr<AMQP::TcpChannel>;

class StorageRabbitMQ final: public ext::shared_ptr_helper<StorageRabbitMQ>, public IStorage, WithContext
{
    friend struct ext::shared_ptr_helper<StorageRabbitMQ>;

public:
    std::string getName() const override { return "RabbitMQ"; }

    bool noPushingToViews() const override { return true; }

    void startup() override;
    void shutdown() override;

    /// Always return virtual columns in addition to required columns
    Pipe read(
        const Names & column_names,
        const StorageMetadataPtr & metadata_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        unsigned num_streams) override;

    BlockOutputStreamPtr write(
        const ASTPtr & query,
        const StorageMetadataPtr & metadata_snapshot,
        ContextPtr context) override;

    void pushReadBuffer(ConsumerBufferPtr buf);
    ConsumerBufferPtr popReadBuffer();
    ConsumerBufferPtr popReadBuffer(std::chrono::milliseconds timeout);

    ProducerBufferPtr createWriteBuffer();

    const String & getFormatName() const { return format_name; }
    NamesAndTypesList getVirtuals() const override;

    String getExchange() const { return exchange_name; }
    void unbindExchange();
    bool exchangeRemoved() { return exchange_removed.load(); }

    bool updateChannel(ChannelPtr & channel);
    void updateQueues(std::vector<String> & queues_) { queues_ = queues; }

protected:
    StorageRabbitMQ(
            const StorageID & table_id_,
            ContextPtr context_,
            const ColumnsDescription & columns_,
            std::unique_ptr<RabbitMQSettings> rabbitmq_settings_);

private:
    ContextPtr rabbitmq_context;
    std::unique_ptr<RabbitMQSettings> rabbitmq_settings;

    const String exchange_name;
    const String format_name;
    AMQP::ExchangeType exchange_type;
    Names routing_keys;
    char row_delimiter;
    const String schema_name;
    size_t num_consumers;
    size_t num_queues;
    String queue_base;
    const String deadletter_exchange;
    const bool persistent;

    bool hash_exchange;
    Poco::Logger * log;
    String address;
    std::pair<String, UInt16> parsed_address;
    std::pair<String, String> login_password;
    String vhost;

    UVLoop loop;
    std::shared_ptr<RabbitMQHandler> event_handler;
    std::unique_ptr<AMQP::TcpConnection> connection; /// Connection for all consumers

    size_t num_created_consumers = 0;
    Poco::Semaphore semaphore;
    std::mutex buffers_mutex;
    std::vector<ConsumerBufferPtr> buffers; /// available buffers for RabbitMQ consumers

    String unique_strbase; /// to make unique consumer channel id

    /// maximum number of messages in RabbitMQ queue (x-max-length). Also used
    /// to setup size of inner buffer for received messages
    uint32_t queue_size;
    String sharding_exchange, bridge_exchange, consumer_exchange;
    size_t consumer_id = 0; /// counter for consumer buffer, needed for channel id
    std::atomic<size_t> producer_id = 1; /// counter for producer buffer, needed for channel id
    std::atomic<bool> wait_confirm = true; /// needed to break waiting for confirmations for producer
    std::atomic<bool> exchange_removed = false, rabbit_is_ready = false;
    std::vector<String> queues;

    std::once_flag flag; /// remove exchange only once
    std::mutex task_mutex;
    BackgroundSchedulePool::TaskHolder streaming_task;
    BackgroundSchedulePool::TaskHolder looping_task;
    BackgroundSchedulePool::TaskHolder connection_task;

    std::atomic<bool> stream_cancelled{false};
    size_t read_attempts = 0;

    ConsumerBufferPtr createReadBuffer();

    /// Functions working in the background
    void streamingToViewsFunc();
    void loopingFunc();
    void connectionFunc();

    static Names parseRoutingKeys(String routing_key_list);
    static AMQP::ExchangeType defineExchangeType(String exchange_type_);
    static String getTableBasedName(String name, const StorageID & table_id);

    std::shared_ptr<Context> addSettings(ContextPtr context) const;
    size_t getMaxBlockSize() const;
    void deactivateTask(BackgroundSchedulePool::TaskHolder & task, bool wait, bool stop_loop);

    void initRabbitMQ();
    void initExchange(RabbitMQChannel & rabbit_channel);
    void bindExchange(RabbitMQChannel & rabbit_channel);
    void bindQueue(size_t queue_id, RabbitMQChannel & rabbit_channel);

    bool restoreConnection(bool reconnecting);
    bool streamToViews();
    bool checkDependencies(const StorageID & table_id);

    String getRandomName() const
    {
        std::uniform_int_distribution<int> distribution('a', 'z');
        String random_str(32, ' ');
        for (auto & c : random_str)
            c = distribution(thread_local_rng);
        return random_str;
    }
};

}
