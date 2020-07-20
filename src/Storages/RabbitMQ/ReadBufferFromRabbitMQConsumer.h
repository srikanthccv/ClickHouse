#pragma once

#include <Core/Names.h>
#include <Core/Types.h>
#include <IO/ReadBuffer.h>
#include <amqpcpp.h>
#include <Storages/RabbitMQ/RabbitMQHandler.h>
#include <Common/ConcurrentBoundedQueue.h>

namespace Poco
{
    class Logger;
}

namespace DB
{

using ChannelPtr = std::shared_ptr<AMQP::TcpChannel>;
using HandlerPtr = std::shared_ptr<RabbitMQHandler>;

class ReadBufferFromRabbitMQConsumer : public ReadBuffer
{

public:
    ReadBufferFromRabbitMQConsumer(
            ChannelPtr consumer_channel_,
            HandlerPtr event_handler_,
            const String & exchange_name_,
            const AMQP::ExchangeType & exchange_type_,
            const Names & routing_keys_,
            size_t channel_id_,
            Poco::Logger * log_,
            char row_delimiter_,
            bool hash_exchange_,
            size_t num_queues_,
            const String & local_exchange_,
            const std::atomic<bool> & stopped_);

    ~ReadBufferFromRabbitMQConsumer() override;

    struct MessageData
    {
        UInt64 delivery_tag;
        String message;
        bool redelivered;
    };

    void allowNext() { allowed = true; } // Allow to read next message.
    void checkSubscription();

    auto getExchange() const { return exchange_name; }
    auto getConsumerTag() const { return consumer_tag; }
    auto getDeliveryTag() const { return current.delivery_tag; }
    auto getRedelivered() const { return current.redelivered; }

private:
    ChannelPtr consumer_channel;
    HandlerPtr event_handler;

    const String exchange_name;
    const AMQP::ExchangeType exchange_type;
    const Names routing_keys;
    const size_t channel_id;
    const bool hash_exchange;
    const size_t num_queues;

    const String local_exchange;
    const String local_default_exchange;
    const String local_hash_exchange;

    Poco::Logger * log;
    char row_delimiter;
    bool allowed = true;
    const std::atomic<bool> & stopped;

    String default_local_exchange;
    bool local_exchange_declared = false, local_hash_exchange_declared = false;

    std::atomic<bool> consumer_error = false;
    std::atomic<size_t> count_subscribed = 0, wait_subscribed;

    String consumer_tag;
    ConcurrentBoundedQueue<MessageData> received;
    MessageData current;
    std::vector<String> queues;
    std::unordered_map<String, bool> subscribed_queue;

    bool nextImpl() override;

    void initExchange();
    void initQueueBindings(const size_t queue_id);
    void subscribe(const String & queue_name);
    void iterateEventLoop();

};
}
