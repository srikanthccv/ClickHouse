#include <Storages/Kafka/StorageKafka.h>
#include <Storages/Kafka/parseSyslogLevel.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/Context.h>
#include <Interpreters/InterpreterInsertQuery.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTCreateQuery.h>
#include <Parsers/ASTExpressionList.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTInsertQuery.h>
#include <Parsers/ASTLiteral.h>
#include <Processors/Executors/CompletedPipelineExecutor.h>
#include <QueryPipeline/QueryPipeline.h>
#include <QueryPipeline/Pipe.h>
#include <Storages/MessageQueueSink.h>
#include <Storages/Kafka/KafkaProducer.h>
#include <Storages/Kafka/KafkaSettings.h>
#include <Storages/Kafka/KafkaSource.h>
#include <Storages/StorageFactory.h>
#include <Storages/StorageMaterializedView.h>
#include <Storages/NamedCollectionsHelpers.h>
#include <base/getFQDNOrHostName.h>
#include <Common/logger_useful.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <librdkafka/rdkafka.h>
#include <Poco/Util/AbstractConfiguration.h>
#include <Common/Exception.h>
#include <Common/Macros.h>
#include <Common/formatReadable.h>
#include <Common/getNumberOfPhysicalCPUCores.h>
#include <Common/quoteString.h>
#include <Common/setThreadName.h>
#include <Formats/FormatFactory.h>

#include "Storages/ColumnDefault.h"
#include "config_version.h"

#include <Common/CurrentMetrics.h>
#include <Common/ProfileEvents.h>
#if USE_KRB5
#include <Access/KerberosInit.h>
#endif // USE_KRB5

namespace CurrentMetrics
{
    extern const Metric KafkaLibrdkafkaThreads;
    extern const Metric KafkaBackgroundReads;
    extern const Metric KafkaConsumersInUse;
    extern const Metric KafkaWrites;
}

namespace ProfileEvents
{
    extern const Event KafkaDirectReads;
    extern const Event KafkaBackgroundReads;
    extern const Event KafkaWrites;
}


namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
    extern const int LOGICAL_ERROR;
    extern const int BAD_ARGUMENTS;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int QUERY_NOT_ALLOWED;
}

struct StorageKafkaInterceptors
{
    static rd_kafka_resp_err_t rdKafkaOnThreadStart(rd_kafka_t *, rd_kafka_thread_type_t thread_type, const char *, void * ctx)
    {
        StorageKafka * self = reinterpret_cast<StorageKafka *>(ctx);
        CurrentMetrics::add(CurrentMetrics::KafkaLibrdkafkaThreads, 1);

        const auto & storage_id = self->getStorageID();
        const auto & table = storage_id.getTableName();

        switch (thread_type)
        {
            case RD_KAFKA_THREAD_MAIN:
                setThreadName(("rdk:m/" + table.substr(0, 9)).c_str());
                break;
            case RD_KAFKA_THREAD_BACKGROUND:
                setThreadName(("rdk:bg/" + table.substr(0, 8)).c_str());
                break;
            case RD_KAFKA_THREAD_BROKER:
                setThreadName(("rdk:b/" + table.substr(0, 9)).c_str());
                break;
        }

        /// Create ThreadStatus to track memory allocations from librdkafka threads.
        //
        /// And store them in a separate list (thread_statuses) to make sure that they will be destroyed,
        /// regardless how librdkafka calls the hooks.
        /// But this can trigger use-after-free if librdkafka will not destroy threads after rd_kafka_wait_destroyed()
        auto thread_status = std::make_shared<ThreadStatus>();
        std::lock_guard lock(self->thread_statuses_mutex);
        self->thread_statuses.emplace_back(std::move(thread_status));

        return RD_KAFKA_RESP_ERR_NO_ERROR;
    }
    static rd_kafka_resp_err_t rdKafkaOnThreadExit(rd_kafka_t *, rd_kafka_thread_type_t, const char *, void * ctx)
    {
        StorageKafka * self = reinterpret_cast<StorageKafka *>(ctx);
        CurrentMetrics::sub(CurrentMetrics::KafkaLibrdkafkaThreads, 1);

        std::lock_guard lock(self->thread_statuses_mutex);
        const auto it = std::find_if(self->thread_statuses.begin(), self->thread_statuses.end(), [](const auto & thread_status_ptr)
        {
            return thread_status_ptr.get() == current_thread;
        });
        if (it == self->thread_statuses.end())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "No thread status for this librdkafka thread.");

        self->thread_statuses.erase(it);

        return RD_KAFKA_RESP_ERR_NO_ERROR;
    }

    static rd_kafka_resp_err_t rdKafkaOnNew(rd_kafka_t * rk, const rd_kafka_conf_t *, void * ctx, char * /*errstr*/, size_t /*errstr_size*/)
    {
        StorageKafka * self = reinterpret_cast<StorageKafka *>(ctx);
        rd_kafka_resp_err_t status;

        status = rd_kafka_interceptor_add_on_thread_start(rk, "init-thread", rdKafkaOnThreadStart, ctx);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
        {
            LOG_ERROR(self->log, "Cannot set on thread start interceptor due to {} error", status);
            return status;
        }

        status = rd_kafka_interceptor_add_on_thread_exit(rk, "exit-thread", rdKafkaOnThreadExit, ctx);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
            LOG_ERROR(self->log, "Cannot set on thread exit interceptor due to {} error", status);

        return status;
    }

    static rd_kafka_resp_err_t rdKafkaOnConfDup(rd_kafka_conf_t * new_conf, const rd_kafka_conf_t * /*old_conf*/, size_t /*filter_cnt*/, const char ** /*filter*/, void * ctx)
    {
        StorageKafka * self = reinterpret_cast<StorageKafka *>(ctx);
        rd_kafka_resp_err_t status;

        // cppkafka copies configuration multiple times
        status = rd_kafka_conf_interceptor_add_on_conf_dup(new_conf, "init", rdKafkaOnConfDup, ctx);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
        {
            LOG_ERROR(self->log, "Cannot set on conf dup interceptor due to {} error", status);
            return status;
        }

        status = rd_kafka_conf_interceptor_add_on_new(new_conf, "init", rdKafkaOnNew, ctx);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
            LOG_ERROR(self->log, "Cannot set on conf new interceptor due to {} error", status);

        return status;
    }
};

namespace
{
    const auto RESCHEDULE_MS = 500;
    const auto CLEANUP_TIMEOUT_MS = 3000;
    const auto MAX_THREAD_WORK_DURATION_MS = 60000;  // once per minute leave do reschedule (we can't lock threads in pool forever)

    const String CONFIG_KAFKA_TAG = "kafka";
    const String CONFIG_KAFKA_TOPIC_TAG = "kafka_topic";
    const String CONFIG_NAME_TAG = "name";

    /// Read server configuration into cppkafka configuration, used by global configuration and by legacy per-topic configuration
    void loadFromConfig(cppkafka::Configuration & kafka_config, const Poco::Util::AbstractConfiguration & config, const String & config_prefix)
    {
        /// Read all tags one level below <kafka>
        Poco::Util::AbstractConfiguration::Keys tags;
        config.keys(config_prefix, tags);

        for (const auto & tag : tags)
        {
            if (tag.starts_with(CONFIG_KAFKA_TOPIC_TAG)) /// multiple occurrences given as "kafka_topic", "kafka_topic[1]", etc.
                continue; /// used by new per-topic configuration, ignore

            const String setting_path = config_prefix + "." + tag;
            const String setting_value = config.getString(setting_path);

            /// "log_level" has valid underscore, the remaining librdkafka setting use dot.separated.format which isn't acceptable for XML.
            /// See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
            const String setting_name_in_kafka_config = (tag == "log_level") ? tag : boost::replace_all_copy(tag, "_", ".");
            kafka_config.set(setting_name_in_kafka_config, setting_value);
        }
    }

    /// Read server configuration into cppkafa configuration, used by new per-topic configuration
    void loadTopicConfig(cppkafka::Configuration & kafka_config, const Poco::Util::AbstractConfiguration & config, const String & config_prefix, const String & topic)
    {
        /// Read all tags one level below <kafka>
        Poco::Util::AbstractConfiguration::Keys tags;
        config.keys(config_prefix, tags);

        for (const auto & tag : tags)
        {
            /// Only consider tag <kafka_topic>. Multiple occurrences given as "kafka_topic", "kafka_topic[1]", etc.
            if (!tag.starts_with(CONFIG_KAFKA_TOPIC_TAG))
                continue;

            /// Read topic name between <name>...</name>
            const String kafka_topic_path = config_prefix + "." + tag;
            const String kafpa_topic_name_path = kafka_topic_path + "." + CONFIG_NAME_TAG;

            const String topic_name = config.getString(kafpa_topic_name_path);
            if (topic_name == topic)
            {
                /// Found it! Now read the per-topic configuration into cppkafka.
                Poco::Util::AbstractConfiguration::Keys inner_tags;
                config.keys(kafka_topic_path, inner_tags);
                for (const auto & inner_tag : inner_tags)
                {
                    if (inner_tag == CONFIG_NAME_TAG)
                        continue; // ignore <name>

                    /// "log_level" has valid underscore, the remaining librdkafka setting use dot.separated.format which isn't acceptable for XML.
                    /// See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
                    const String setting_path = kafka_topic_path + "." + inner_tag;
                    const String setting_value = config.getString(setting_path);

                    const String setting_name_in_kafka_config = (inner_tag == "log_level") ? inner_tag : boost::replace_all_copy(inner_tag, "_", ".");
                    kafka_config.set(setting_name_in_kafka_config, setting_value);
                }
            }
        }
    }
}

StorageKafka::StorageKafka(
    const StorageID & table_id_, ContextPtr context_,
    const ColumnsDescription & columns_, std::unique_ptr<KafkaSettings> kafka_settings_,
    const String & collection_name_)
    : IStorage(table_id_)
    , WithContext(context_->getGlobalContext())
    , kafka_settings(std::move(kafka_settings_))
    , macros_info{.table_id = table_id_}
    , topics(parseTopics(getContext()->getMacros()->expand(kafka_settings->kafka_topic_list.value, macros_info)))
    , brokers(getContext()->getMacros()->expand(kafka_settings->kafka_broker_list.value, macros_info))
    , group(getContext()->getMacros()->expand(kafka_settings->kafka_group_name.value, macros_info))
    , client_id(
          kafka_settings->kafka_client_id.value.empty() ? getDefaultClientId(table_id_)
                                                        : getContext()->getMacros()->expand(kafka_settings->kafka_client_id.value, macros_info))
    , format_name(getContext()->getMacros()->expand(kafka_settings->kafka_format.value))
    , max_rows_per_message(kafka_settings->kafka_max_rows_per_message.value)
    , schema_name(getContext()->getMacros()->expand(kafka_settings->kafka_schema.value, macros_info))
    , num_consumers(kafka_settings->kafka_num_consumers.value)
    , log(&Poco::Logger::get("StorageKafka (" + table_id_.table_name + ")"))
    , semaphore(0, static_cast<int>(num_consumers))
    , intermediate_commit(kafka_settings->kafka_commit_every_batch.value)
    , settings_adjustments(createSettingsAdjustments())
    , thread_per_consumer(kafka_settings->kafka_thread_per_consumer.value)
    , collection_name(collection_name_)
{
    if (kafka_settings->kafka_handle_error_mode == HandleKafkaErrorMode::STREAM)
    {
        kafka_settings->input_format_allow_errors_num = 0;
        kafka_settings->input_format_allow_errors_ratio = 0;
    }
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    setInMemoryMetadata(storage_metadata);
    auto task_count = thread_per_consumer ? num_consumers : 1;
    for (size_t i = 0; i < task_count; ++i)
    {
        auto task = getContext()->getMessageBrokerSchedulePool().createTask(log->name(), [this, i]{ threadFunc(i); });
        task->deactivate();
        tasks.emplace_back(std::make_shared<TaskContext>(std::move(task)));
    }
}

SettingsChanges StorageKafka::createSettingsAdjustments()
{
    SettingsChanges result;
    // Needed for backward compatibility
    if (!kafka_settings->input_format_skip_unknown_fields.changed)
    {
        // Always skip unknown fields regardless of the context (JSON or TSKV)
        kafka_settings->input_format_skip_unknown_fields = true;
    }

    if (!kafka_settings->input_format_allow_errors_ratio.changed)
    {
        kafka_settings->input_format_allow_errors_ratio = 0.;
    }

    if (!kafka_settings->input_format_allow_errors_num.changed)
    {
        kafka_settings->input_format_allow_errors_num = kafka_settings->kafka_skip_broken_messages.value;
    }

    if (!schema_name.empty())
        result.emplace_back("format_schema", schema_name);

    for (const auto & setting : *kafka_settings)
    {
        const auto & name = setting.getName();
        if (name.find("kafka_") == std::string::npos)
            result.emplace_back(name, setting.getValue());
    }
    return result;
}

Names StorageKafka::parseTopics(String topic_list)
{
    Names result;
    boost::split(result,topic_list,[](char c){ return c == ','; });
    for (String & topic : result)
    {
        boost::trim(topic);
    }
    return result;
}

String StorageKafka::getDefaultClientId(const StorageID & table_id_)
{
    return fmt::format("{}-{}-{}-{}", VERSION_NAME, getFQDNOrHostName(), table_id_.database_name, table_id_.table_name);
}


Pipe StorageKafka::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & /* query_info */,
    ContextPtr local_context,
    QueryProcessingStage::Enum /* processed_stage */,
    size_t /* max_block_size */,
    size_t /* num_streams */)
{
    if (num_created_consumers == 0)
        return {};

    if (!local_context->getSettingsRef().stream_like_engine_allow_direct_select)
        throw Exception(ErrorCodes::QUERY_NOT_ALLOWED,
                        "Direct select is not allowed. To enable use setting `stream_like_engine_allow_direct_select`");

    if (mv_attached)
        throw Exception(ErrorCodes::QUERY_NOT_ALLOWED, "Cannot read from StorageKafka with attached materialized views");

    ProfileEvents::increment(ProfileEvents::KafkaDirectReads);

    /// Always use all consumers at once, otherwise SELECT may not read messages from all partitions.
    Pipes pipes;
    pipes.reserve(num_created_consumers);
    auto modified_context = Context::createCopy(local_context);
    modified_context->applySettingsChanges(settings_adjustments);

    // Claim as many consumers as requested, but don't block
    for (size_t i = 0; i < num_created_consumers; ++i)
    {
        /// Use block size of 1, otherwise LIMIT won't work properly as it will buffer excess messages in the last block
        /// TODO: probably that leads to awful performance.
        /// FIXME: seems that doesn't help with extra reading and committing unprocessed messages.
        pipes.emplace_back(std::make_shared<KafkaSource>(*this, storage_snapshot, modified_context, column_names, log, 1, kafka_settings->kafka_commit_on_select));
    }

    LOG_DEBUG(log, "Starting reading {} streams", pipes.size());
    return Pipe::unitePipes(std::move(pipes));
}


SinkToStoragePtr StorageKafka::write(const ASTPtr &, const StorageMetadataPtr & metadata_snapshot, ContextPtr local_context, bool /*async_insert*/)
{
    auto modified_context = Context::createCopy(local_context);
    modified_context->applySettingsChanges(settings_adjustments);

    CurrentMetrics::Increment metric_increment{CurrentMetrics::KafkaWrites};
    ProfileEvents::increment(ProfileEvents::KafkaWrites);

    if (topics.size() > 1)
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Can't write to Kafka table with multiple topics!");

    cppkafka::Configuration conf;
    conf.set("metadata.broker.list", brokers);
    conf.set("client.id", client_id);
    conf.set("client.software.name", VERSION_NAME);
    conf.set("client.software.version", VERSION_DESCRIBE);
    // TODO: fill required settings
    updateConfiguration(conf);

    const Settings & settings = getContext()->getSettingsRef();
    size_t poll_timeout = settings.stream_poll_timeout_ms.totalMilliseconds();
    const auto & header = metadata_snapshot->getSampleBlockNonMaterialized();

    auto producer = std::make_unique<KafkaProducer>(
        std::make_shared<cppkafka::Producer>(conf), topics[0], std::chrono::milliseconds(poll_timeout), shutdown_called, header);

    size_t max_rows = max_rows_per_message;
    /// Need for backward compatibility.
    if (format_name == "Avro" && local_context->getSettingsRef().output_format_avro_rows_in_file.changed)
        max_rows = local_context->getSettingsRef().output_format_avro_rows_in_file.value;
    return std::make_shared<MessageQueueSink>(
        header, getFormatName(), max_rows, std::move(producer), getName(), modified_context);
}


void StorageKafka::startup()
{
    for (size_t i = 0; i < num_consumers; ++i)
    {
        try
        {
            auto consumer = createConsumer(i);
            pushConsumer(consumer);
            all_consumers.push_back(consumer);
            ++num_created_consumers;
        }
        catch (const cppkafka::Exception &)
        {
            tryLogCurrentException(log);
        }
    }

    // Start the reader thread
    for (auto & task : tasks)
    {
        task->holder->activateAndSchedule();
    }
}


void StorageKafka::shutdown()
{
    for (auto & task : tasks)
    {
        // Interrupt streaming thread
        task->stream_cancelled = true;

        LOG_TRACE(log, "Waiting for cleanup");
        task->holder->deactivate();
    }

    LOG_TRACE(log, "Closing consumers");
    for (size_t i = 0; i < num_created_consumers; ++i)
        auto consumer = popConsumer();
    LOG_TRACE(log, "Consumers closed");

    rd_kafka_wait_destroyed(CLEANUP_TIMEOUT_MS);
}


void StorageKafka::pushConsumer(KafkaConsumerPtr consumer)
{
    std::lock_guard lock(mutex);
    consumer->notInUse();
    consumers.push_back(consumer);
    semaphore.set();
    CurrentMetrics::sub(CurrentMetrics::KafkaConsumersInUse, 1);
}


KafkaConsumerPtr StorageKafka::popConsumer()
{
    return popConsumer(std::chrono::milliseconds::zero());
}


KafkaConsumerPtr StorageKafka::popConsumer(std::chrono::milliseconds timeout)
{
    // Wait for the first free buffer
    if (timeout == std::chrono::milliseconds::zero())
        semaphore.wait();
    else
    {
        if (!semaphore.tryWait(timeout.count()))
            return nullptr;
    }

    // Take the first available buffer from the list
    std::lock_guard lock(mutex);
    auto consumer = consumers.back();
    consumers.pop_back();
    CurrentMetrics::add(CurrentMetrics::KafkaConsumersInUse, 1);
    consumer->inUse();
    return consumer;
}


KafkaConsumerPtr StorageKafka::createConsumer(size_t consumer_number)
{
    cppkafka::Configuration conf;

    conf.set("metadata.broker.list", brokers);
    conf.set("group.id", group);
    if (num_consumers > 1)
    {
        conf.set("client.id", fmt::format("{}-{}", client_id, consumer_number));
    }
    else
    {
        conf.set("client.id", client_id);
    }
    conf.set("client.software.name", VERSION_NAME);
    conf.set("client.software.version", VERSION_DESCRIBE);
    conf.set("auto.offset.reset", "earliest");     // If no offset stored for this group, read all messages from the start

    // that allows to prevent fast draining of the librdkafka queue
    // during building of single insert block. Improves performance
    // significantly, but may lead to bigger memory consumption.
    size_t default_queued_min_messages = 100000; // we don't want to decrease the default
    conf.set("queued.min.messages", std::max(getMaxBlockSize(),default_queued_min_messages));

    /// a reference to the consumer is needed in statistic callback
    /// although the consumer does not exist when callback is being registered
    /// shared_ptr<weak_ptr<KafkaConsumer>> comes to the rescue
    auto consumer_weak_ptr_ptr = std::make_shared<KafkaConsumerWeakPtr>();
    updateConfiguration(conf, consumer_weak_ptr_ptr);

    // those settings should not be changed by users.
    conf.set("enable.auto.commit", "false");       // We manually commit offsets after a stream successfully finished
    conf.set("enable.auto.offset.store", "false"); // Update offset automatically - to commit them all at once.
    conf.set("enable.partition.eof", "false");     // Ignore EOF messages

    // Create a consumer and subscribe to topics
    auto consumer_impl = std::make_shared<cppkafka::Consumer>(conf);
    consumer_impl->set_destroy_flags(RD_KAFKA_DESTROY_F_NO_CONSUMER_CLOSE);

    KafkaConsumerPtr kafka_consumer_ptr;

    /// NOTE: we pass |stream_cancelled| by reference here, so the buffers should not outlive the storage.
    if (thread_per_consumer)
    {
        auto& stream_cancelled = tasks[consumer_number]->stream_cancelled;
        kafka_consumer_ptr = std::make_shared<KafkaConsumer>(consumer_impl, log, getPollMaxBatchSize(), getPollTimeoutMillisecond(), intermediate_commit, stream_cancelled, topics);
    }
    else
    {
        kafka_consumer_ptr = std::make_shared<KafkaConsumer>(consumer_impl, log, getPollMaxBatchSize(), getPollTimeoutMillisecond(), intermediate_commit, tasks.back()->stream_cancelled, topics);
    }
    *consumer_weak_ptr_ptr = kafka_consumer_ptr;
    return kafka_consumer_ptr;
}

size_t StorageKafka::getMaxBlockSize() const
{
    return kafka_settings->kafka_max_block_size.changed
        ? kafka_settings->kafka_max_block_size.value
        : (getContext()->getSettingsRef().max_insert_block_size.value / num_consumers);
}

size_t StorageKafka::getPollMaxBatchSize() const
{
    size_t batch_size = kafka_settings->kafka_poll_max_batch_size.changed
                        ? kafka_settings->kafka_poll_max_batch_size.value
                        : getContext()->getSettingsRef().max_block_size.value;

    return std::min(batch_size,getMaxBlockSize());
}

size_t StorageKafka::getPollTimeoutMillisecond() const
{
    return kafka_settings->kafka_poll_timeout_ms.changed
        ? kafka_settings->kafka_poll_timeout_ms.totalMilliseconds()
        : getContext()->getSettingsRef().stream_poll_timeout_ms.totalMilliseconds();
}

String StorageKafka::getConfigPrefix() const
{
    if (!collection_name.empty())
        return "named_collections." + collection_name + "." + CONFIG_KAFKA_TAG; /// Add one more level to separate librdkafka configuration.
    return CONFIG_KAFKA_TAG;
}

void StorageKafka::updateConfiguration(cppkafka::Configuration & kafka_config,
    std::shared_ptr<KafkaConsumerWeakPtr>  kafka_consumer_weak_ptr_ptr)
{
    // Update consumer configuration from the configuration. Example:
    //     <kafka>
    //         <retry_backoff_ms>250</retry_backoff_ms>
    //         <fetch_min_bytes>100000</fetch_min_bytes>
    //     </kafka>
    const auto & config = getContext()->getConfigRef();
    auto config_prefix = getConfigPrefix();
    if (config.has(config_prefix))
        loadFromConfig(kafka_config, config, config_prefix);

#if USE_KRB5
    if (kafka_config.has_property("sasl.kerberos.kinit.cmd"))
        LOG_WARNING(log, "sasl.kerberos.kinit.cmd configuration parameter is ignored.");

    kafka_config.set("sasl.kerberos.kinit.cmd","");
    kafka_config.set("sasl.kerberos.min.time.before.relogin","0");

    if (kafka_config.has_property("sasl.kerberos.keytab") && kafka_config.has_property("sasl.kerberos.principal"))
    {
        String keytab = kafka_config.get("sasl.kerberos.keytab");
        String principal = kafka_config.get("sasl.kerberos.principal");
        LOG_DEBUG(log, "Running KerberosInit");
        try
        {
            kerberosInit(keytab,principal);
        }
        catch (const Exception & e)
        {
            LOG_ERROR(log, "KerberosInit failure: {}", getExceptionMessage(e, false));
        }
        LOG_DEBUG(log, "Finished KerberosInit");
    }
#else // USE_KRB5
    if (kafka_config.has_property("sasl.kerberos.keytab") || kafka_config.has_property("sasl.kerberos.principal"))
        LOG_WARNING(log, "Ignoring Kerberos-related parameters because ClickHouse was built without krb5 library support.");
#endif // USE_KRB5

    // Update consumer topic-specific configuration (legacy syntax, retained for compatibility). Example with topic "football":
    //     <kafka_football>
    //         <retry_backoff_ms>250</retry_backoff_ms>
    //         <fetch_min_bytes>100000</fetch_min_bytes>
    //     </kafka_football>
    // The legacy syntax has the problem that periods in topic names (e.g. "sports.football") are not supported because the Poco
    // configuration framework hierarchy is based on periods as level separators. Besides that, per-topic tags at the same level
    // as <kafka> are ugly.
    for (const auto & topic : topics)
    {
        const auto topic_config_key = config_prefix + "_" + topic;
        if (config.has(topic_config_key))
            loadFromConfig(kafka_config, config, topic_config_key);
    }

    // Update consumer topic-specific configuration (new syntax). Example with topics "football" and "baseball":
    //     <kafka>
    //         <kafka_topic>
    //             <name>football</name>
    //             <retry_backoff_ms>250</retry_backoff_ms>
    //             <fetch_min_bytes>5000</fetch_min_bytes>
    //         </kafka_topic>
    //         <kafka_topic>
    //             <name>baseball</name>
    //             <retry_backoff_ms>300</retry_backoff_ms>
    //             <fetch_min_bytes>2000</fetch_min_bytes>
    //         </kafka_topic>
    //     </kafka>
    // Advantages: The period restriction no longer applies (e.g. <name>sports.football</name> will work), everything
    // Kafka-related is below <kafka>.
    for (const auto & topic : topics)
        if (config.has(config_prefix))
            loadTopicConfig(kafka_config, config, config_prefix, topic);

    // No need to add any prefix, messages can be distinguished
    kafka_config.set_log_callback([this](cppkafka::KafkaHandleBase &, int level, const std::string & facility, const std::string & message)
    {
        auto [poco_level, client_logs_level] = parseSyslogLevel(level);
        LOG_IMPL(log, client_logs_level, poco_level, "[rdk:{}] {}", facility, message);
    });

    if (kafka_consumer_weak_ptr_ptr)
    {
        if (!config.has(config_prefix + "." + "statistics_interval_ms"))
        {
            kafka_config.set("statistics.interval.ms", "3000"); // every 3 seconds by default. set to 0 to disable.
        }

        if (kafka_config.get("statistics.interval.ms") != "0")
        {
            kafka_config.set_stats_callback([kafka_consumer_weak_ptr_ptr](cppkafka::KafkaHandleBase &, const std::string & stat_json_string)
            {
                auto kafka_consumer_ptr = kafka_consumer_weak_ptr_ptr->lock();
                if (kafka_consumer_ptr)
                {
                    kafka_consumer_ptr->setRDKafkaStat(stat_json_string);
                }
            });
        }
    }

    // Configure interceptor to change thread name
    //
    // TODO: add interceptors support into the cppkafka.
    // XXX:  rdkafka uses pthread_set_name_np(), but glibc-compatibliity overrides it to noop.
    {
        // This should be safe, since we wait the rdkafka object anyway.
        void * self = static_cast<void *>(this);

        int status;

        status = rd_kafka_conf_interceptor_add_on_new(kafka_config.get_handle(),
            "init", StorageKafkaInterceptors::rdKafkaOnNew, self);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
            LOG_ERROR(log, "Cannot set new interceptor due to {} error", status);

        // cppkafka always copy the configuration
        status = rd_kafka_conf_interceptor_add_on_conf_dup(kafka_config.get_handle(),
            "init", StorageKafkaInterceptors::rdKafkaOnConfDup, self);
        if (status != RD_KAFKA_RESP_ERR_NO_ERROR)
            LOG_ERROR(log, "Cannot set dup conf interceptor due to {} error", status);
    }
}

bool StorageKafka::checkDependencies(const StorageID & table_id)
{
    // Check if all dependencies are attached
    auto view_ids = DatabaseCatalog::instance().getDependentViews(table_id);
    if (view_ids.empty())
        return true;

    // Check the dependencies are ready?
    for (const auto & view_id : view_ids)
    {
        auto view = DatabaseCatalog::instance().tryGetTable(view_id, getContext());
        if (!view)
            return false;

        // If it materialized view, check it's target table
        auto * materialized_view = dynamic_cast<StorageMaterializedView *>(view.get());
        if (materialized_view && !materialized_view->tryGetTargetTable())
            return false;

        // Check all its dependencies
        if (!checkDependencies(view_id))
            return false;
    }

    return true;
}

void StorageKafka::threadFunc(size_t idx)
{
    assert(idx < tasks.size());
    auto task = tasks[idx];
    try
    {
        auto table_id = getStorageID();
        // Check if at least one direct dependency is attached
        size_t num_views = DatabaseCatalog::instance().getDependentViews(table_id).size();
        if (num_views)
        {
            auto start_time = std::chrono::steady_clock::now();

            mv_attached.store(true);

            // Keep streaming as long as there are attached views and streaming is not cancelled
            while (!task->stream_cancelled && num_created_consumers > 0)
            {
                if (!checkDependencies(table_id))
                    break;

                LOG_DEBUG(log, "Started streaming to {} attached views", num_views);

                // Exit the loop & reschedule if some stream stalled
                auto some_stream_is_stalled = streamToViews();
                if (some_stream_is_stalled)
                {
                    LOG_TRACE(log, "Stream(s) stalled. Reschedule.");
                    break;
                }

                auto ts = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ts-start_time);
                if (duration.count() > MAX_THREAD_WORK_DURATION_MS)
                {
                    LOG_TRACE(log, "Thread work duration limit exceeded. Reschedule.");
                    break;
                }
            }
        }
    }
    catch (...)
    {
        // !!!
        // std::string getCurrentExceptionMessage(bool with_stacktrace, bool check_embedded_stacktrace /*= false*/, bool with_extra_info /*= true*/);

        auto last_exception = std::current_exception();
        tryLogException(last_exception, log, __PRETTY_FUNCTION__);

        auto exception_str = getExceptionMessage(last_exception, true /* with_stacktrace */);

        for (auto consumer_ptr_weak : all_consumers)
        {
            if (auto consumer_ptr = consumer_ptr_weak.lock())
            {
                consumer_ptr->setExceptionInfo(exception_str);
            }
        }



        // tryLogCurrentException(__PRETTY_FUNCTION__);


    }

    mv_attached.store(false);

    // Wait for attached views
    if (!task->stream_cancelled)
        task->holder->scheduleAfter(RESCHEDULE_MS);
}


bool StorageKafka::streamToViews()
{
    Stopwatch watch;

    auto table_id = getStorageID();
    auto table = DatabaseCatalog::instance().getTable(table_id, getContext());
    if (!table)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Engine table {} doesn't exist.", table_id.getNameForLogs());

    CurrentMetrics::Increment metric_increment{CurrentMetrics::KafkaBackgroundReads};
    ProfileEvents::increment(ProfileEvents::KafkaBackgroundReads);

    auto storage_snapshot = getStorageSnapshot(getInMemoryMetadataPtr(), getContext());

    // Create an INSERT query for streaming data
    auto insert = std::make_shared<ASTInsertQuery>();
    insert->table_id = table_id;

    size_t block_size = getMaxBlockSize();

    auto kafka_context = Context::createCopy(getContext());
    kafka_context->makeQueryContext();
    kafka_context->applySettingsChanges(settings_adjustments);

    // Create a stream for each consumer and join them in a union stream
    // Only insert into dependent views and expect that input blocks contain virtual columns
    InterpreterInsertQuery interpreter(insert, kafka_context, false, true, true);
    auto block_io = interpreter.execute();

    // Create a stream for each consumer and join them in a union stream
    std::vector<std::shared_ptr<KafkaSource>> sources;
    Pipes pipes;

    auto stream_count = thread_per_consumer ? 1 : num_created_consumers;
    sources.reserve(stream_count);
    pipes.reserve(stream_count);
    for (size_t i = 0; i < stream_count; ++i)
    {
        auto source = std::make_shared<KafkaSource>(*this, storage_snapshot, kafka_context, block_io.pipeline.getHeader().getNames(), log, block_size, false);
        sources.emplace_back(source);
        pipes.emplace_back(source);

        // Limit read batch to maximum block size to allow DDL
        StreamLocalLimits limits;

        Poco::Timespan max_execution_time = kafka_settings->kafka_flush_interval_ms.changed
                                          ? kafka_settings->kafka_flush_interval_ms
                                          : getContext()->getSettingsRef().stream_flush_interval_ms;

        source->setTimeLimit(max_execution_time);
    }

    auto pipe = Pipe::unitePipes(std::move(pipes));

    // We can't cancel during copyData, as it's not aware of commits and other kafka-related stuff.
    // It will be cancelled on underlying layer (kafka buffer)

    std::atomic_size_t rows = 0;
    {
        block_io.pipeline.complete(std::move(pipe));

        // we need to read all consumers in parallel (sequential read may lead to situation
        // when some of consumers are not used, and will break some Kafka consumer invariants)
        block_io.pipeline.setNumThreads(stream_count);

        block_io.pipeline.setProgressCallback([&](const Progress & progress) { rows += progress.read_rows.load(); });
        CompletedPipelineExecutor executor(block_io.pipeline);
        executor.execute();
    }

    bool some_stream_is_stalled = false;
    for (auto & source : sources)
    {
        some_stream_is_stalled = some_stream_is_stalled || source->isStalled();
        source->commit();
    }

    UInt64 milliseconds = watch.elapsedMilliseconds();
    LOG_DEBUG(log, "Pushing {} rows to {} took {} ms.",
        formatReadableQuantity(rows), table_id.getNameForLogs(), milliseconds);

    return some_stream_is_stalled;
}

void registerStorageKafka(StorageFactory & factory)
{
    auto creator_fn = [](const StorageFactory::Arguments & args)
    {
        ASTs & engine_args = args.engine_args;
        size_t args_count = engine_args.size();
        const bool has_settings = args.storage_def->settings;

        auto kafka_settings = std::make_unique<KafkaSettings>();
        String collection_name;
        if (auto named_collection = tryGetNamedCollectionWithOverrides(args.engine_args, args.getLocalContext()))
        {
            for (const auto & setting : kafka_settings->all())
            {
                const auto & setting_name = setting.getName();
                if (named_collection->has(setting_name))
                    kafka_settings->set(setting_name, named_collection->get<String>(setting_name));
            }
            collection_name = assert_cast<const ASTIdentifier *>(args.engine_args[0].get())->name();
        }

        if (has_settings)
        {
            kafka_settings->loadFromQuery(*args.storage_def);
        }

        // Check arguments and settings
        #define CHECK_KAFKA_STORAGE_ARGUMENT(ARG_NUM, PAR_NAME, EVAL)       \
            /* One of the four required arguments is not specified */       \
            if (args_count < (ARG_NUM) && (ARG_NUM) <= 4 &&                 \
                !kafka_settings->PAR_NAME.changed)                          \
            {                                                               \
                throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,\
                    "Required parameter '{}' "                              \
                    "for storage Kafka not specified",                      \
                    #PAR_NAME);                                             \
            }                                                               \
            if (args_count >= (ARG_NUM))                                    \
            {                                                               \
                /* The same argument is given in two places */              \
                if (has_settings &&                                         \
                    kafka_settings->PAR_NAME.changed)                       \
                {                                                           \
                    throw Exception(ErrorCodes::BAD_ARGUMENTS,              \
                        "The argument №{} of storage Kafka "                \
                        "and the parameter '{}' "                           \
                        "in SETTINGS cannot be specified at the same time", \
                        #ARG_NUM, #PAR_NAME);                               \
                }                                                           \
                /* move engine args to settings */                          \
                else                                                        \
                {                                                           \
                    if ((EVAL) == 1)                                        \
                    {                                                       \
                        engine_args[(ARG_NUM)-1] =                          \
                            evaluateConstantExpressionAsLiteral(            \
                                engine_args[(ARG_NUM)-1],                   \
                                args.getLocalContext());                    \
                    }                                                       \
                    if ((EVAL) == 2)                                        \
                    {                                                       \
                        engine_args[(ARG_NUM)-1] =                          \
                           evaluateConstantExpressionOrIdentifierAsLiteral( \
                                engine_args[(ARG_NUM)-1],                   \
                                args.getLocalContext());                    \
                    }                                                       \
                    kafka_settings->PAR_NAME =                              \
                        engine_args[(ARG_NUM)-1]->as<ASTLiteral &>().value; \
                }                                                           \
            }

        /** Arguments of engine is following:
          * - Kafka broker list
          * - List of topics
          * - Group ID (may be a constant expression with a string result)
          * - Message format (string)
          * - Row delimiter
          * - Schema (optional, if the format supports it)
          * - Number of consumers
          * - Max block size for background consumption
          * - Skip (at least) unreadable messages number
          * - Do intermediate commits when the batch consumed and handled
          */

        /* 0 = raw, 1 = evaluateConstantExpressionAsLiteral, 2=evaluateConstantExpressionOrIdentifierAsLiteral */
        /// In case of named collection we already validated the arguments.
        if (collection_name.empty())
        {
            CHECK_KAFKA_STORAGE_ARGUMENT(1, kafka_broker_list, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(2, kafka_topic_list, 1)
            CHECK_KAFKA_STORAGE_ARGUMENT(3, kafka_group_name, 2)
            CHECK_KAFKA_STORAGE_ARGUMENT(4, kafka_format, 2)
            CHECK_KAFKA_STORAGE_ARGUMENT(5, kafka_row_delimiter, 2)
            CHECK_KAFKA_STORAGE_ARGUMENT(6, kafka_schema, 2)
            CHECK_KAFKA_STORAGE_ARGUMENT(7, kafka_num_consumers, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(8, kafka_max_block_size, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(9, kafka_skip_broken_messages, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(10, kafka_commit_every_batch, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(11, kafka_client_id, 2)
            CHECK_KAFKA_STORAGE_ARGUMENT(12, kafka_poll_timeout_ms, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(13, kafka_flush_interval_ms, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(14, kafka_thread_per_consumer, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(15, kafka_handle_error_mode, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(16, kafka_commit_on_select, 0)
            CHECK_KAFKA_STORAGE_ARGUMENT(17, kafka_max_rows_per_message, 0)
        }

        #undef CHECK_KAFKA_STORAGE_ARGUMENT

        auto num_consumers = kafka_settings->kafka_num_consumers.value;
        auto max_consumers = std::max<uint32_t>(getNumberOfPhysicalCPUCores(), 16);

        if (!args.getLocalContext()->getSettingsRef().kafka_disable_num_consumers_limit && num_consumers > max_consumers)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of consumers can not be bigger than {}. "
                            "A single consumer can read any number of partitions. "
                            "Extra consumers are relatively expensive, "
                            "and using a lot of them can lead to high memory and CPU usage. "
                            "To achieve better performance "
                            "of getting data from Kafka, consider using a setting kafka_thread_per_consumer=1, "
                            "and ensure you have enough threads "
                            "in MessageBrokerSchedulePool (background_message_broker_schedule_pool_size). "
                            "See also https://clickhouse.com/docs/en/integrations/kafka#tuning-performance", max_consumers);
        }
        else if (num_consumers < 1)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Number of consumers can not be lower than 1");
        }

        if (kafka_settings->kafka_max_block_size.changed && kafka_settings->kafka_max_block_size.value < 1)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "kafka_max_block_size can not be lower than 1");
        }

        if (kafka_settings->kafka_poll_max_batch_size.changed && kafka_settings->kafka_poll_max_batch_size.value < 1)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "kafka_poll_max_batch_size can not be lower than 1");
        }
        NamesAndTypesList supported_columns;
        for (const auto & column : args.columns)
        {
            if (column.default_desc.kind == ColumnDefaultKind::Alias)
                supported_columns.emplace_back(column.name, column.type);
            if (column.default_desc.kind == ColumnDefaultKind::Default && !column.default_desc.expression)
                supported_columns.emplace_back(column.name, column.type);
        }
        // Kafka engine allows only ordinary columns without default expression or alias columns.
        if (args.columns.getAll() != supported_columns)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "KafkaEngine doesn't support DEFAULT/MATERIALIZED/EPHEMERAL expressions for columns. "
                                                       "See https://clickhouse.com/docs/en/engines/table-engines/integrations/kafka/#configuration");
        }

        return std::make_shared<StorageKafka>(args.table_id, args.getContext(), args.columns, std::move(kafka_settings), collection_name);
    };

    factory.registerStorage("Kafka", creator_fn, StorageFactory::StorageFeatures{ .supports_settings = true, });
}

NamesAndTypesList StorageKafka::getVirtuals() const
{
    auto result = NamesAndTypesList{
        {"_topic", std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>())},
        {"_key", std::make_shared<DataTypeString>()},
        {"_offset", std::make_shared<DataTypeUInt64>()},
        {"_partition", std::make_shared<DataTypeUInt64>()},
        {"_timestamp", std::make_shared<DataTypeNullable>(std::make_shared<DataTypeDateTime>())},
        {"_timestamp_ms", std::make_shared<DataTypeNullable>(std::make_shared<DataTypeDateTime64>(3))},
        {"_headers.name", std::make_shared<DataTypeArray>(std::make_shared<DataTypeString>())},
        {"_headers.value", std::make_shared<DataTypeArray>(std::make_shared<DataTypeString>())}};
    if (kafka_settings->kafka_handle_error_mode == HandleKafkaErrorMode::STREAM)
    {
        result.push_back({"_raw_message", std::make_shared<DataTypeString>()});
        result.push_back({"_error", std::make_shared<DataTypeString>()});
    }
    return result;
}

Names StorageKafka::getVirtualColumnNames() const
{
    auto result = Names {
        "_topic",
        "_key",
        "_offset",
        "_partition",
        "_timestamp",
        "_timestamp_ms",
        "_headers.name",
        "_headers.value",
    };
    if (kafka_settings->kafka_handle_error_mode == HandleKafkaErrorMode::STREAM)
    {
        result.push_back({"_raw_message"});
        result.push_back({"_error"});
    }
    return result;
}

}
