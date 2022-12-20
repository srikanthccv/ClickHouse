if (TARGET ch_contrib::rocksdb)
    set(USE_ROCKSDB 1)
endif()
if (TARGET ch_contrib::bzip2)
    set(USE_BZIP2 1)
endif()
if (TARGET ch_contrib::minizip)
    set(USE_MINIZIP 1)
endif()
if (TARGET ch_contrib::snappy)
    set(USE_SNAPPY 1)
endif()
if (TARGET ch_contrib::brotli)
    set(USE_BROTLI 1)
endif()
if (TARGET ch_contrib::hivemetastore)
    set(USE_HIVE 1)
endif()
if (TARGET ch_contrib::rdkafka)
    set(USE_RDKAFKA 1)
endif()
if (TARGET ch_rust::blake3)
    set(USE_BLAKE3 1)
endif()
if (TARGET ch_rust::skim)
    set(USE_SKIM 1)
endif()
if (TARGET OpenSSL::SSL)
    set(USE_SSL 1)
endif()
if (TARGET ch_contrib::ldap)
    set(USE_LDAP 1)
endif()
if (TARGET ch_contrib::grpc)
    set(USE_GRPC 1)
endif()
if (TARGET ch_contrib::hdfs)
    set(USE_HDFS 1)
endif()
if (TARGET ch_contrib::nuraft)
    set(USE_NURAFT 1)
endif()
if (TARGET ch_contrib::icu)
    set(USE_ICU 1)
endif()
if (TARGET ch_contrib::simdjson)
    set(USE_SIMDJSON 1)
endif()
if (TARGET ch_contrib::rapidjson)
    set(USE_RAPIDJSON 1)
endif()
if (TARGET ch_contrib::azure_sdk)
    set(USE_AZURE_BLOB_STORAGE 1)
endif()
if (TARGET ch_contrib::amqp_cpp)
    set(USE_AMQPCPP 1)
endif()
if (TARGET ch_contrib::nats_io)
    set(USE_NATSIO 1)
endif()
if (TARGET ch_contrib::cassandra)
    set(USE_CASSANDRA 1)
endif()
if (TARGET ch_contrib::base64)
    set(USE_BASE64 1)
endif()
if (TARGET ch_contrib::yaml_cpp)
    set(USE_YAML_CPP 1)
endif()
if (OS_LINUX)
    set(USE_FILELOG 1)
endif()
if (TARGET ch_contrib::sqlite)
    set(USE_SQLITE 1)
endif()
if (TARGET ch_contrib::libpqxx)
    set(USE_LIBPQXX 1)
endif()
if (TARGET ch_contrib::krb5)
    set(USE_KRB5 1)
endif()
if (TARGET ch_contrib::sentry)
    set(USE_SENTRY 1)
endif()
if (TARGET ch_contrib::datasketches)
    set(USE_DATASKETCHES 1)
endif()
if (TARGET ch_contrib::aws_s3)
    set(USE_AWS_S3 1)
endif()
if (TARGET ch_contrib::mariadbclient) # ch::mysqlxx
    set(USE_MYSQL 1)
endif()
if (ENABLE_NLP)
    set(USE_NLP 1)
endif()
if (TARGET ch_contrib::llvm)
    set(USE_EMBEDDED_COMPILER 1)
endif()
if (TARGET ch_contrib::unixodbc)
    set(USE_ODBC 1)
endif()
if (TARGET ch_contrib::replxx)
    set(USE_REPLXX 1)
endif()
if (TARGET ch_contrib::cpuid)
    set(USE_CPUID 1)
endif()
if (TARGET ch_contrib::jemalloc)
    set(USE_JEMALLOC 1)
endif()
if (TARGET ch_contrib::h3)
    set(USE_H3 1)
endif()
if (TARGET ch_contrib::s2)
    set(USE_S2_GEOMETRY 1)
endif()
if (TARGET ch_contrib::fastops)
    set(USE_FASTOPS 1)
endif()
if (TARGET ch_contrib::vectorscan)
    set(USE_VECTORSCAN 1)
endif()
if (TARGET ch_contrib::avrocpp)
    set(USE_AVRO 1)
endif()
if (TARGET ch_contrib::parquet)
    set(USE_PARQUET 1)
    set(USE_ARROW 1)
    set(USE_ORC 1)
endif()
if (TARGET ch_contrib::protobuf)
    set(USE_PROTOBUF 1)
endif()
if (TARGET ch_contrib::msgpack)
    set(USE_MSGPACK 1)
endif()
if (TARGET ch_contrib::capnp)
    set(USE_CAPNP 1)
endif()
if (NOT (ENABLE_OPENSSL OR ENABLE_OPENSSL_DYNAMIC))
    set(USE_BORINGSSL 1)
endif ()
if (ENABLE_OPENSSL)
    set(USE_OPENSSL_INTREE 1)
endif ()
