option(ENABLE_HIVE "Enable Hive" ${ENABLE_LIBRARIES})

if (NOT ENABLE_HIVE)
    message("Hive disabled")
    return()
endif()

if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/hive-metastore")
    message(WARNING "submodule contrib/hive-metastore is missing. to fix try run: \n git submodule update --init")
    set(USE_HIVE 0)
elseif (NOT USE_THRIFT)
    message(WARNING "Thrift is not found, which is needed by Hive")
    set(USE_HIVE 0)
elseif (NOT USE_ORC OR NOT USE_ARROW OR NOT USE_PARQUET)
    message(WARNING "ORC/Arrow/Parquet is not found, which are needed by Hive")
    set(USE_HIVE 0)
else()
    set(USE_HIVE 1)
    set(HIVE_METASTORE_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/hive-metastore)
    set(HIVE_METASTORE_LIBRARY hivemetastore)
endif()

message (STATUS "Using_Hive=${USE_HIVE}: ${HIVE_METASTORE_INCLUDE_DIR} : ${HIVE_METASTORE_LIBRARY}")
