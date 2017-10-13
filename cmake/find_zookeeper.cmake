option (USE_INTERNAL_ZOOKEEPER_LIBRARY "Set to FALSE to use system zookeeper library instead of bundled" ${NOT_UNBUNDLED})

if (NOT USE_INTERNAL_ZOOKEEPER_LIBRARY)
    find_library (ZOOKEEPER_LIBRARY zookeeper_mt)
    find_path (ZOOKEEPER_INCLUDE_DIR NAMES zookeeper/zookeeper.h PATHS ${ZOOKEEPER_INCLUDE_PATHS})
endif ()

if (ZOOKEEPER_LIBRARY AND ZOOKEEPER_INCLUDE_DIR)
    include_directories (${ZOOKEEPER_INCLUDE_DIR})
else ()
    set (USE_INTERNAL_ZOOKEEPER_LIBRARY 1)
    set(WANT_CPPUNIT 0 CACHE BOOL "")
    set (ZOOKEEPER_LIBRARY zookeeper)
endif ()

message (STATUS "Using zookeeper: ${ZOOKEEPER_INCLUDE_DIR} : ${ZOOKEEPER_LIBRARY}")


# how to make cickhouse branch of https://github.com/ClickHouse-Extras/zookeeper.git :
# clickhouse_misc
# clickhouse_706
