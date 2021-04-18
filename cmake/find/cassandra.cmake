if (OS_DARWIN AND COMPILER_GCC)
    # Cassandra requires libuv which cannot be built with GCC in macOS due to a bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93082
    set (ENABLE_CASSANDRA OFF CACHE INTERNAL "")
endif()

option(ENABLE_CASSANDRA "Enable Cassandra" ${ENABLE_LIBRARIES})

if (NOT ENABLE_CASSANDRA)
    return()
endif()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

include(cmake/find/libuv.cmake)

if (MISSING_INTERNAL_LIBUV_LIBRARY)
    message (ERROR "submodule contrib/libuv is missing. to fix try run: \n git submodule update --init --recursive")
    message (${RECONFIGURE_MESSAGE_LEVEL} "Can't find internal libuv needed for Cassandra")
elseif (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/cassandra")
    message (ERROR "submodule contrib/cassandra is missing. to fix try run: \n git submodule update --init --recursive")
    message (${RECONFIGURE_MESSAGE_LEVEL} "Can't find internal Cassandra")
else()
    set (CASSANDRA_INCLUDE_DIR
            "${ClickHouse_SOURCE_DIR}/contrib/cassandra/include/")
    if (MAKE_STATIC_LIBRARIES)
        set (CASSANDRA_LIBRARY cassandra_static)
    else()
        set (CASSANDRA_LIBRARY cassandra)
    endif()

    set (USE_CASSANDRA 1)
    set (CASS_ROOT_DIR "${ClickHouse_SOURCE_DIR}/contrib/cassandra")
endif()

message (STATUS "Using cassandra=${USE_CASSANDRA}: ${CASSANDRA_INCLUDE_DIR} : ${CASSANDRA_LIBRARY}")

