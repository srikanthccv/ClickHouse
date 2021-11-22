option(ENABLE_THRIFT "Enable Thrift" ${ENABLE_LIBRARIES})

if (NOT ENABLE_THRIFT)
    message (STATUS "thrift disabled")
    set(USE_INTERNAL_THRIFT_LIBRARY 0)
    return()
endif()

option(USE_INTERNAL_THRIFT_LIBRARY "Set to FALSE to use system thrift library instead of bundled" ${NOT_UNBUNDLED})
if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/thrift")
    if (USE_INTERNAL_THRIFT_LIBRARY)
        message (WARNING "submodule contrib is missing. to fix try run: \n git submodule update --init --recursive")
        set(USE_INTERNAL_THRIFT_LIBRARY 0)
    endif ()
endif()

if (USE_INTERNAL_THRIFT_LIBRARY)
    if (MAKE_STATIC_LIBRARIES)
        set(THRIFT_LIBRARY thrift_static)
    else()
        set(THRIFT_LIBRARY thrift)
    endif()
    set (THRIFT_INCLUDE_DIR "${ClickHouse_SOURCE_DIR}/contrib/thrift/lib/cpp/src")
    set(USE_THRIFT 1)
else()
    find_library(THRIFT_LIBRARY thrift)
    if (NOT THRIFT_LIBRARY)
        set(USE_THRIFT 0)
    else()
        set(USE_THRIFT 1)
    endif()
endif ()

message (STATUS "Using_THRIFT=${USE_THRIFT}: ${THRIFT_INCLUDE_DIR} : ${THRIFT_LIBRARY}")
