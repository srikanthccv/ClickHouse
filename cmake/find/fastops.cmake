if(NOT ARCH_ARM AND NOT OS_FREEBSD AND NOT OS_DARWIN)
    option(ENABLE_FASTOPS "Enable fast vectorized mathematical functions library by Mikhail Parakhin" ${ENABLE_LIBRARIES})
endif()

if(ENABLE_FASTOPS)
    if(NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/fastops/fastops/fastops.h")
        message(WARNING "submodule contrib/fastops is missing. to fix try run: \n git submodule update --init --recursive")
        set(MISSING_INTERNAL_FASTOPS_LIBRARY 1)
    endif()
    if(NOT MISSING_INTERNAL_FASTOPS_LIBRARY)
        set(USE_FASTOPS 1)
        set(FASTOPS_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/fastops/)
        set(FASTOPS_LIBRARY fastops)
    endif()
else()
    set(USE_FASTOPS 0)
endif()

message(STATUS "Using fastops=${USE_FASTOPS}: ${FASTOPS_INCLUDE_DIR} : ${FASTOPS_LIBRARY}")
