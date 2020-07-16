OPTION(ENABLE_CYRUS_SASL "Enable cyrus-sasl" ${ENABLE_LIBRARIES})
if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/cyrus-sasl/README")
    message (WARNING "submodule contrib/cyrus-sasl is missing. to fix try run: \n git submodule update --init --recursive")
    set (ENABLE_CYRUS_SASL 0)
endif ()

if (ENABLE_CYRUS_SASL)

    set (USE_CYRUS_SASL 1)
    set (CYRUS_SASL_LIBRARY sasl2)

    set (CYRUS_SASL_INCLUDE_DIR "${ClickHouse_SOURCE_DIR}/contrib/cyrus-sasl/include")


endif ()

message (STATUS "Using krb5=${USE_KRB5}: ${CYRUS_SASL_INCLUDE_DIR} : ${CYRUS_SASL_LIBRARY}")
