if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/base64/LICENSE")
    set (MISSING_INTERNAL_BASE64_LIBRARY 1)
    message (WARNING "submodule contrib/base64 is missing. to fix try run: \n git submodule update --init --recursive")
endif ()

if (NOT MISSING_INTERNAL_BASE64_LIBRARY)
    option (ENABLE_BASE64 "Enable base64" ${ENABLE_LIBRARIES})
endif ()

if (ENABLE_BASE64)
    if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/base64")
        message (WARNING "submodule contrib/base64 is missing. to fix try run: \n git submodule update --init --recursive")
    else()
        set (BASE64_LIBRARY base64)
        set (USE_BASE64 1)
    endif()
endif ()
