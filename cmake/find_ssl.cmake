option (ENABLE_SSL "Enable ssl" ON)

if (ENABLE_SSL)

if(NOT ARCH_32)
    option(USE_INTERNAL_SSL_LIBRARY "Set to FALSE to use system *ssl library instead of bundled" ${NOT_UNBUNDLED})
endif()

if(NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/ssl/CMakeLists.txt")
    if(USE_INTERNAL_SSL_LIBRARY)
        message(WARNING "submodule contrib/ssl is missing. to fix try run: \n git submodule update --init --recursive")
    endif()
    set(USE_INTERNAL_SSL_LIBRARY 0)
    set(MISSING_INTERNAL_SSL_LIBRARY 1)
endif()

set (OPENSSL_USE_STATIC_LIBS ${USE_STATIC_LIBRARIES})

if (NOT USE_INTERNAL_SSL_LIBRARY)
    if (APPLE)
        set (OPENSSL_ROOT_DIR "/usr/local/opt/openssl" CACHE INTERNAL "")
        # https://rt.openssl.org/Ticket/Display.html?user=guest&pass=guest&id=2232
        if (USE_STATIC_LIBRARIES)
            message(WARNING "Disable USE_STATIC_LIBRARIES if you have linking problems with OpenSSL on MacOS")
        endif ()
    endif ()
    find_package (OpenSSL)

    if (NOT OPENSSL_FOUND)
        # Try to find manually.
        set (OPENSSL_INCLUDE_PATHS "/usr/local/opt/openssl/include")
        set (OPENSSL_PATHS "/usr/local/opt/openssl/lib")
        find_path (OPENSSL_INCLUDE_DIR NAMES openssl/ssl.h PATHS ${OPENSSL_INCLUDE_PATHS})
        find_library (OPENSSL_SSL_LIBRARY ssl PATHS ${OPENSSL_PATHS})
        find_library (OPENSSL_CRYPTO_LIBRARY crypto PATHS ${OPENSSL_PATHS})
        if (OPENSSL_SSL_LIBRARY AND OPENSSL_CRYPTO_LIBRARY AND OPENSSL_INCLUDE_DIR)
            set (OPENSSL_LIBRARIES ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
            set (OPENSSL_FOUND 1)
        endif ()
    endif ()
endif ()

if (NOT OPENSSL_FOUND AND NOT MISSING_INTERNAL_SSL_LIBRARY)
    set (USE_INTERNAL_SSL_LIBRARY 1)
    set (OPENSSL_ROOT_DIR "${ClickHouse_SOURCE_DIR}/contrib/ssl")
    set (OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include")
    if (NOT USE_STATIC_LIBRARIES AND TARGET crypto-shared AND TARGET ssl-shared)
        set (OPENSSL_CRYPTO_LIBRARY crypto-shared)
        set (OPENSSL_SSL_LIBRARY ssl-shared)
    else ()
        set (OPENSSL_CRYPTO_LIBRARY crypto)
        set (OPENSSL_SSL_LIBRARY ssl)
    endif ()
    set (OPENSSL_LIBRARIES ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
    set (OPENSSL_FOUND 1)
endif ()

if(OPENSSL_FOUND)
    # we need keep OPENSSL_FOUND for many libs in contrib
    set(USE_SSL 1)
endif()

endif ()

message (STATUS "Using ssl=${USE_SSL}: ${OPENSSL_INCLUDE_DIR} : ${OPENSSL_LIBRARIES}")
