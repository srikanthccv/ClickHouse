if (HAVE_SSSE3)
    set (HYPERSCAN_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/hyperscan/src)
    set (HYPERSCAN_LIBRARY hs)
    set (USE_HYPERSCAN 1)
    set (USE_INTERNAL_HYPERSCAN_LIBRARY 1)
    message (STATUS "Using hyperscan: ${HYPERSCAN_INCLUDE_DIR} " : ${HYPERSCAN_LIBRARY})
endif()
