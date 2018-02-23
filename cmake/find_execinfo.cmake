if (ARCH_FREEBSD)
    find_library (EXECINFO_LIBRARY execinfo)
    message (STATUS "Using execinfo: ${EXECINFO_LIBRARY}")
else ()
    set (EXECINFO_LIBRARY "")
endif ()
