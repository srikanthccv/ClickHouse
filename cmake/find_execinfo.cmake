if (OS_FREEBSD)
    find_library (EXECINFO_LIBRARY execinfo)
    find_library (ELF_LIBRARY elf)
    set (EXECINFO_LIBRARIES ${EXECINFO_LIBRARY} ${ELF_LIBRARY})
    message (STATUS "Using execinfo: ${EXECINFO_LIBRARIES}")
else ()
    set (EXECINFO_LIBRARIES "")
endif ()
