option (USE_INTERNAL_XZ_LIBRARY "Set to OFF to use system xz (lzma) library instead of bundled" ON)

if(NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/xz/src/liblzma/api/lzma.h")
    if(USE_INTERNAL_XZ_LIBRARY)
        message(WARNING "submodule contrib/xz is missing. to fix try run: \n git submodule update --init")
        message (${RECONFIGURE_MESSAGE_LEVEL} "Can't find internal xz (lzma) library")
        set(USE_INTERNAL_XZ_LIBRARY 0)
    endif()
    set(MISSING_INTERNAL_XZ_LIBRARY 1)
endif()

if (NOT USE_INTERNAL_XZ_LIBRARY)
    find_library (XZ_LIBRARY lzma)
    find_path (XZ_INCLUDE_DIR NAMES lzma.h PATHS ${XZ_INCLUDE_PATHS})
    if (NOT XZ_LIBRARY OR NOT XZ_INCLUDE_DIR)
        message (${RECONFIGURE_MESSAGE_LEVEL} "Can't find system xz (lzma) library")
    endif ()
endif ()

if (XZ_LIBRARY AND XZ_INCLUDE_DIR)
elseif (NOT MISSING_INTERNAL_XZ_LIBRARY)
    set (USE_INTERNAL_XZ_LIBRARY 1)
    set (XZ_LIBRARY liblzma)
    set (XZ_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/xz/src/liblzma/api)
endif ()

message (STATUS "Using xz (lzma): ${XZ_INCLUDE_DIR} : ${XZ_LIBRARY}")
