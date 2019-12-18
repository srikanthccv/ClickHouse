# Broken in macos. TODO: update clang, re-test, enable
if (NOT APPLE)
    option (ENABLE_EMBEDDED_COMPILER "Set to TRUE to enable support for 'compile_expressions' option for query execution" ${ENABLE_LIBRARIES})
    option (USE_INTERNAL_LLVM_LIBRARY "Use bundled or system LLVM library." 1)
endif ()

if (ENABLE_EMBEDDED_COMPILER)
    if (USE_INTERNAL_LLVM_LIBRARY AND NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/llvm/llvm/CMakeLists.txt")
        message (WARNING "submodule contrib/llvm is missing. to fix try run: \n git submodule update --init --recursive")
        set (USE_INTERNAL_LLVM_LIBRARY 0)
    endif ()

    if (NOT USE_INTERNAL_LLVM_LIBRARY)
        set (LLVM_PATHS "/usr/local/lib/llvm")

        foreach(llvm_v 9 8)
            if (NOT LLVM_FOUND)
                find_package (LLVM ${llvm_v} CONFIG PATHS ${LLVM_PATHS})
            endif ()
        endforeach ()

        if (LLVM_FOUND)
            # Remove dynamically-linked zlib and libedit from LLVM's dependencies:
            set_target_properties(LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES "-lpthread;LLVMDemangle;${ZLIB_LIBRARIES}")
            set_target_properties(LLVMLineEditor PROPERTIES INTERFACE_LINK_LIBRARIES "LLVMSupport")

            option(LLVM_HAS_RTTI "Enable if LLVM was build with RTTI enabled" ON)
            set (USE_EMBEDDED_COMPILER 1)
        else()
            set (USE_EMBEDDED_COMPILER 0)
        endif()

        if (LLVM_FOUND AND OS_LINUX AND USE_LIBCXX)
            message(WARNING "Option USE_INTERNAL_LLVM_LIBRARY is not set but the LLVM library from OS packages in Linux is incompatible with libc++ ABI. LLVM Will be disabled.")
            set (LLVM_FOUND 0)
            set (USE_EMBEDDED_COMPILER 0)
        endif ()
    else()
        if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
            message(WARNING "Option ENABLE_EMBEDDED_COMPILER is set but LLVM library cannot build if build directory is the same as source directory.")
            set (LLVM_FOUND 0)
            set (USE_EMBEDDED_COMPILER 0)
        else()
            set (LLVM_FOUND 1)
            set (USE_EMBEDDED_COMPILER 1)
            set (LLVM_VERSION "9.0.0bundled")
            set (LLVM_INCLUDE_DIRS
                ${ClickHouse_SOURCE_DIR}/contrib/llvm/llvm/include
                ${ClickHouse_BINARY_DIR}/contrib/llvm/llvm/include
            )
            set (LLVM_LIBRARY_DIRS ${ClickHouse_BINARY_DIR}/contrib/llvm/llvm)
        endif()
    endif()

    if (LLVM_FOUND)
        message(STATUS "LLVM include Directory: ${LLVM_INCLUDE_DIRS}")
        message(STATUS "LLVM library Directory: ${LLVM_LIBRARY_DIRS}")
        message(STATUS "LLVM C++ compiler flags: ${LLVM_CXXFLAGS}")
    endif()
endif()


function(llvm_libs_all REQUIRED_LLVM_LIBRARIES)
    llvm_map_components_to_libnames (result all)
    if (USE_STATIC_LIBRARIES OR NOT "LLVM" IN_LIST result)
        list (REMOVE_ITEM result "LTO" "LLVM")
    else()
        set (result "LLVM")
    endif ()
    list (APPEND result ${CMAKE_DL_LIBS} ${ZLIB_LIBRARIES})
    set (${REQUIRED_LLVM_LIBRARIES} ${result} PARENT_SCOPE)
endfunction()
