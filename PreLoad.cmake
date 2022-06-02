# Use Ninja instead of Unix Makefiles by default.
# https://stackoverflow.com/questions/11269833/cmake-selecting-a-generator-within-cmakelists-txt
#
# Reason: it has better startup time than make and it parallelizes jobs more uniformly.
# (when comparing to make with Makefiles that was generated by CMake)
#
# How to install Ninja on Ubuntu:
#  sudo apt-get install ninja-build

# CLion does not support Ninja
# You can add your vote on CLion task tracker:
# https://youtrack.jetbrains.com/issue/CPP-2659
# https://youtrack.jetbrains.com/issue/CPP-870

if (NOT DEFINED ENV{CLION_IDE} AND NOT DEFINED ENV{XCODE_IDE})
    find_program(NINJA_PATH ninja)
    if (NINJA_PATH)
        set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "")
    endif ()
endif()

# Check if environment is polluted.
if (NOT $ENV{CFLAGS} STREQUAL ""
    OR NOT $ENV{CXXFLAGS} STREQUAL ""
    OR NOT $ENV{LDFLAGS} STREQUAL ""
    OR CMAKE_C_FLAGS OR CMAKE_CXX_FLAGS OR CMAKE_EXE_LINKER_FLAGS OR CMAKE_SHARED_LINKER_FLAGS OR CMAKE_MODULE_LINKER_FLAGS
    OR CMAKE_C_FLAGS_INIT OR CMAKE_CXX_FLAGS_INIT OR CMAKE_EXE_LINKER_FLAGS_INIT OR CMAKE_SHARED_LINKER_FLAGS_INIT OR CMAKE_MODULE_LINKER_FLAGS_INIT)

    message("CFLAGS: $ENV{CFLAGS}")
    message("CXXFLAGS: $ENV{CXXFLAGS}")
    message("LDFLAGS: $ENV{LDFLAGS}")
    message("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
    message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    message("CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
    message("CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
    message("CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")

    message(FATAL_ERROR "
        Some of the variables like CFLAGS, CXXFLAGS, LDFLAGS are not empty.
        It is not possible to build ClickHouse with custom flags.
        These variables can be set up by previous invocation of some other build tools.
        You should cleanup these variables and start over again.

        Run the `env` command to check the details.
        You will also need to remove the contents of the build directory.

        Note: if you don't like this behavior, you can manually edit the cmake files, but please don't complain to developers.")
endif()

# Default toolchain - this is needed to avoid dependency on OS files.
execute_process(COMMAND uname -s OUTPUT_VARIABLE OS)
execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH)

if (OS MATCHES "Linux"
    AND NOT DEFINED CMAKE_TOOLCHAIN_FILE
    AND NOT DISABLE_HERMETIC_BUILD
    AND ($ENV{CC} MATCHES ".*clang.*" OR CMAKE_C_COMPILER MATCHES ".*clang.*"))

    if (ARCH MATCHES "amd64|x86_64")
        set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-x86_64.cmake" CACHE INTERNAL "")
    elseif (ARCH MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-aarch64.cmake" CACHE INTERNAL "")
    elseif (ARCH MATCHES "^(ppc64le.*|PPC64LE.*)")
        set (CMAKE_TOOLCHAIN_FILE "cmake/linux/toolchain-ppc64le.cmake" CACHE INTERNAL "")
else ()
        message (FATAL_ERROR "Unsupported architecture: ${ARCH}")
    endif ()

endif()
