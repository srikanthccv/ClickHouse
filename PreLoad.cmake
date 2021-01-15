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
        set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
    endif ()
endif()
