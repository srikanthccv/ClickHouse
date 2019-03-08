# This strings autochanged from release_lib.sh:
set(VERSION_REVISION 54416)
set(VERSION_MAJOR 19)
set(VERSION_MINOR 4)
set(VERSION_PATCH 0)
set(VERSION_GITHASH 84dd4aac8bf9847b6a5a496af1babf2dc133966a)
set(VERSION_DESCRIBE v19.4.0-testing)
set(VERSION_STRING 19.4.0)
# end of autochange

set(VERSION_EXTRA "" CACHE STRING "")
set(VERSION_TWEAK "" CACHE STRING "")

if (VERSION_TWEAK)
    string(CONCAT VERSION_STRING ${VERSION_STRING} "." ${VERSION_TWEAK})
endif ()

if (VERSION_EXTRA)
    string(CONCAT VERSION_STRING ${VERSION_STRING} "." ${VERSION_EXTRA})
endif ()

set (VERSION_NAME "${PROJECT_NAME}")
set (VERSION_FULL "${VERSION_NAME} ${VERSION_STRING}")
set (VERSION_SO "${VERSION_STRING}")

math (EXPR VERSION_INTEGER "${VERSION_PATCH} + ${VERSION_MINOR}*1000 + ${VERSION_MAJOR}*1000000")
