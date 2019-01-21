# This strings autochanged from release_lib.sh:
set(VERSION_REVISION 54413)
set(VERSION_MAJOR 19)
set(VERSION_MINOR 1)
set(VERSION_PATCH 2)
set(VERSION_GITHASH 0728519b2ddd94f3c59fa79fd9ac97ffda4e605d)
set(VERSION_DESCRIBE v19.1.2-testing)
set(VERSION_STRING 19.1.2)
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
