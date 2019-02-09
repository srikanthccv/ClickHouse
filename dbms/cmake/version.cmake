# This strings autochanged from release_lib.sh:
set(VERSION_REVISION 54414)
set(VERSION_MAJOR 19)
set(VERSION_MINOR 2)
set(VERSION_PATCH 0)
set(VERSION_GITHASH dcfca1355468a2d083b33c867effa8f79642ed6e)
set(VERSION_DESCRIBE v19.2.0-testing)
set(VERSION_STRING 19.2.0)
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
