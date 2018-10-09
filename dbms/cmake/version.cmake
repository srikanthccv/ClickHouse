# This strings autochanged from release_lib.sh:
set(VERSION_REVISION 54409 CACHE STRING "")
set(VERSION_MAJOR 18 CACHE STRING "")
set(VERSION_MINOR 14 CACHE STRING "")
set(VERSION_PATCH 5 CACHE STRING "")
set(VERSION_GITHASH ab198e77d1c0ee73c711f24f8638a6d6c226eec9 CACHE STRING "")
set(VERSION_DESCRIBE v18.14.5-testing CACHE STRING "")
set(VERSION_STRING 18.14.5 CACHE STRING "")
# end of autochange

set(VERSION_EXTRA "" CACHE STRING "")
set(VERSION_TWEAK "" CACHE STRING "")

if (VERSION_TWEAK)
    string(CONCAT VERSION_STRING ${VERSION_STRING} "." ${VERSION_TWEAK})
endif ()

if (VERSION_EXTRA)
    string(CONCAT VERSION_STRING ${VERSION_STRING} "." ${VERSION_EXTRA})
endif ()

set (VERSION_NAME "${PROJECT_NAME}" CACHE STRING "")
set (VERSION_FULL "${VERSION_NAME} ${VERSION_STRING}" CACHE STRING "")
set (VERSION_SO "${VERSION_STRING}" CACHE STRING "")
