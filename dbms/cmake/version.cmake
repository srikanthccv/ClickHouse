# This strings autochanged from release_lib.sh:
set(VERSION_REVISION 54405 CACHE STRING "")
set(VERSION_MAJOR 18 CACHE STRING "")
set(VERSION_MINOR 10 CACHE STRING "")
set(VERSION_PATCH 2 CACHE STRING "")
set(VERSION_GITHASH 39bee180bd7c15dbb35244cc78387628345c1efe CACHE STRING "")
set(VERSION_DESCRIBE v18.10.2-testing CACHE STRING "")
set(VERSION_STRING 18.10.2 CACHE STRING "")
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
