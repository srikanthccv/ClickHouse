# This strings autochanged from release_lib.sh:
set(VERSION_DESCRIBE v1.1.54378-testing)
set(VERSION_REVISION 54378)
set(VERSION_GITHASH 5b19d89133a5ff7c72e40cc8c0226cb00466ba10)
# end of autochange

set (VERSION_MAJOR 1)
set (VERSION_MINOR 1)
set (VERSION_PATCH ${VERSION_REVISION})
set (VERSION_EXTRA "")
set (VERSION_TWEAK "")

set (VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
if (VERSION_TWEAK)
    set(VERSION_STRING "${VERSION_STRING}.${VERSION_TWEAK}")
endif ()
if (VERSION_EXTRA)
    set(VERSION_STRING "${VERSION_STRING}${VERSION_EXTRA}")
endif ()

set (VERSION_FULL "${PROJECT_NAME} ${VERSION_STRING}")

if (APPLE)
    # dirty hack: ld: malformed 64-bit a.b.c.d.e version number: 1.1.54160
    math (EXPR VERSION_SO1 "${VERSION_REVISION}/255")
    math (EXPR VERSION_SO2 "${VERSION_REVISION}%255")
    set (VERSION_SO "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_SO1}.${VERSION_SO2}")
else ()
    set (VERSION_SO "${VERSION_STRING}")
endif ()
