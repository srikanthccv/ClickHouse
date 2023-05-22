set (FULL_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UC}}")
set (FULL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UC}}")
set (FULL_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${CMAKE_BUILD_TYPE_UC}}")

message (STATUS "compiler C   = ${CMAKE_C_COMPILER} ${FULL_C_FLAGS}")
message (STATUS "compiler CXX = ${CMAKE_CXX_COMPILER} ${FULL_CXX_FLAGS}")
message (STATUS "LINKER_FLAGS = ${FULL_EXE_LINKER_FLAGS}")

# Reproducible builds
string (REPLACE "${PROJECT_SOURCE_DIR}" "." FULL_C_FLAGS_NORMALIZED "${FULL_C_FLAGS}")
string (REPLACE "${PROJECT_SOURCE_DIR}" "." FULL_CXX_FLAGS_NORMALIZED "${FULL_CXX_FLAGS}")
string (REPLACE "${PROJECT_SOURCE_DIR}" "." FULL_EXE_LINKER_FLAGS_NORMALIZED "${FULL_EXE_LINKER_FLAGS}")
