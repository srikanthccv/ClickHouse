set (CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set (CMAKE_SYSTEM_NAME "Linux")
set (CMAKE_SYSTEM_PROCESSOR "x86_64")
set (CMAKE_C_COMPILER_TARGET "x86_64-linux-musl")
set (CMAKE_CXX_COMPILER_TARGET "x86_64-linux-musl")
set (CMAKE_ASM_COMPILER_TARGET "x86_64-linux-musl")

set (TOOLCHAIN_PATH "${CMAKE_CURRENT_LIST_DIR}/../../contrib/sysroot/linux-x86_64-musl/x86_64-linux-musl-cross")

set (CMAKE_SYSROOT "${TOOLCHAIN_PATH}/x86_64-linux-musl")

find_program (LLVM_AR_PATH NAMES "llvm-ar" "llvm-ar-13" "llvm-ar-12" "llvm-ar-11" "llvm-ar-10" "llvm-ar-9" "llvm-ar-8")
find_program (LLVM_RANLIB_PATH NAMES "llvm-ranlib" "llvm-ranlib-13" "llvm-ranlib-12" "llvm-ranlib-11" "llvm-ranlib-10" "llvm-ranlib-9")

set (CMAKE_AR "${LLVM_AR_PATH}" CACHE FILEPATH "" FORCE)
set (CMAKE_RANLIB "${LLVM_RANLIB_PATH}" CACHE FILEPATH "" FORCE)

set (CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS} --gcc-toolchain=${TOOLCHAIN_PATH}")
set (CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS} --gcc-toolchain=${TOOLCHAIN_PATH}")
set (CMAKE_ASM_FLAGS_INIT "${CMAKE_ASM_FLAGS} --gcc-toolchain=${TOOLCHAIN_PATH}")

set (LINKER_NAME "ld.lld" CACHE STRING "" FORCE)

set (CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set (CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")

set (HAS_PRE_1970_EXITCODE "0" CACHE STRING "Result from TRY_RUN" FORCE)
set (HAS_PRE_1970_EXITCODE__TRYRUN_OUTPUT "" CACHE STRING "Output from TRY_RUN" FORCE)

set (HAS_POST_2038_EXITCODE "0" CACHE STRING "Result from TRY_RUN" FORCE)
set (HAS_POST_2038_EXITCODE__TRYRUN_OUTPUT "" CACHE STRING "Output from TRY_RUN" FORCE)

set (USE_MUSL 1)
add_definitions(-DUSE_MUSL=1)
