#!/bin/bash

# Manual run:
# env CXX=g++-7 CC=gcc-7 utils/travis/normal.sh
# env CXX=clang++-5.0 CC=clang-5.0 utils/travis/normal.sh

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

set -e
set -x

date

# clean not used ~600mb
[ -n "$TRAVIS" ] && rm -rf .git contrib/poco/openssl

ccache -s
ccache -M ${CCACHE_SIZE:=4G}
df -h

date

mkdir -p build
cd build
cmake $CUR_DIR/../.. -D CMAKE_CXX_COMPILER=`which $DEB_CXX $CXX` -D CMAKE_C_COMPILER=`which $DEB_CC $CC` \
    `# Does not optimize to speedup build, skip debug info to use less disk` \
    -D CMAKE_C_FLAGS_ADD="-O0 -g0" -D CMAKE_CXX_FLAGS_ADD="-O0 -g0" \
    `# ignore ccache disabler on trusty` \
    -D CMAKE_C_COMPILER_LAUNCHER=`which ccache` -D CMAKE_CXX_COMPILER_LAUNCHER=`which ccache` \
    `# Use all possible contrib libs from system` \
    -D UNBUNDLED=1 \
    `# Disable all features` \
    -D ENABLE_CAPNP=0 -D ENABLE_RDKAFKA=0 -D ENABLE_EMBEDDED_COMPILER=0 -D ENABLE_TCMALLOC=0 -D ENABLE_UNWIND=0 -D ENABLE_MYSQL=0 -D USE_INTERNAL_LLVM_LIBRARY=0 $CMAKE_FLAGS \
    && make -j `nproc || grep -c ^processor /proc/cpuinfo || sysctl -n hw.ncpu || echo 4` clickhouse-bundle \
    `# Skip tests:` \
    `# 00281 requires internal compiler` \
    `# 00428 requires sudo (not all vms allow this)` \
    && ( [ ! ${TEST_RUN=1} ] || ( ( cd $CUR_DIR/../.. && env TEST_OPT="--skip long compile 00428 $TEST_OPT" TEST_PERF= bash -x dbms/tests/clickhouse-test-server ) || ${TEST_TRUE=false} ) )

date
