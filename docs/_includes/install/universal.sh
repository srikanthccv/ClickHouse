#!/bin/sh -e

OS=$(uname -s)
ARCH=$(uname -m)

DIR=

if [ "${OS}" = "Linux" ]
then
    if [ "${ARCH}" = "x86_64" -o "${ARCH}" = "amd64" ]
    then
        # Require at least SSE4.2 (introduced in 2006) but also provide a fallback to SSE2 (introduced in 2000) for older systems. SSE2
        # builds are much less tested than SSE 4.2 builds. Hardware w/o SSE2 is unsupported.
        HAS_SSE42=$(grep sse4_2 /proc/cpuinfo)
        if [ "${HAS_SSE42}" ]
        then
            DIR="amd64"
        else
            HAS_SSE2=$(grep sse2 /proc/cpuinfo)
            if [ "${HAS_SSE2}" ]
            then
                DIR="amd64sse2"
            fi
        fi
    elif [ "${ARCH}" = "aarch64" -o "${ARCH}" = "arm64" ]
    then
        # If the system has >=ARMv8.2 (https://en.wikipedia.org/wiki/AArch64), choose the corresponding build, else fall back to a v8.0
        # compat build. Unfortunately, the ARM ISA level cannot be read directly, we need to guess from the "features" in /proc/cpuinfo.
        # Also, the flags in /proc/cpuinfo are named differently than the flags passed to the compiler (cmake/cpu_features.cmake).
        HAS_ARMV82=$(grep -m 1 'Features' /proc/cpuinfo | awk '/asimd/ && /sha1/ && /aes/ && /atomics/ && /lrcpc/')
        if [ "${HAS_ARMV82}" ]
        then
            DIR="aarch64"
        else
            DIR="aarch64v80compat"
        fi
    elif [ "${ARCH}" = "powerpc64le" -o "${ARCH}" = "ppc64le" ]
    then
        DIR="powerpc64le"
    fi
elif [ "${OS}" = "FreeBSD" ]
then
    if [ "${ARCH}" = "x86_64" -o "${ARCH}" = "amd64" ]
    then
        DIR="freebsd"
    fi
elif [ "${OS}" = "Darwin" ]
then
    if [ "${ARCH}" = "x86_64" -o "${ARCH}" = "amd64" ]
    then
        DIR="macos"
    elif [ "${ARCH}" = "aarch64" -o "${ARCH}" = "arm64" ]
    then
        DIR="macos-aarch64"
    fi
fi

if [ -z "${DIR}" ]
then
    echo "Operating system '${OS}' / architecture '${ARCH}' is unsupported."
    exit 1
fi

clickhouse_download_filename_prefix="clickhouse"
clickhouse="$clickhouse_download_filename_prefix"

i=0
while [ -f "$clickhouse" ]
do
    clickhouse="${clickhouse_download_filename_prefix}.${i}"
    i=$(($i+1))
done

URL="https://builds.clickhouse.com/master/${DIR}/clickhouse"
echo
echo "Will download ${URL} into ${clickhouse}"
echo
curl "${URL}" -o "${clickhouse}" && chmod a+x "${clickhouse}" || exit 1
echo
echo "Successfully downloaded the ClickHouse binary, you can run it as:
    ./${clickhouse}"

if [ "${OS}" = "Linux" ]
then
    echo
    echo "You can also install it:
    sudo ./${clickhouse} install"
fi
