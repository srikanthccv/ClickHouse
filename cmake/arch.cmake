if (CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64")
    if (CMAKE_LIBRARY_ARCHITECTURE MATCHES "i386")
        message (FATAL_ERROR "32bit platforms are not supported")
    endif ()
    set (ARCH_AMD64 1)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set (ARCH_AARCH64 1)
    set (ARCH_ARM 1)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64le.*|PPC64LE.*)")
    set (ARCH_PPC64LE 1)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set (ARCH_RISCV64 1)
else ()
    message (FATAL_ERROR "Platform ${CMAKE_SYSTEM_PROCESSOR} is not supported")
endif ()
