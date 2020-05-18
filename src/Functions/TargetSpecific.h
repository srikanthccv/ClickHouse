#pragma once

#include <Core/Types.h>

/* This file contains macros and helpers for writing platform-dependent code.
 * 
 * Macros DECLARE_<Arch>_SPECIFIC_CODE will wrap code inside them into the 
 * namespace TargetSpecific::<Arch> and enable Arch-specific compile options.
 * Thus, it's allowed to call functions inside these namespaces only after
 * checking platform in runtime (see IsArchSupported() below).
 *
 * For similarities there is a macros DECLARE_DEFAULT_CODE, which wraps code
 * into the namespace TargetSpecific::Default but dosn't specify any additional
 * copile options.
 * 
 * Example:
 * 
 * DECLARE_DEFAULT_CODE (
 * int funcImpl() {
 *     return 1;
 * }
 * ) // DECLARE_DEFAULT_CODE
 * 
 * DECLARE_AVX2_SPECIFIC_CODE (
 * int funcImpl() {
 *     return 2;
 * }
 * ) // DECLARE_DEFAULT_CODE
 * 
 * int func() {
 *     if (IsArchSupported(TargetArch::AVX2)) 
 *         return TargetSpecifc::AVX2::funcImpl();
 *     return TargetSpecifc::Default::funcImpl();
 * } 
 * 
 * Sometimes code may benefit from compiling with different options.
 * For these purposes use DECLARE_MULTITARGET_CODE macros. It will create several
 * copies of the code and compile it with different options. These copies are
 * available via TargetSpecifc namespaces described above.
 * 
 * Inside every TargetSpecific namespace there is a constexpr variable BuildArch, 
 * which indicates the target platform for current code.
 * 
 * Example:
 * 
 * DECLARE_MULTITARGET_CODE(
 * int funcImpl(int size, ...) {
 *     int iteration_size = 1;
 *     if constexpr (BuildArch == TargetArch::SSE42)
 *         iteration_size = 2
 *     else if constexpr (BuildArch == TargetArch::AVX || BuildArch == TargetArch::AVX2)
 *         iteration_size = 4;
 *     else if constexpr (BuildArch == TargetArch::AVX512)
 *         iteration_size = 8;
 *     for (int i = 0; i < size; i += iteration_size)
 *     ...
 * }
 * ) // DECLARE_MULTITARGET_CODE
 *
 * // All 5 versions of func are available here. Use runtime detection to choose one.
 *
 * If you want to write IFunction or IExecutableFuncionImpl with runtime dispatching, see PerformanceAdaptors.h.
 */

namespace DB
{

enum class TargetArch : UInt32
{
    Default  = 0,         /// Without any additional compiler options.
    SSE42    = (1 << 0),  /// SSE4.2
    AVX      = (1 << 1),
    AVX2     = (1 << 2),
    AVX512F  = (1 << 3),
};

// Runtime detection.
bool IsArchSupported(TargetArch arch);

String ToString(TargetArch arch);

#if USE_MULTITARGET_CODE && defined(__GNUC__) && defined(__x86_64__)

constexpr bool UseMultitargetCode = true;

#if defined(__clang__)
#   define BEGIN_AVX512F_SPECIFIC_CODE \
        _Pragma("clang attribute push(__attribute__((target(\"sse,sse2,sse3,ssse3,sse4,popcnt,mmx,avx,avx2,avx512f\"))),apply_to=function)")
#   define BEGIN_AVX2_SPECIFIC_CODE \
        _Pragma("clang attribute push(__attribute__((target(\"sse,sse2,sse3,ssse3,sse4,popcnt,mmx,avx,avx2\"))),apply_to=function)")
#   define BEGIN_AVX_SPECIFIC_CODE \
        _Pragma("clang attribute push(__attribute__((target(\"sse,sse2,sse3,ssse3,sse4,popcnt,mmx,avx\"))),apply_to=function)")
#   define BEGIN_SSE42_SPECIFIC_CODE \
        _Pragma("clang attribute push(__attribute__((target(\"sse,sse2,sse3,ssse3,sse4,popcnt,mmx\"))),apply_to=function)")
#   define END_TARGET_SPECIFIC_CODE \
        _Pragma("clang attribute pop")

/* Clang shows warning when there aren't any objects to apply pragma.
 * To prevent this warning we define this function inside every macros with pragmas.
 */
#   define DUMMY_FUNCTION_DEFINITION void __dummy_function_definition();
#else
#   define BEGIN_AVX512F_SPECIFIC_CODE \
        _Pragma("GCC push_options") \
        _Pragma("GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,avx512f,tune=native\")")
#   define BEGIN_AVX2_SPECIFIC_CODE \
        _Pragma("GCC push_options") \
        _Pragma("GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native\")")
#   define BEGIN_AVX_SPECIFIC_CODE \
        _Pragma("GCC push_options") \
        _Pragma("GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native\")")
#   define BEGIN_SSE42_SPECIFIC_CODE \
        _Pragma("GCC push_options") \
        _Pragma("GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,tune=native\")")
#   define END_TARGET_SPECIFIC_CODE \
        _Pragma("GCC pop_options")

/* GCC doesn't show such warning, we don't need to define anything.
 */
#   define DUMMY_FUNCTION_DEFINITION
#endif

#define DECLARE_SSE42_SPECIFIC_CODE(...) \
BEGIN_SSE42_SPECIFIC_CODE \
namespace TargetSpecific::SSE42 { \
    DUMMY_FUNCTION_DEFINITION \
    using namespace DB::TargetSpecific::SSE42; \
    __VA_ARGS__ \
} \
END_TARGET_SPECIFIC_CODE

#define DECLARE_AVX_SPECIFIC_CODE(...) \
BEGIN_AVX_SPECIFIC_CODE \
namespace TargetSpecific::AVX { \
    DUMMY_FUNCTION_DEFINITION \
    using namespace DB::TargetSpecific::AVX; \
    __VA_ARGS__ \
} \
END_TARGET_SPECIFIC_CODE

#define DECLARE_AVX2_SPECIFIC_CODE(...) \
BEGIN_AVX2_SPECIFIC_CODE \
namespace TargetSpecific::AVX2 { \
    DUMMY_FUNCTION_DEFINITION \
    using namespace DB::TargetSpecific::AVX2; \
    __VA_ARGS__ \
} \
END_TARGET_SPECIFIC_CODE

#define DECLARE_AVX512F_SPECIFIC_CODE(...) \
BEGIN_AVX512F_SPECIFIC_CODE \
namespace TargetSpecific::AVX512F { \
    DUMMY_FUNCTION_DEFINITION \
    using namespace DB::TargetSpecific::AVX512F; \
    __VA_ARGS__ \
} \
END_TARGET_SPECIFIC_CODE

#else

constexpr bool UseMultitargetCode = false;

#define DECLARE_SSE42_SPECIFIC_CODE(...)
#define DECLARE_AVX_SPECIFIC_CODE(...)
#define DECLARE_AVX2_SPECIFIC_CODE(...)
#define DECLARE_AVX512F_SPECIFIC_CODE(...)

#endif

#define DECLARE_DEFAULT_CODE(...) \
namespace TargetSpecific::Default { \
    using namespace DB::TargetSpecific::Default; \
    __VA_ARGS__ \
}

#define DECLARE_MULTITARGET_CODE(...) \
DECLARE_DEFAULT_CODE         (__VA_ARGS__) \
DECLARE_SSE42_SPECIFIC_CODE  (__VA_ARGS__) \
DECLARE_AVX_SPECIFIC_CODE    (__VA_ARGS__) \
DECLARE_AVX2_SPECIFIC_CODE   (__VA_ARGS__) \
DECLARE_AVX512F_SPECIFIC_CODE(__VA_ARGS__)

DECLARE_DEFAULT_CODE(
    constexpr auto BuildArch = TargetArch::Default;
) // DECLARE_DEFAULT_CODE

DECLARE_SSE42_SPECIFIC_CODE(
    constexpr auto BuildArch = TargetArch::SSE42;
) // DECLARE_SSE42_SPECIFIC_CODE

DECLARE_AVX_SPECIFIC_CODE(
    constexpr auto BuildArch = TargetArch::AVX;
) // DECLARE_AVX_SPECIFIC_CODE

DECLARE_AVX2_SPECIFIC_CODE(
    constexpr auto BuildArch = TargetArch::AVX2;
) // DECLARE_AVX2_SPECIFIC_CODE

DECLARE_AVX512F_SPECIFIC_CODE(
    constexpr auto BuildArch = TargetArch::AVX512F;
) // DECLARE_AVX512F_SPECIFIC_CODE

}
