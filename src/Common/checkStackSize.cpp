#include <Common/checkStackSize.h>
#include <Common/Exception.h>
#include <base/scope_guard.h>
#include <base/defines.h> /// THREAD_SANITIZER
#include <pthread.h>
#include <cstdint>

#if defined(OS_FREEBSD)
#   include <pthread_np.h>
#endif


namespace DB
{
    namespace ErrorCodes
    {
        extern const int CANNOT_PTHREAD_ATTR;
        extern const int LOGICAL_ERROR;
        extern const int TOO_DEEP_RECURSION;
    }
}

static thread_local void * stack_address = nullptr;
static thread_local size_t max_stack_size = 0;

/**
 * @param out_address - if not nullptr, here the address of the stack will be written.
 * @return stack size
 */
size_t getStackSize(void ** out_address)
{
    using namespace DB;

    size_t size;
    void * address;

#if defined(OS_DARWIN)
    // pthread_get_stacksize_np() returns a value too low for the main thread on
    // OSX 10.9, http://mail.openjdk.java.net/pipermail/hotspot-dev/2013-October/011369.html
    //
    // Multiple workarounds possible, adopt the one made by https://github.com/robovm/robovm/issues/274
    // https://developer.apple.com/library/mac/documentation/Cocoa/Conceptual/Multithreading/CreatingThreads/CreatingThreads.html
    // Stack size for the main thread is 8MB on OSX excluding the guard page size.
    pthread_t thread = pthread_self();
    size = pthread_main_np() ? (8 * 1024 * 1024) : pthread_get_stacksize_np(thread);

    // stack address points to the start of the stack, not the end how it's returned by pthread_get_stackaddr_np
    address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(thread)) - size);
#else
    pthread_attr_t attr;
#   if defined(OS_FREEBSD) || defined(OS_SUNOS)
    pthread_attr_init(&attr);
    if (0 != pthread_attr_get_np(pthread_self(), &attr))
        throwFromErrno("Cannot pthread_attr_get_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#   else
    if (0 != pthread_getattr_np(pthread_self(), &attr))
        throwFromErrno("Cannot pthread_getattr_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#   endif

    SCOPE_EXIT({ pthread_attr_destroy(&attr); });

    if (0 != pthread_attr_getstack(&attr, &address, &size))
        throwFromErrno("Cannot pthread_getattr_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#endif // OS_DARWIN

    if (out_address)
        *out_address = address;

    return size;
}

/** It works fine when interpreters are instantiated by ClickHouse code in properly prepared threads,
  *  but there are cases when ClickHouse runs as a library inside another application.
  * If application is using user-space lightweight threads with manually allocated stacks,
  *  current implementation is not reasonable, as it has no way to properly check the remaining
  *  stack size without knowing the details of how stacks are allocated.
  * We mark this function as weak symbol to be able to replace it in another ClickHouse-based products.
  */
__attribute__((__weak__)) void checkStackSize()
{
    using namespace DB;

    if (!stack_address)
        max_stack_size = getStackSize(&stack_address);

    const void * frame_address = __builtin_frame_address(0);
    uintptr_t int_frame_address = reinterpret_cast<uintptr_t>(frame_address);
    uintptr_t int_stack_address = reinterpret_cast<uintptr_t>(stack_address);

#if !defined(THREAD_SANITIZER)
    /// It's overkill to use more then half of stack.
    static constexpr double STACK_SIZE_FREE_RATIO = 0.5;
#else
    /// Under TSan recursion eats too much RAM, so half of stack is too much.
    /// So under TSan only 5% of a stack is allowed (this is ~400K)
    static constexpr double STACK_SIZE_FREE_RATIO = 0.05;
#endif

    /// We assume that stack grows towards lower addresses. And that it starts to grow from the end of a chunk of memory of max_stack_size.
    if (int_frame_address > int_stack_address + max_stack_size)
        throw Exception("Logical error: frame address is greater than stack begin address", ErrorCodes::LOGICAL_ERROR);

    size_t stack_size = int_stack_address + max_stack_size - int_frame_address;
    size_t max_stack_size_allowed = max_stack_size * STACK_SIZE_FREE_RATIO;

    /// Just check if we have eat more than a STACK_SIZE_FREE_RATIO of stack size already.
    if (stack_size > max_stack_size_allowed)
    {
        throw Exception(ErrorCodes::TOO_DEEP_RECURSION,
                        "Stack size too large. Stack address: {}, frame address: {}, stack size: {}, maximum stack size: {}",
                        stack_address, frame_address, stack_size, max_stack_size);
    }
}
