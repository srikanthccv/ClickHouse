#include <Common/checkStackSize.h>
#include <Common/Exception.h>
#include <ext/scope_guard.h>
#include <pthread.h>
#include <cstdint>
#include <sstream>

#if defined(__FreeBSD__)
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

__attribute__((__weak__)) void checkStackSize()
{
    using namespace DB;

    if (!stack_address)
    {
#if defined(OS_DARWIN)
        // pthread_get_stacksize_np() returns a value too low for the main thread on
        // OSX 10.9, http://mail.openjdk.java.net/pipermail/hotspot-dev/2013-October/011369.html
        //
        // Multiple workarounds possible, adopt the one made by https://github.com/robovm/robovm/issues/274
        // https://developer.apple.com/library/mac/documentation/Cocoa/Conceptual/Multithreading/CreatingThreads/CreatingThreads.html
        // Stack size for the main thread is 8MB on OSX excluding the guard page size.
        pthread_t thread = pthread_self();
        max_stack_size = pthread_main_np() ? (8 * 1024 * 1024) : pthread_get_stacksize_np(thread);

        // stack_address points to the start of the stack, not the end how it's returned by pthread_get_stackaddr_np
        stack_address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(thread)) - max_stack_size);
#else
        pthread_attr_t attr;
#   if defined(__FreeBSD__)
        pthread_attr_init(&attr);
        if (0 != pthread_attr_get_np(pthread_self(), &attr))
            throwFromErrno("Cannot pthread_attr_get_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#   else
        if (0 != pthread_getattr_np(pthread_self(), &attr))
            throwFromErrno("Cannot pthread_getattr_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#   endif

        SCOPE_EXIT({ pthread_attr_destroy(&attr); });

        if (0 != pthread_attr_getstack(&attr, &stack_address, &max_stack_size))
            throwFromErrno("Cannot pthread_getattr_np", ErrorCodes::CANNOT_PTHREAD_ATTR);
#endif // OS_DARWIN
    }

    const void * frame_address = __builtin_frame_address(0);
    uintptr_t int_frame_address = reinterpret_cast<uintptr_t>(frame_address);
    uintptr_t int_stack_address = reinterpret_cast<uintptr_t>(stack_address);

    /// We assume that stack grows towards lower addresses. And that it starts to grow from the end of a chunk of memory of max_stack_size.
    if (int_frame_address > int_stack_address + max_stack_size)
        throw Exception("Logical error: frame address is greater than stack begin address", ErrorCodes::LOGICAL_ERROR);

    size_t stack_size = int_stack_address + max_stack_size - int_frame_address;

    /// Just check if we have already eat more than a half of stack size. It's a bit overkill (a half of stack size is wasted).
    /// It's safe to assume that overflow in multiplying by two cannot occur.
    if (stack_size * 2 > max_stack_size)
    {
        std::stringstream message;
        message << "Stack size too large"
            << ". Stack address: " << stack_address
            << ", frame address: " << frame_address
            << ", stack size: " << stack_size
            << ", maximum stack size: " << max_stack_size;
        throw Exception(message.str(), ErrorCodes::TOO_DEEP_RECURSION);
    }
}
