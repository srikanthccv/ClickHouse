#include <sys/auxv.h>
#include <unistd.h> // __environ
#include <errno.h>

// We don't have libc struct available here. Compute aux vector manually.
static unsigned long * __auxv = NULL;
static unsigned long __auxv_secure = 0;

static size_t __find_auxv(unsigned long type)
{
    size_t i;
    for (i = 0; __auxv[i]; i += 2)
    {
        if (__auxv[i] == type)
            return i + 1;
    }
    return (size_t) -1;
}

/// __auxv_init should happen BEFORE the first use of getauxval.
/// but getauxval can be used in other init sections (namely in musl/clock_gettime.c::cgt_init), 
/// so constructor(0) is needed to prioritize that constructor
/// see also: https://stackoverflow.com/questions/11106875/attribute-constructor-call-order-confusion/11198936
__attribute__((constructor(0))) static void __auxv_init()
{
    size_t i;
    for (i = 0; __environ[i]; i++);
    __auxv = (unsigned long *) (__environ + i + 1);

    size_t secure_idx = __find_auxv(AT_SECURE);
    if (secure_idx != ((size_t) -1))
        __auxv_secure = __auxv[secure_idx];
}

unsigned long getauxval(unsigned long type)
{
    if (type == AT_SECURE)
        return __auxv_secure;

    if (__auxv)
    {
        size_t index = __find_auxv(type);
        if (index != ((size_t) -1))
            return __auxv[index];
    }

    errno = ENOENT;
    return 0;
}
