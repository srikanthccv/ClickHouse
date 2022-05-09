/** This library provides runtime instrumentation (hardening)
  * that ensures no "harmful" functions from libc are called
  * (by terminating the program immediately).
  */

/// It is only enabled in debug build (its intended use is for CI checks).
#if !defined(NDEBUG)

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wincompatible-library-redeclaration"
#else
    #pragma GCC diagnostic ignored "-Wbuiltin-declaration-mismatch"
#endif

/// We cannot use libc headers here.
long write(int, const void *, unsigned long);
#define TRAP(func) void func() { write(2, #func "\n", __builtin_strlen(#func) + 1); __builtin_trap(); }

/// Trap all non thread-safe functions:
/// nm -D /lib/x86_64-linux-gnu/{libc.so.6,libdl.so.2,libm.so.6,libpthread.so.0,librt.so.1,libnss_dns.so.2,libresolv.so.2} | grep -P '_r@?$' | awk '{ print $3 }' | sed -r -e 's/_r//' | grep -vP '^_'

/// See also https://reviews.llvm.org/D90944

/// You can edit this list and even comment out some functions.
/// The only purpose of the library is to force you to pay attention.

TRAP(argp_error)
TRAP(argp_help)
TRAP(argp_parse)
TRAP(argp_state_help)
TRAP(argp_usage)
TRAP(asctime)
TRAP(clearenv)
TRAP(crypt)
TRAP(ctime)
TRAP(cuserid)
TRAP(drand48)
TRAP(ecvt)
TRAP(encrypt)
TRAP(endfsent)
TRAP(endgrent)
TRAP(endhostent)
TRAP(endnetent)
TRAP(endnetgrent)
TRAP(endprotoent)
TRAP(endpwent)
TRAP(endservent)
TRAP(endutent)
TRAP(endutxent)
TRAP(erand48)
TRAP(error_at_line)
///TRAP(exit)
TRAP(fcloseall)
TRAP(fcvt)
TRAP(fgetgrent)
TRAP(fgetpwent)
TRAP(gammal)
TRAP(getchar_unlocked)
TRAP(getdate)
TRAP(getfsent)
TRAP(getfsfile)
TRAP(getfsspec)
TRAP(getgrent)
TRAP(getgrent_r)
TRAP(getgrgid)
TRAP(getgrnam)
TRAP(gethostbyaddr)
TRAP(gethostbyname)
TRAP(gethostbyname2)
TRAP(gethostent)
TRAP(getlogin)
TRAP(getmntent)
TRAP(getnetbyaddr)
TRAP(getnetbyname)
TRAP(getnetent)
TRAP(getnetgrent)
TRAP(getnetgrent_r)
TRAP(getopt)
TRAP(getopt_long)
TRAP(getopt_long_only)
TRAP(getpass)
TRAP(getprotobyname)
TRAP(getprotobynumber)
TRAP(getprotoent)
TRAP(getpwent)
TRAP(getpwent_r)
TRAP(getpwnam)
TRAP(getpwuid)
TRAP(getservbyname)
TRAP(getservbyport)
TRAP(getservent)
TRAP(getutent)
TRAP(getutent_r)
TRAP(getutid)
TRAP(getutid_r)
TRAP(getutline)
TRAP(getutline_r)
TRAP(getutxent)
TRAP(getutxid)
TRAP(getutxline)
TRAP(getwchar_unlocked)
//TRAP(glob)
//TRAP(glob64)
TRAP(gmtime)
TRAP(hcreate)
TRAP(hdestroy)
TRAP(hsearch)
TRAP(innetgr)
TRAP(jrand48)
TRAP(l64a)
TRAP(lcong48)
TRAP(lgammafNx)
TRAP(localeconv)
TRAP(localtime)
TRAP(login)
TRAP(login_tty)
TRAP(logout)
TRAP(logwtmp)
TRAP(lrand48)
TRAP(mallinfo)
#if !defined(SANITIZER)
TRAP(mallopt) // Used by tsan
#endif
TRAP(mblen)
TRAP(mbrlen)
TRAP(mbrtowc)
TRAP(mbsnrtowcs)
TRAP(mbsrtowcs)
//TRAP(mbtowc) // Used by Standard C++ library
TRAP(mcheck)
TRAP(mprobe)
TRAP(mrand48)
TRAP(mtrace)
TRAP(muntrace)
TRAP(nrand48)
TRAP(__ppc_get_timebase_freq)
TRAP(ptsname)
TRAP(putchar_unlocked)
TRAP(putenv)
TRAP(pututline)
TRAP(pututxline)
TRAP(putwchar_unlocked)
TRAP(qecvt)
TRAP(qfcvt)
TRAP(register_printf_function)
TRAP(seed48)
//TRAP(setenv)
TRAP(setfsent)
TRAP(setgrent)
TRAP(sethostent)
TRAP(sethostid)
TRAP(setkey)
//TRAP(setlocale) // Used by replxx at startup
TRAP(setlogmask)
TRAP(setnetent)
TRAP(setnetgrent)
TRAP(setprotoent)
TRAP(setpwent)
TRAP(setservent)
TRAP(setutent)
TRAP(setutxent)
TRAP(siginterrupt)
TRAP(sigpause)
//TRAP(sigprocmask)
TRAP(sigsuspend)
TRAP(sleep)
TRAP(srand48)
//TRAP(strerror) // Used by RocksDB and many other libraries, unfortunately.
//TRAP(strsignal) // This function is imported from Musl and is thread safe.
TRAP(strtok)
TRAP(tcflow)
TRAP(tcsendbreak)
TRAP(tmpnam)
TRAP(ttyname)
TRAP(unsetenv)
TRAP(updwtmp)
TRAP(utmpname)
TRAP(utmpxname)
//TRAP(valloc)
TRAP(vlimit)
//TRAP(wcrtomb) // Used by Standard C++ library
TRAP(wcsnrtombs)
TRAP(wcsrtombs)
TRAP(wctomb)
TRAP(basename)
TRAP(catgets)
TRAP(dbm_clearerr)
TRAP(dbm_close)
TRAP(dbm_delete)
TRAP(dbm_error)
TRAP(dbm_fetch)
TRAP(dbm_firstkey)
TRAP(dbm_nextkey)
TRAP(dbm_open)
TRAP(dbm_store)
TRAP(dirname)
// TRAP(dlerror) // It is not thread-safe. But it is used by dynamic linker to load some name resolution plugins. Also used by TSan.
/// Note: we should better get rid of glibc, dynamic linking and all that sort of annoying garbage altogether.
TRAP(ftw)
TRAP(getc_unlocked)
//TRAP(getenv) // Ok at program startup
TRAP(inet_ntoa)
TRAP(lgamma)
TRAP(lgammaf)
TRAP(lgammal)
TRAP(nftw)
TRAP(nl_langinfo)
TRAP(putc_unlocked)
//TRAP(rand) Used by nats-io at startup
/** In  the current POSIX.1 specification (POSIX.1-2008), readdir() is not required to be thread-safe.  However, in modern
  * implementations (including the glibc implementation), concurrent calls to readdir() that specify different directory streams
  * are thread-safe.  In cases where multiple threads must read from the same directory stream, using readdir() with external
  * synchronization is still preferable to the use of the deprecated readdir_r(3)  function. It is expected that a future
  * version of POSIX.1 will require that readdir() be thread-safe when concurrently employed on different directory streams.
  * - man readdir
  */
//TRAP(readdir)
TRAP(system)
TRAP(wcstombs)
TRAP(ether_aton)
TRAP(ether_ntoa)
TRAP(fgetsgent)
TRAP(fgetspent)
TRAP(getaliasbyname)
TRAP(getaliasent)
TRAP(getrpcbyname)
TRAP(getrpcbynumber)
TRAP(getrpcent)
TRAP(getsgent)
TRAP(getsgnam)
TRAP(getspent)
TRAP(getspnam)
TRAP(initstate)
TRAP(random)
TRAP(setstate)
TRAP(sgetsgent)
TRAP(sgetspent)
TRAP(srandom)
TRAP(twalk)
TRAP(lgammaf128)
TRAP(lgammaf32)
TRAP(lgammaf32x)
TRAP(lgammaf64)
TRAP(lgammaf64x)

/// These functions are unused by ClickHouse and we should be aware if they are accidentally get used.
/// Sometimes people report that these function contain vulnerabilities (these reports are bogus for ClickHouse).
TRAP(mq_close)
TRAP(mq_getattr)
TRAP(mq_setattr)
TRAP(mq_notify)
TRAP(mq_open)
TRAP(mq_receive)
TRAP(mq_send)
TRAP(mq_unlink)
TRAP(mq_timedsend)
TRAP(mq_timedreceive)

/// These functions are also unused by ClickHouse.
TRAP(wordexp)
TRAP(wordfree)

#endif
