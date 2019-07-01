// /*	$OpenBSD: readpassphrase.h,v 1.5 2003/06/17 21:56:23 millert Exp $	*/

/*
 * Copyright (c) 2000, 2002 Todd C. Miller <Todd.Miller@courtesan.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and Air Force Research Laboratory, Air Force
 * Materiel Command, USAF, under agreement number F39502-99-1-0512.
 */

/* OPENBSD ORIGINAL: include/readpassphrase.h */

#pragma once
// #ifndef _READPASSPHRASE_H_
// #define _READPASSPHRASE_H_

//#include "includes.h"
#include "config_client.h"

// Should not be included on BSD systems, but if it happen...
#ifdef HAVE_READPASSPHRASE
#   include_next <readpassphrase.h>
#endif

#ifndef HAVE_READPASSPHRASE

#    ifdef __cplusplus
extern "C" {
#    endif


#    define RPP_ECHO_OFF 0x00 /* Turn off echo (default). */
#    define RPP_ECHO_ON 0x01 /* Leave echo on. */
#    define RPP_REQUIRE_TTY 0x02 /* Fail if there is no tty. */
#    define RPP_FORCELOWER 0x04 /* Force input to lower case. */
#    define RPP_FORCEUPPER 0x08 /* Force input to upper case. */
#    define RPP_SEVENBIT 0x10 /* Strip the high bit from input. */
#    define RPP_STDIN 0x20 /* Read from stdin, not /dev/tty */

char * readpassphrase(const char *, char *, size_t, int);

#    ifdef __cplusplus
}
#    endif


#endif /* HAVE_READPASSPHRASE */

// #endif /* !_READPASSPHRASE_H_ */
