/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * POSIX-shaped types for the filesystem client's descriptor layer.
 *
 * The descriptor layer (OpenFd/ReadFd/PwriteFd/SeekFd/...) is a descriptor
 * table and a set of seek offsets over the chimod's task API. It makes no
 * system calls: what tied it to POSIX was its vocabulary -- ssize_t, off_t,
 * O_SYNC, blksize_t -- not its behaviour. This header supplies that vocabulary
 * everywhere, so the layer builds on Windows as well.
 *
 * Two of those need care rather than a typedef:
 *
 *   off_t    MSVC defines it, as a 32-bit long, and it cannot be redefined.
 *            Using it would cap every offset in the layer at 2 GiB. The API
 *            therefore names FsOff, which is 64-bit on every platform; POSIX
 *            callers pass off_t and it converts.
 *
 *   O_SYNC   MSVC has neither O_SYNC nor O_DSYNC, and no caller there can ask
 *            for them, so IsSyncFd() reports "not requested" rather than
 *            pretending a flag bit exists.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef CLIO_CTE_FILESYSTEM_POSIX_COMPAT_H_
#define CLIO_CTE_FILESYSTEM_POSIX_COMPAT_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>

#if defined(_WIN32)
#include <BaseTsd.h>

/* MSVC has no ssize_t at all, so defining it is safe (unlike off_t). */
#if !defined(_SSIZE_T_DEFINED)
typedef SSIZE_T ssize_t;
#define _SSIZE_T_DEFINED
#endif

/* Only ever used for the casts in the synthesized stat; MSVC's struct stat has
 * no st_blksize/st_blocks, and the code that fills them is a template that is
 * never instantiated there. */
#if !defined(_BLKSIZE_T_DEFINED)
typedef long blksize_t;
typedef int64_t blkcnt_t;
#define _BLKSIZE_T_DEFINED
#endif
#endif /* _WIN32 */

namespace clio::cte::filesystem {

/** 64-bit file offset for the descriptor API. Deliberately not off_t: MSVC's
 * is 32 bits, and on POSIX this is the same width as off_t already, so callers
 * on either platform pass their native type unchanged. */
using FsOff = int64_t;

/** Byte count / error return for the descriptor API, mirroring ssize_t. */
using FsSsize = int64_t;

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_POSIX_COMPAT_H_
