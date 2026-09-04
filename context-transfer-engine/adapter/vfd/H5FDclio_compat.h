/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Platform layer for the CLIO HDF5 VFD.
 *
 * The driver keeps an authoritative native file on disk and performs ordinary
 * file I/O on it. That I/O is POSIX in H5FDclio.cc; this header supplies the
 * same operations on Windows, so the driver body reads identically on both
 * platforms and the port stays reviewable.
 *
 * Where Windows differs from POSIX in a way that matters, the choice here is
 * the one HDF5's own sec2 driver (H5FDsec2.c) makes:
 *
 *   file identity   MSVC reports st_ino == 0 for every file, so dev/ino cannot
 *                   tell two files apart. Identity comes from the volume serial
 *                   number plus the NTFS file index, as in H5FDsec2.c. cmp()
 *                   depends on this: if it says two opens of one file are
 *                   different files, HDF5 opens it twice with independent
 *                   metadata caches and corrupts it.
 *
 *   positional I/O  There is no pread/pwrite. ReadFile/WriteFile with an
 *                   OVERLAPPED offset are the real equivalent -- they do not
 *                   disturb the shared file pointer, so concurrent positional
 *                   reads stay correct. Seek-then-read would not.
 *
 *   binary mode     _O_BINARY is not optional. Without it the CRT rewrites
 *                   \n as \r\n on the way out, which corrupts an HDF5 file
 *                   silently and unrecoverably.
 *
 *   locking         flock() has no Windows equivalent; LockFileEx over the
 *                   whole range is the standard stand-in, with
 *                   LOCKFILE_FAIL_IMMEDIATELY for flock's LOCK_NB.
 *
 *   errno           Win32 reports through GetLastError(), but the driver's
 *                   error macro prints errno. Every failure path below sets
 *                   errno so those messages stay meaningful.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef CLIO_CTE_ADAPTER_VFD_H5FDCLIO_COMPAT_H_
#define CLIO_CTE_ADAPTER_VFD_H5FDCLIO_COMPAT_H_

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>

#ifndef _WIN32
#include <sys/file.h>
#include <unistd.h>
#endif

/* NOTE: <windows.h> is deliberately NOT included here. It defines Yield(),
 * min/max, GetMessage and friends as macros, and this header is pulled into
 * H5FDclio.cc ahead of clio's own headers -- where `void Yield()` in
 * clio_ctp/thread/thread_model/std_thread.h then fails to parse. The Win32
 * implementations live in H5FDclio_compat_win.cc, which is the only
 * translation unit that sees <windows.h>. */

/* 64-bit file offset on every platform. MSVC's off_t is 32 bits, which would
 * silently cap the driver's usable address space -- and MAXADDR, which is
 * derived from sizeof() this type -- at 2 GiB. */
typedef int64_t clio_vfd_off_t;
typedef int64_t clio_vfd_ssize_t;

/* Filesystem identity of the authoritative native file; see the note above. */
typedef struct clio_vfd_file_id_t {
#ifdef _WIN32
  uint32_t volume_serial; /* dwVolumeSerialNumber */
  uint32_t index_high;    /* nFileIndexHigh */
  uint32_t index_low;     /* nFileIndexLow */
#else
  dev_t dev;
  ino_t ino;
#endif
} clio_vfd_file_id_t;


#ifdef _WIN32
int clio_vfd_open(const char *path, int o_flags, int mode);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_open(const char *path, int o_flags, int mode) {
  return open(path, o_flags, mode);
}
#endif

#ifdef _WIN32
int clio_vfd_close(int fd);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_close(int fd) {
  return close(fd);
}
#endif

#ifdef _WIN32
clio_vfd_ssize_t clio_vfd_pread(int fd, void *buf, size_t count,
                                              clio_vfd_off_t off);   /* H5FDclio_compat_win.cc */
#else
static inline clio_vfd_ssize_t clio_vfd_pread(int fd, void *buf, size_t count,
                                              clio_vfd_off_t off) {
  return (clio_vfd_ssize_t)pread(fd, buf, count, (off_t)off);
}
#endif

#ifdef _WIN32
clio_vfd_ssize_t clio_vfd_pwrite(int fd, const void *buf,
                                               size_t count,
                                               clio_vfd_off_t off);   /* H5FDclio_compat_win.cc */
#else
static inline clio_vfd_ssize_t clio_vfd_pwrite(int fd, const void *buf,
                                               size_t count,
                                               clio_vfd_off_t off) {
  return (clio_vfd_ssize_t)pwrite(fd, buf, count, (off_t)off);
}
#endif

#ifdef _WIN32
int clio_vfd_ftruncate(int fd, clio_vfd_off_t length);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_ftruncate(int fd, clio_vfd_off_t length) {
  return ftruncate(fd, (off_t)length);
}
#endif

#ifdef _WIN32
int clio_vfd_fsync(int fd);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_fsync(int fd) {
  return fsync(fd);
}
#endif

/* Non-blocking advisory whole-file lock. Returns 0, or -1 with errno set;
 * EWOULDBLOCK means another handle holds it. */
#ifdef _WIN32
int clio_vfd_lock(int fd, int exclusive);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_lock(int fd, int exclusive) {
  int op = (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
  return flock(fd, op);
}
#endif

#ifdef _WIN32
int clio_vfd_unlock(int fd);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_unlock(int fd) {
  return flock(fd, LOCK_UN);
}
#endif

/* One call for both things the driver needs at open: the identity cmp() will
 * compare, and the current size it uses as EOF. */
#ifdef _WIN32
int clio_vfd_fstat(int fd, clio_vfd_file_id_t *id,
                                 clio_vfd_off_t *size);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_fstat(int fd, clio_vfd_file_id_t *id,
                                 clio_vfd_off_t *size) {
  struct stat st;
  if (fstat(fd, &st) < 0) {
    return -1;
  }
  id->dev = st.st_dev;
  id->ino = st.st_ino;
  *size = (clio_vfd_off_t)st.st_size;
  return 0;
}
#endif

/* Ordering comparison for cmp(): -1 / 0 / 1. */
static inline int clio_vfd_cmp_file_id(const clio_vfd_file_id_t *a,
                                       const clio_vfd_file_id_t *b) {
#ifdef _WIN32
  if (a->volume_serial < b->volume_serial) return -1;
  if (a->volume_serial > b->volume_serial) return 1;
  if (a->index_high < b->index_high) return -1;
  if (a->index_high > b->index_high) return 1;
  if (a->index_low < b->index_low) return -1;
  if (a->index_low > b->index_low) return 1;
#else
  if (a->dev < b->dev) return -1;
  if (a->dev > b->dev) return 1;
  if (a->ino < b->ino) return -1;
  if (a->ino > b->ino) return 1;
#endif
  return 0;
}

#ifdef _WIN32
int clio_vfd_unlink(const char *path);   /* H5FDclio_compat_win.cc */
#else
static inline int clio_vfd_unlink(const char *path) {
  return unlink(path);
}
#endif

#endif /* CLIO_CTE_ADAPTER_VFD_H5FDCLIO_COMPAT_H_ */
