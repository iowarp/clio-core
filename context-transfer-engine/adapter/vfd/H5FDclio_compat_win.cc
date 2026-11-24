/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Win32 implementations of the CLIO VFD's platform layer.
 *
 * This is the ONLY translation unit in the driver that includes <windows.h>.
 * That is deliberate: windows.h defines Yield(), min/max, GetMessage and other
 * ordinary-looking identifiers as macros, and clio's own headers use several of
 * them as member names (clio_ctp/thread/thread_model/std_thread.h declares
 * `void Yield()`). Including it from H5FDclio_compat.h, which H5FDclio.cc pulls
 * in ahead of those headers, breaks their parse in ways that look nothing like
 * the real cause. Keeping it here means the driver body never sees it.
 *
 * The rationale for each mapping is in H5FDclio_compat.h.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
/* windows.h first: <io.h> and friends expect its types. */
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <string.h>
#include <sys/stat.h>

#include "H5FDclio_compat.h"

namespace {

/* Win32 reports failure through GetLastError(), but the driver's error macro
 * prints errno. Map the cases the driver distinguishes; everything else is
 * EIO. */
void SetErrnoFromWin32(DWORD err) {
  switch (err) {
    case ERROR_SUCCESS:
      errno = 0;
      break;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      errno = ENOENT;
      break;
    case ERROR_ACCESS_DENIED:
      errno = EACCES;
      break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      errno = EEXIST;
      break;
    /* Another handle holds the lock. flock(LOCK_NB) reports contention as
     * EWOULDBLOCK, and the driver names that case specifically. */
    case ERROR_LOCK_VIOLATION:
    case ERROR_SHARING_VIOLATION:
      errno = EWOULDBLOCK;
      break;
    case ERROR_DISK_FULL:
      errno = ENOSPC;
      break;
    case ERROR_INVALID_HANDLE:
      errno = EBADF;
      break;
    case ERROR_NOT_SUPPORTED:
      errno = ENOSYS;
      break;
    default:
      errno = EIO;
      break;
  }
}

HANDLE HandleOf(int fd) {
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
  }
  return h;
}

/* A single ReadFile/WriteFile takes a DWORD count; the driver already loops for
 * short transfers, so capping here is just another short transfer to it. */
DWORD ClampCount(size_t count) {
  const size_t kMax = 0x7fffffffu;
  return static_cast<DWORD>(count > kMax ? kMax : count);
}

OVERLAPPED OverlappedAt(clio_vfd_off_t off) {
  OVERLAPPED ov;
  memset(&ov, 0, sizeof(ov));
  LARGE_INTEGER li;
  li.QuadPart = off;
  ov.Offset = li.LowPart;
  ov.OffsetHigh = static_cast<DWORD>(li.HighPart);
  return ov;
}

}  // namespace

int clio_vfd_open(const char *path, int o_flags, int mode) {
  /* _O_BINARY is mandatory: without it the CRT rewrites \n as \r\n and
   * silently corrupts the file. _SH_DENYNO keeps the file readable by other
   * tools while it is open, as open(2) does -- the driver's "h5dump can read
   * it live" property depends on that. */
  int fd = -1;
  errno_t e = _sopen_s(&fd, path, o_flags | _O_BINARY, _SH_DENYNO, mode);
  if (e != 0) {
    errno = e;
    return -1;
  }
  return fd;
}

int clio_vfd_close(int fd) { return _close(fd); }

clio_vfd_ssize_t clio_vfd_pread(int fd, void *buf, size_t count,
                                clio_vfd_off_t off) {
  HANDLE h = HandleOf(fd);
  if (h == INVALID_HANDLE_VALUE) {
    return -1;
  }
  OVERLAPPED ov = OverlappedAt(off);
  DWORD got = 0;
  if (!ReadFile(h, buf, ClampCount(count), &got, &ov)) {
    DWORD e = GetLastError();
    /* Reading at or past EOF: pread reports 0, and the driver zero-fills the
     * remainder on exactly that signal. */
    if (e == ERROR_HANDLE_EOF) {
      return 0;
    }
    SetErrnoFromWin32(e);
    return -1;
  }
  return static_cast<clio_vfd_ssize_t>(got);
}

clio_vfd_ssize_t clio_vfd_pwrite(int fd, const void *buf, size_t count,
                                 clio_vfd_off_t off) {
  HANDLE h = HandleOf(fd);
  if (h == INVALID_HANDLE_VALUE) {
    return -1;
  }
  OVERLAPPED ov = OverlappedAt(off);
  DWORD put = 0;
  if (!WriteFile(h, buf, ClampCount(count), &put, &ov)) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  return static_cast<clio_vfd_ssize_t>(put);
}

int clio_vfd_ftruncate(int fd, clio_vfd_off_t length) {
  /* _chsize_s takes the 64-bit length and returns an errno value rather than
   * setting it. */
  errno_t e = _chsize_s(fd, static_cast<__int64>(length));
  if (e != 0) {
    errno = e;
    return -1;
  }
  return 0;
}

int clio_vfd_fsync(int fd) { return _commit(fd); }

int clio_vfd_lock(int fd, int exclusive) {
  HANDLE h = HandleOf(fd);
  if (h == INVALID_HANDLE_VALUE) {
    return -1;
  }
  DWORD flags = LOCKFILE_FAIL_IMMEDIATELY; /* flock's LOCK_NB */
  if (exclusive) {
    flags |= LOCKFILE_EXCLUSIVE_LOCK;
  }
  OVERLAPPED ov = OverlappedAt(0);
  if (!LockFileEx(h, flags, 0, MAXDWORD, MAXDWORD, &ov)) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  return 0;
}

int clio_vfd_unlock(int fd) {
  HANDLE h = HandleOf(fd);
  if (h == INVALID_HANDLE_VALUE) {
    return -1;
  }
  OVERLAPPED ov = OverlappedAt(0);
  if (!UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov)) {
    DWORD e = GetLastError();
    /* The driver unlocks on paths where the lock may never have been taken,
     * and flock(LOCK_UN) is equally forgiving of that. */
    if (e == ERROR_NOT_LOCKED) {
      return 0;
    }
    SetErrnoFromWin32(e);
    return -1;
  }
  return 0;
}

int clio_vfd_fstat(int fd, clio_vfd_file_id_t *id, clio_vfd_off_t *size) {
  HANDLE h = HandleOf(fd);
  if (h == INVALID_HANDLE_VALUE) {
    return -1;
  }
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(h, &info)) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  /* Identity as H5FDsec2.c computes it: MSVC's st_ino is always 0. */
  id->volume_serial = info.dwVolumeSerialNumber;
  id->index_high = info.nFileIndexHigh;
  id->index_low = info.nFileIndexLow;
  LARGE_INTEGER li;
  li.LowPart = info.nFileSizeLow;
  li.HighPart = static_cast<LONG>(info.nFileSizeHigh);
  *size = static_cast<clio_vfd_off_t>(li.QuadPart);
  return 0;
}

int clio_vfd_unlink(const char *path) { return _unlink(path); }

#endif /* _WIN32 */
