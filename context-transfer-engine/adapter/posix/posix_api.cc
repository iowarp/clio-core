/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * POSIX interceptor. Tracked paths (those carrying the "clio::" marker) and
 * the descriptors opened from them are routed to the context-filesystem
 * chimod via the shared CfsIo core (CLIO_CTE_CFS); everything else falls
 * through to the real libc API.
 */

// Dynamically checked to see which are the real APIs and which are intercepted
bool posix_intercepted = true;

#include "posix_api.h"

#include <fcntl.h>
#include <stdarg.h>
#include <sys/stat.h>

#include "clio_runtime/clio_runtime.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/singleton.h"
#include "adapter/cfs/cfs_io.h"

namespace clio::cae {
// Define global pointer variable in source file
CTP_DEFINE_GLOBAL_PTR_VAR_CC(PosixApi, g_posix_api);

/** Used for compatability with older kernel versions */
int fxstat_to_fstat(int fd, struct stat *stbuf) {
#ifdef _STAT_VER
  return CLIO_CTE_POSIX_API->__fxstat(_STAT_VER, fd, stbuf);
#else
  (void)fd;
  (void)stbuf;
  return -1;
#endif
}
}  // namespace clio::cae

extern "C" {

/**
 * POSIX
 */
int CLIO_CTE_DECL(open)(const char *path, int flags, ...) {
  int mode = 0;
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (flags & O_CREAT || flags & O_TMPFILE) {
    va_list arg;
    va_start(arg, flags);
    mode = va_arg(arg, int);
    va_end(arg);
  }
  if (real_api->IsInterceptorLoaded() &&
      clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercept open for filename: {} flags: {}", path, flags);
    return cfs->Open(path, flags, mode);
  }
  if (flags & O_CREAT || flags & O_TMPFILE) {
    return real_api->open(path, flags, mode);
  }
  return real_api->open(path, flags);
}

int CLIO_CTE_DECL(open64)(const char *path, int flags, ...) {
  int mode = 0;
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (flags & O_CREAT || flags & O_TMPFILE) {
    va_list arg;
    va_start(arg, flags);
    mode = va_arg(arg, int);
    va_end(arg);
  }
  if (real_api->IsInterceptorLoaded() &&
      clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercept open64 for filename: {} flags: {}", path, flags);
    return cfs->Open(path, flags, mode);
  }
  if (flags & O_CREAT || flags & O_TMPFILE) {
    return real_api->open64(path, flags, mode);
  }
  return real_api->open64(path, flags);
}

int CLIO_CTE_DECL(__open_2)(const char *path, int oflag) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (real_api->IsInterceptorLoaded() &&
      clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercept __open_2 for filename: {} flags: {}", path, oflag);
    return cfs->Open(path, oflag, 0);
  }
  return real_api->__open_2(path, oflag);
}

int CLIO_CTE_DECL(creat)(const char *path, mode_t mode) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (real_api->IsInterceptorLoaded() &&
      clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercept creat for filename: {} mode: {}", path, mode);
    return cfs->Open(path, O_CREAT | O_WRONLY | O_TRUNC, mode);
  }
  return real_api->creat(path, mode);
}

int CLIO_CTE_DECL(creat64)(const char *path, mode_t mode) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (real_api->IsInterceptorLoaded() &&
      clio::cae::CfsIo::IsPathTracked(path)) {
    HLOG(kDebug, "Intercept creat64 for filename: {} mode: {}", path, mode);
    return cfs->Open(path, O_CREAT | O_WRONLY | O_TRUNC, mode);
  }
  return real_api->creat64(path, mode);
}

ssize_t CLIO_CTE_DECL(read)(int fd, void *buf, size_t count) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept read.");
    return cfs->Read(fd, buf, count);
  }
  return real_api->read(fd, buf, count);
}

ssize_t CLIO_CTE_DECL(write)(int fd, const void *buf, size_t count) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept write.");
    return cfs->Write(fd, buf, count);
  }
  return real_api->write(fd, buf, count);
}

ssize_t CLIO_CTE_DECL(pread)(int fd, void *buf, size_t count, off_t offset) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept pread.");
    return cfs->Pread(fd, buf, count, offset);
  }
  return real_api->pread(fd, buf, count, offset);
}

ssize_t CLIO_CTE_DECL(pwrite)(int fd, const void *buf, size_t count,
                              off_t offset) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept pwrite.");
    return cfs->Pwrite(fd, buf, count, offset);
  }
  return real_api->pwrite(fd, buf, count, offset);
}

ssize_t CLIO_CTE_DECL(pread64)(int fd, void *buf, size_t count,
                               off64_t offset) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept pread64.");
    return cfs->Pread(fd, buf, count, static_cast<off_t>(offset));
  }
  return real_api->pread64(fd, buf, count, offset);
}

ssize_t CLIO_CTE_DECL(pwrite64)(int fd, const void *buf, size_t count,
                                off64_t offset) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept pwrite64.");
    return cfs->Pwrite(fd, buf, count, static_cast<off_t>(offset));
  }
  return real_api->pwrite64(fd, buf, count, offset);
}

off_t CLIO_CTE_DECL(lseek)(int fd, off_t offset, int whence) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept lseek offset: {} whence: {}.", offset, whence);
    return cfs->Seek(fd, offset, whence);
  }
  return real_api->lseek(fd, offset, whence);
}

off64_t CLIO_CTE_DECL(lseek64)(int fd, off64_t offset, int whence) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercept lseek64 offset: {} whence: {}.", offset, whence);
    return cfs->Seek(fd, static_cast<off_t>(offset), whence);
  }
  return real_api->lseek64(fd, offset, whence);
}

int CLIO_CTE_DECL(__fxstat)(int __ver, int fd, struct stat *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted __fxstat.");
    return cfs->StatFd(fd, buf);
  }
  return real_api->__fxstat(__ver, fd, buf);
}

int CLIO_CTE_DECL(__fxstatat)(int __ver, int __fildes, const char *__filename,
                              struct stat *__stat_buf, int __flag) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    HLOG(kDebug, "Intercepted __fxstatat.");
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__fxstatat(__ver, __fildes, __filename, __stat_buf, __flag);
}

int CLIO_CTE_DECL(__xstat)(int __ver, const char *__filename,
                           struct stat *__stat_buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__xstat(__ver, __filename, __stat_buf);
}

int CLIO_CTE_DECL(__lxstat)(int __ver, const char *__filename,
                            struct stat *__stat_buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    HLOG(kDebug, "Intercepted __lxstat.");
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__lxstat(__ver, __filename, __stat_buf);
}

int CLIO_CTE_DECL(fstat)(int fd, struct stat *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted fstat.");
    return cfs->StatFd(fd, buf);
  }
  return real_api->fstat(fd, buf);
}

int CLIO_CTE_DECL(stat)(const char *pathname, struct stat *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(pathname)) {
    HLOG(kDebug, "Intercepted stat.");
    return cfs->StatPath(pathname, buf);
  }
  return real_api->stat(pathname, buf);
}

int CLIO_CTE_DECL(__fxstat64)(int __ver, int fd, struct stat64 *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted __fxstat64.");
    return cfs->StatFd(fd, buf);
  }
  return real_api->__fxstat64(__ver, fd, buf);
}

int CLIO_CTE_DECL(__fxstatat64)(int __ver, int __fildes, const char *__filename,
                                struct stat64 *__stat_buf, int __flag) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    HLOG(kDebug, "Intercepted __fxstatat64.");
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__fxstatat64(__ver, __fildes, __filename, __stat_buf,
                                __flag);
}

int CLIO_CTE_DECL(__xstat64)(int __ver, const char *__filename,
                             struct stat64 *__stat_buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    HLOG(kDebug, "Intercepted __xstat64.");
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__xstat64(__ver, __filename, __stat_buf);
}

int CLIO_CTE_DECL(__lxstat64)(int __ver, const char *__filename,
                              struct stat64 *__stat_buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(__filename)) {
    HLOG(kDebug, "Intercepted __lxstat64.");
    return cfs->StatPath(__filename, __stat_buf);
  }
  return real_api->__lxstat64(__ver, __filename, __stat_buf);
}

int CLIO_CTE_DECL(fstat64)(int fd, struct stat64 *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted fstat64.");
    return cfs->StatFd(fd, buf);
  }
  return real_api->fstat64(fd, buf);
}

int CLIO_CTE_DECL(stat64)(const char *pathname, struct stat64 *buf) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(pathname)) {
    HLOG(kDebug, "Intercepted stat64.");
    return cfs->StatPath(pathname, buf);
  }
  return real_api->stat64(pathname, buf);
}

int CLIO_CTE_DECL(fsync)(int fd) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted fsync.");
    return cfs->Sync(fd);
  }
  return real_api->fsync(fd);
}

int CLIO_CTE_DECL(ftruncate)(int fd, off_t length) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted ftruncate.");
    return cfs->FtruncateFd(fd, length);
  }
  return real_api->ftruncate(fd, length);
}

int CLIO_CTE_DECL(ftruncate64)(int fd, off64_t length) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted ftruncate64.");
    return cfs->FtruncateFd(fd, static_cast<off_t>(length));
  }
  return real_api->ftruncate64(fd, length);
}

int CLIO_CTE_DECL(close)(int fd) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted close({}).", fd);
    return cfs->Close(fd);
  }
  return real_api->close(fd);
}

int CLIO_CTE_DECL(flock)(int fd, int operation) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (cfs->IsFdTracked(fd)) {
    HLOG(kDebug, "Intercepted flock({}).", fd);
    return 0;  // advisory locks are a no-op for clio:: files
  }
  return real_api->flock(fd, operation);
}

int CLIO_CTE_DECL(remove)(const char *pathname) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(pathname)) {
    HLOG(kDebug, "Intercepted remove({})", pathname);
    return cfs->RemovePath(pathname);
  }
  return real_api->remove(pathname);
}

int CLIO_CTE_DECL(unlink)(const char *pathname) {
  auto real_api = CLIO_CTE_POSIX_API;
  auto cfs = CLIO_CTE_CFS;
  if (clio::cae::CfsIo::IsPathTracked(pathname)) {
    HLOG(kDebug, "Intercepted unlink({})", pathname);
    return cfs->RemovePath(pathname);
  }
  return real_api->unlink(pathname);
}

}  // extern C
