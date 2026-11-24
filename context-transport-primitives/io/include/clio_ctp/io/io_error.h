/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_CTP_IO_IO_ERROR_H_
#define CLIO_CTP_IO_IO_ERROR_H_

#include <cerrno>
#include <cstdint>

/**
 * Cross-platform classification of I/O errors.
 *
 * The async-I/O backends (libaio, POSIX AIO, io_uring, IOCP, NIXL) report
 * failures as an errno-valued integer in IoResult::error_code. This header maps
 * that raw value to a small, portable IoError category that callers can act on
 * uniformly — most importantly distinguishing a TRANSIENT hiccup (retry) from a
 * DEVICE FAULT (media error) from a DISCONNECTED device (removed/absent),
 * which is what storage layers like safe-bdev need in order to decide whether
 * to retry, fault a member, or reconstruct.
 *
 * The mapping is errno-based and guarded with #ifdef so it compiles on any
 * platform regardless of which errno constants that platform defines (POSIX and
 * the Windows UCRT both define the common ones). A Win32 GetLastError() helper
 * is also provided under _WIN32 for callers that hold a raw Win32 code rather
 * than an errno.
 */

namespace ctp {

/** Portable I/O error category. */
enum class IoError : uint32_t {
  kOk = 0,            /**< No error. */
  kTransient = 1,     /**< Retryable: EAGAIN/EWOULDBLOCK/EINTR/EBUSY. */
  kDeviceFault = 2,   /**< Media/controller error: EIO. */
  kDisconnected = 3,  /**< Device removed/absent: ENODEV/ENXIO/ENOENT. */
  kNoSpace = 4,       /**< Out of space/quota: ENOSPC/EDQUOT/EFBIG. */
  kPermission = 5,    /**< Access denied/read-only: EACCES/EPERM/EROFS. */
  kInvalid = 6,       /**< Bad request/handle: EINVAL/EBADF/EFAULT. */
  kTimeout = 7,       /**< Operation timed out: ETIMEDOUT. */
  kUnknown = 8        /**< Unrecognized error. */
};

/** Human-readable name for logging. */
inline const char *IoErrorName(IoError e) {
  switch (e) {
    case IoError::kOk:           return "Ok";
    case IoError::kTransient:    return "Transient";
    case IoError::kDeviceFault:  return "DeviceFault";
    case IoError::kDisconnected: return "Disconnected";
    case IoError::kNoSpace:      return "NoSpace";
    case IoError::kPermission:   return "Permission";
    case IoError::kInvalid:      return "Invalid";
    case IoError::kTimeout:      return "Timeout";
    case IoError::kUnknown:      return "Unknown";
  }
  return "Unknown";
}

/** True if the failure is worth retrying on the same device. */
inline bool IsTransient(IoError e) { return e == IoError::kTransient; }

/** True if the failure means the device should be faulted (gone or broken). */
inline bool IsFatalDevice(IoError e) {
  return e == IoError::kDeviceFault || e == IoError::kDisconnected;
}

/**
 * Map a POSIX errno value to an IoError. `error == 0` yields kOk.
 * Specific categories are checked before the generic transient bucket; cases are
 * #ifdef-guarded so the function compiles wherever a constant is absent.
 */
inline IoError ClassifyErrno(int error) {
  if (error == 0) {
    return IoError::kOk;
  }
#ifdef EIO
  if (error == EIO) return IoError::kDeviceFault;
#endif
#ifdef ENODEV
  if (error == ENODEV) return IoError::kDisconnected;
#endif
#ifdef ENXIO
  if (error == ENXIO) return IoError::kDisconnected;
#endif
#ifdef ENOENT
  if (error == ENOENT) return IoError::kDisconnected;  // backing file vanished
#endif
#ifdef ENOSPC
  if (error == ENOSPC) return IoError::kNoSpace;
#endif
#ifdef EDQUOT
  if (error == EDQUOT) return IoError::kNoSpace;
#endif
#ifdef EFBIG
  if (error == EFBIG) return IoError::kNoSpace;
#endif
#ifdef EACCES
  if (error == EACCES) return IoError::kPermission;
#endif
#ifdef EPERM
  if (error == EPERM) return IoError::kPermission;
#endif
#ifdef EROFS
  if (error == EROFS) return IoError::kPermission;
#endif
#ifdef ETIMEDOUT
  if (error == ETIMEDOUT) return IoError::kTimeout;
#endif
#ifdef EAGAIN
  if (error == EAGAIN) return IoError::kTransient;
#endif
#if defined(EWOULDBLOCK) && (!defined(EAGAIN) || EWOULDBLOCK != EAGAIN)
  if (error == EWOULDBLOCK) return IoError::kTransient;
#endif
#ifdef EINTR
  if (error == EINTR) return IoError::kTransient;
#endif
#ifdef EBUSY
  if (error == EBUSY) return IoError::kTransient;
#endif
#ifdef EINVAL
  if (error == EINVAL) return IoError::kInvalid;
#endif
#ifdef EBADF
  if (error == EBADF) return IoError::kInvalid;
#endif
#ifdef EFAULT
  if (error == EFAULT) return IoError::kInvalid;
#endif
  return IoError::kUnknown;
}

/** Classify the current value of `errno`. */
inline IoError ClassifyLastErrno() { return ClassifyErrno(errno); }

#if defined(_WIN32)
/**
 * Map a Win32 error code (GetLastError()) to an IoError, for callers holding a
 * raw Win32 code rather than an errno. Uses the numeric Win32 constants so this
 * header never needs to include <windows.h> (which would leak min/max/Yield/
 * SendMessage macros into every including TU).
 */
inline IoError ClassifyWinError(unsigned long error) {
  switch (error) {
    case 0UL:     // ERROR_SUCCESS
      return IoError::kOk;
    case 21UL:    // ERROR_NOT_READY
    case 170UL:   // ERROR_BUSY
      return IoError::kTransient;
    case 23UL:    // ERROR_CRC
    case 1117UL:  // ERROR_IO_DEVICE
      return IoError::kDeviceFault;
    case 2UL:     // ERROR_FILE_NOT_FOUND
    case 3UL:     // ERROR_PATH_NOT_FOUND
    case 55UL:    // ERROR_DEV_NOT_EXIST
    case 1167UL:  // ERROR_DEVICE_NOT_CONNECTED
      return IoError::kDisconnected;
    case 39UL:    // ERROR_DISK_FULL
    case 112UL:   // ERROR_DISK_FULL (alt)
      return IoError::kNoSpace;
    case 5UL:     // ERROR_ACCESS_DENIED
    case 19UL:    // ERROR_WRITE_PROTECT
      return IoError::kPermission;
    case 1460UL:  // ERROR_TIMEOUT
      return IoError::kTimeout;
    case 6UL:     // ERROR_INVALID_HANDLE
    case 87UL:    // ERROR_INVALID_PARAMETER
      return IoError::kInvalid;
    default:
      return IoError::kUnknown;
  }
}
#endif  // _WIN32

}  // namespace ctp

#endif  // CLIO_CTP_IO_IO_ERROR_H_
