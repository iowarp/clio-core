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

#ifndef CLIO_CTE_ADAPTER_LIBFUSE_FUSE_WIN_COMPAT_H_
#define CLIO_CTE_ADAPTER_LIBFUSE_FUSE_WIN_COMPAT_H_

#ifdef _WIN32

// Windows (WinFsp) compatibility shim for the cross-platform FUSE adapter.
//
// This header is included by fuse_cte.cc *after* <fuse3/fuse.h> (WinFsp's
// FUSE3-compatibility header, pulled in transitively via fuse_cte.h). On
// MSVC, WinFsp exposes the FUSE data types under fuse_-prefixed names
// (fuse_stat, fuse_off_t, fuse_mode_t, fuse_timespec) and does NOT remap the
// bare POSIX spellings the way it does for Cygwin/MinGW. It also does not
// provide getuid()/getgid() or the S_IF* mode-bit macros, and <unistd.h> /
// <sys/stat.h> are not available. This header bridges exactly those gaps so
// the platform-agnostic callback bodies in fuse_cte.cc compile unchanged.
//
// Nothing here changes the CTE data path: it only makes the FUSE frontend's
// type names and helpers resolve on Windows. The matching Linux definitions
// (the bare POSIX types) live inline in fuse_cte.cc.

// --- Type bridge -----------------------------------------------------------
// The shared callback signatures must match the function-pointer types in
// WinFsp's `struct fuse_operations` exactly, so each POSIX spelling maps to
// the corresponding fuse_* type. The callback bodies only touch fields that
// fuse_stat shares with POSIX struct stat (st_mode/st_nlink/st_uid/st_gid/
// st_size), so they need no further change.
using cte_stat_t = struct fuse_stat;
using cte_off_t = fuse_off_t;
using cte_mode_t = fuse_mode_t;
using cte_timespec_t = struct fuse_timespec;
// statfs() reports through struct fuse_statvfs on WinFsp; the callback param
// uses cte_statvfs_t so its signature matches fuse_operations::statfs on both
// platforms (the bare `struct statvfs` spelling is not declared on MSVC).
using cte_statvfs_t = struct fuse_statvfs;

// Integer field types the shared callback bodies spell the POSIX way, but which
// WinFsp only provides under fuse_-prefixed names (MSVC has no <sys/stat.h> /
// <sys/statvfs.h> to declare the bare spellings).
using nlink_t = fuse_nlink_t;
using fsblkcnt_t = fuse_fsblkcnt_t;
using fsfilcnt_t = fuse_fsfilcnt_t;

// Owner id types the getattr/chown callbacks spell the POSIX way; WinFsp only
// declares them fuse_-prefixed. The .chown slot in struct fuse_operations is
// typed int(const char*, fuse_uid_t, fuse_gid_t, fuse_file_info*), so aliasing
// to the fuse_ types keeps cte_fuse_chown's signature matching the slot.
using uid_t = fuse_uid_t;
using gid_t = fuse_gid_t;

// --- File mode bits --------------------------------------------------------
// POSIX octal values; fuse_stat::st_mode uses the same encoding. Guarded in
// case a future WinFsp header begins providing them.
#ifndef S_IFMT
#define S_IFMT 0170000
#endif
#ifndef S_IFDIR
#define S_IFDIR 0040000
#endif
#ifndef S_IFREG
#define S_IFREG 0100000
#endif
#ifndef S_IFLNK
#define S_IFLNK 0120000
#endif

// --- Owner identity --------------------------------------------------------
// Windows has no POSIX uid/gid; WinFsp fabricates a mapping per request.
// getattr only uses these to populate st_uid/st_gid for display, so report
// the WinFsp-supplied context identity when a request is in flight, else 0.
static inline unsigned cte_fuse_getuid() {
  struct fuse_context *ctx = fuse_get_context();
  return ctx ? static_cast<unsigned>(ctx->uid) : 0u;
}
static inline unsigned cte_fuse_getgid() {
  struct fuse_context *ctx = fuse_get_context();
  return ctx ? static_cast<unsigned>(ctx->gid) : 0u;
}
#ifndef getuid
#define getuid cte_fuse_getuid
#endif
#ifndef getgid
#define getgid cte_fuse_getgid
#endif

#endif  // _WIN32
#endif  // CLIO_CTE_ADAPTER_LIBFUSE_FUSE_WIN_COMPAT_H_
