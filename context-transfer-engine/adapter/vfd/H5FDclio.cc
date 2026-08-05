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

/*
 * Programmer:  Kimmy Mu
 *              March 2021
 *
 * Purpose: An HDF5 Virtual File Driver that writes every byte through to an
 *          authoritative on-disk native HDF5 file (so standard tools read it
 *          live), while opening a CLIO CTE handle alongside as groundwork for a
 *          future read/tiering cache.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* HDF5 header for dynamic plugin loading */
#include "H5FDclio.h" /* Clio file driver     */
#include "H5PLextern.h"
#include "adapter/cfs/cfs_io.h"
#include "clio_cte/core/core_client.h"
#include <clio_ctp/util/logging.h>

/* The driver identification number, initialized at runtime */
static hid_t H5FD_CLIO_g = H5I_INVALID_HID;

/* Identifiers for HDF5's error API */
hid_t H5FDclio_err_class_g = H5I_INVALID_HID;
static hid_t H5FDclio_err_major_g = H5I_INVALID_HID;
static hid_t H5FDclio_err_minor_g = H5I_INVALID_HID;

/* Observability: number of times the vector callbacks have run. Exported (not
 * static) so a test can confirm HDF5 actually took the vector I/O path. */
unsigned long H5FDclio_read_vector_calls_g = 0;
unsigned long H5FDclio_write_vector_calls_g = 0;

/* Observability: CTE cache-tier operations that FAILED. The cache is populate-
 * only today, so a failure is survivable (the authoritative native file already
 * has the bytes) -- but it must not be invisible: once the read tier lands, a
 * dropped Pwrite is a range the cache believes it holds and does not, which is
 * exactly the stale-data hazard VFD_2.1_READ_CACHE_SCOPING.md is written to
 * contain. Counted and logged here so residency work has a signal to build on.
 * Exported (not static) so tests can assert on them. */
unsigned long H5FDclio_cache_write_failures_g = 0;
unsigned long H5FDclio_cache_truncate_failures_g = 0;

/* Push a driver error onto HDF5's default error stack. Callbacks still return
 * FAIL/NULL to signal the failure to the library; this records a diagnosable
 * reason (incl. errno) that composes with HDF5's own stack and is surfaced by
 * the normal H5Eprint auto-handler at the API boundary -- instead of failing
 * silently. No-op until the class is registered by H5FD_clio_init(). */
#define H5FD_CLIO_ERROR(msg)                                               \
  do {                                                                     \
    if (H5FDclio_err_class_g >= 0) {                                       \
      H5Epush2(H5E_DEFAULT, __FILE__, __func__, __LINE__,                  \
               H5FDclio_err_class_g, H5FDclio_err_major_g,                 \
               H5FDclio_err_minor_g, "%s (errno=%d: %s)", (msg), errno,    \
               strerror(errno));                                           \
    }                                                                      \
  } while (0)

/* POSIX I/O mode used as the third parameter to open/_open
 * when creating a new file (O_CREAT is set). */
#if defined(H5_HAVE_WIN32_API)
#define H5FD_CLIO_POSIX_CREATE_MODE_RW (_S_IREAD | _S_IWRITE)
#else
#define H5FD_CLIO_POSIX_CREATE_MODE_RW 0666
#endif

#define MAXADDR (((haddr_t)1 << (8 * sizeof(off_t) - 1)) - 1)
#define SUCCEED 0
#define FAIL (-1)

/* Largest byte count handed to a single pread/pwrite. Linux caps a single
 * transfer at 0x7ffff000 and returns a SHORT count above it, so an unlooped
 * call would silently transfer less than asked. sec2 chunks for the same
 * reason (H5_POSIX_MAX_IO_BYTES); 1 GiB is comfortably under every platform
 * limit and keeps the loop count trivial for realistic HDF5 requests.
 *
 * CLIO_VFD_MAX_IO_BYTES overrides it. This exists so the multi-pass path can be
 * exercised with ordinary kilobyte-sized transfers instead of requiring a
 * multi-gigabyte one: the splitting/resume logic is identical at any threshold,
 * and a test that needs 2 GiB of disk and memory to run is a test that does not
 * get run. Read once, on first use. */
static size_t H5FD__clio_max_io_bytes(void) {
  static const size_t limit = []() -> size_t {
    const char *v = getenv("CLIO_VFD_MAX_IO_BYTES");
    if (v && *v) {
      unsigned long long n = strtoull(v, nullptr, 10);
      if (n > 0) {
        return (size_t)n;
      }
    }
    return (size_t)1 << 30;
  }();
  return limit;
}

/* Attach to the CLIO runtime, at most once per process, and remember the
 * answer.
 *
 * Attaching to an absent runtime is not cheap: the client retries for
 * CLIO_CLIENT_RETRY_TIMEOUT seconds (60 by default) before giving up. Asking
 * again on every H5Fopen would make a down runtime cost that timeout per file
 * rather than once per process -- a workload opening a hundred files would
 * appear to hang. Cache it.
 *
 * Consequence, accepted deliberately: a runtime that starts AFTER the first
 * open is not picked up for the life of the process. Files stay native-only,
 * which is correct, just unaccelerated. */
static bool H5FD__clio_cache_available(void) {
  static const bool available = clio::cte::core::CLIO_CTE_CLIENT_INIT();
  return available;
}

/* Read the CLIO_VFD_CACHE opt-out. Default on; "0"/"off"/"false"/"no" disable
 * the CTE tier and make the driver pure write-through to the native file.
 *
 * Deliberately identical in name-shape and accepted values to the VOL's
 * CLIO_VOL_CACHE, because the two connectors should not need different muscle
 * memory to switch off the same thing.
 *
 * Why an env var when H5Pset_fapl_clio(fapl, false) already exists: the FAPL
 * property requires editing and recompiling the application, and the entire
 * deployment story for this driver is HDF5_DRIVER=clio_vfd with NO source
 * edits. A switch reachable only from source is not reachable at all for the
 * workloads this driver targets. §1.4 of VFD_VOL_TECHNICAL_GOALS.md asks for
 * exactly this shape -- "one env var and one FAPL property, either of which is
 * sufficient" -- and only the FAPL half existed.
 *
 * Precedence: the env var is an opt-OUT only. It can force the cache off, but
 * it never forces it on over an explicit H5Pset_fapl_clio(fapl, false). "Either
 * is sufficient" means either is sufficient to DISABLE, which is the
 * fail-closed direction; a caller that asked for native-only in code must not
 * have the cache switched back on underneath by an environment it did not set.
 */
static bool H5FD__clio_cache_env_enabled(void) {
  const char *v = getenv("CLIO_VFD_CACHE");
  if (!v || !*v) return true;
  return !(strcmp(v, "0") == 0 || strcmp(v, "off") == 0 ||
           strcmp(v, "false") == 0 || strcmp(v, "no") == 0);
}

/* True when addr/size cannot be expressed as a POSIX file region: an undefined
 * address, an address past the driver's advertised maxaddr, or a length that
 * wraps when added to the address. sec2 performs the equivalent checks; without
 * them the casts below silently produce a nonsense off_t. */
#define H5FD_CLIO_REGION_INVALID(addr, size)                               \
  (HADDR_UNDEF == (addr) || (addr) > MAXADDR ||                            \
   (haddr_t)(size) > (haddr_t)(MAXADDR - (addr)))

#ifdef __cplusplus
extern "C" {
#endif

/* Driver-specific file access properties: the tiering policy an application can
 * set via its FAPL. POD (no pointers), so copy/free are trivial. Extensible --
 * more knobs (e.g. cache page size / tiering policy) can be added when the CTE
 * read tier makes them meaningful. */
typedef struct H5FD_clio_fapl_t {
  hbool_t cache_enabled; /* populate the CTE cache tier (default on) */
} H5FD_clio_fapl_t;

/* Default policy when a file is opened without a driver-specific FAPL
 * (e.g. H5Pset_driver(fapl, driver, NULL)): cache on. */
static const H5FD_clio_fapl_t H5FD_clio_fapl_default_g = {/*cache_enabled*/ 1};

/* The description of a file belonging to this driver. */
typedef struct H5FD_clio_t {
  H5FD_t pub;         /* public stuff, must be first           */
  haddr_t eoa;        /* end of allocated region               */
  haddr_t eof;        /* end of file; current file size        */
  int fd;             /* CTE cache handle (-1 if none this session) */
  int posix_fd;       /* authoritative on-disk native file fd  */
  char *filename_;    /* the name of the file (NULL if empty)  */
  unsigned flags;     /* the flags passed from H5Fcreate/H5Fopen */
  H5FD_clio_fapl_t fa; /* driver-specific FAPL config for this file */
  /* Filesystem identity of the authoritative native file, captured at open.
   * This -- NOT the filename -- is what cmp() compares: HDF5 uses cmp() to
   * decide whether an already-open file IS the same file, and two spellings of
   * one path (relative vs absolute, symlink vs target, with vs without the
   * clio:: marker) must compare equal or the library opens the same file twice
   * with two independent metadata caches, which corrupts it. sec2 parity. */
  dev_t st_dev;       /* device id of the authoritative native file */
  ino_t st_ino;       /* inode number of the authoritative native file */
} H5FD_clio_t;

/* Prototypes */
static herr_t H5FD__clio_term(void);
static void *H5FD__clio_fapl_get(H5FD_t *_file);
static void *H5FD__clio_fapl_copy(const void *_old_fa);
static herr_t H5FD__clio_fapl_free(void *_fa);
static H5FD_t *H5FD__clio_open(const char *name, unsigned flags,
                                 hid_t fapl_id, haddr_t maxaddr);
static herr_t H5FD__clio_close(H5FD_t *_file);
static int H5FD__clio_cmp(const H5FD_t *_f1, const H5FD_t *_f2);
static herr_t H5FD__clio_query(const H5FD_t *_f1, unsigned long *flags);
static haddr_t H5FD__clio_get_eoa(const H5FD_t *_file, H5FD_mem_t type);
static herr_t H5FD__clio_set_eoa(H5FD_t *_file, H5FD_mem_t type,
                                   haddr_t addr);
static haddr_t H5FD__clio_get_eof(const H5FD_t *_file, H5FD_mem_t type);
static herr_t H5FD__clio_read(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id,
                                haddr_t addr, size_t size, void *buf);
static herr_t H5FD__clio_write(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id,
                                 haddr_t addr, size_t size, const void *buf);
static herr_t H5FD__clio_read_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                     H5FD_mem_t types[], haddr_t addrs[],
                                     size_t sizes[], void *bufs[]);
static herr_t H5FD__clio_write_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                      H5FD_mem_t types[], haddr_t addrs[],
                                      size_t sizes[], const void *bufs[]);
static herr_t H5FD__clio_get_handle(H5FD_t *_file, hid_t fapl,
                                    void **file_handle);
static herr_t H5FD__clio_flush(H5FD_t *_file, hid_t dxpl_id, bool closing);
static herr_t H5FD__clio_truncate(H5FD_t *_file, hid_t dxpl_id, bool closing);
static herr_t H5FD__clio_lock(H5FD_t *_file, bool rw);
static herr_t H5FD__clio_unlock(H5FD_t *_file);
static herr_t H5FD__clio_del(const char *name, hid_t fapl);

static const H5FD_class_t H5FD_clio_g = {
    H5FD_CLASS_VERSION,   /* struct version       */
    H5FD_CLIO_VALUE,   /* value                */
    H5FD_CLIO_NAME,    /* name                 */
    MAXADDR,              /* maxaddr              */
    /* sec2 parity: the driver's DEFAULT file-close degree. Under STRONG (the
     * previous value) H5Fclose tears the file down even with objects still
     * open, invalidating their ids; under WEAK the close defers until the last
     * object closes. Applications observe the difference, so differing from
     * sec2 here is a silent native-compatibility deviation. */
    H5F_CLOSE_WEAK,       /* fc_degree            */
    H5FD__clio_term,    /* terminate            */
    NULL,                 /* sb_size              */
    NULL,                 /* sb_encode            */
    NULL,                 /* sb_decode            */
    sizeof(H5FD_clio_fapl_t), /* fapl_size        */
    H5FD__clio_fapl_get,      /* fapl_get         */
    H5FD__clio_fapl_copy,     /* fapl_copy        */
    H5FD__clio_fapl_free,     /* fapl_free        */
    0,                    /* dxpl_size            */
    NULL,                 /* dxpl_copy            */
    NULL,                 /* dxpl_free            */
    H5FD__clio_open,    /* open                 */
    H5FD__clio_close,   /* close                */
    H5FD__clio_cmp,     /* cmp                  */
    H5FD__clio_query,   /* query                */
    NULL,                 /* get_type_map         */
    NULL,                 /* alloc                */
    NULL,                 /* free                 */
    H5FD__clio_get_eoa, /* get_eoa              */
    H5FD__clio_set_eoa, /* set_eoa              */
    H5FD__clio_get_eof,    /* get_eof            */
    H5FD__clio_get_handle, /* get_handle         */
    H5FD__clio_read,        /* read              */
    H5FD__clio_write,       /* write             */
    H5FD__clio_read_vector, /* read_vector       */
    H5FD__clio_write_vector,/* write_vector      */
    NULL,                  /* read_selection     */
    NULL,                  /* write_selection    */
    H5FD__clio_flush,      /* flush              */
    H5FD__clio_truncate,   /* truncate           */
    H5FD__clio_lock,       /* lock               */
    H5FD__clio_unlock,     /* unlock             */
    H5FD__clio_del,        /* del                  */
    NULL,                 /* ctl                  */
    H5FD_FLMAP_DICHOTOMY  /* fl_map               */
};

/*-------------------------------------------------------------------------
 * Function:    H5FD_clio_init
 *
 * Purpose:     Initialize this driver by registering the driver with the
 *              library.
 *
 * Return:      Success:    The driver ID for the clio driver
 *              Failure:    H5I_INVALID_HID
 *
 *-------------------------------------------------------------------------
 */
hid_t H5FD_clio_init(void) {
  hid_t ret_value = H5I_INVALID_HID; /* Return value */

  /* Register the driver's HDF5 error class + messages once. Without this,
   * term() unregistered a class that init() never registered, so the error
   * path was dead and failures were silent. */
  if (H5FDclio_err_class_g < 0) {
    H5FDclio_err_class_g =
        H5Eregister_class("CLIO VFD", H5FD_CLIO_NAME, "0.1");
    if (H5FDclio_err_class_g >= 0) {
      H5FDclio_err_major_g =
          H5Ecreate_msg(H5FDclio_err_class_g, H5E_MAJOR, "CLIO VFD I/O");
      H5FDclio_err_minor_g = H5Ecreate_msg(H5FDclio_err_class_g, H5E_MINOR,
                                           "operation failed");
    }
  }

  if (H5I_VFL != H5Iget_type(H5FD_CLIO_g)) {
    H5FD_CLIO_g = H5FDregister(&H5FD_clio_g);
  }

  /* Set return value */
  ret_value = H5FD_CLIO_g;
  return ret_value;
} /* end H5FD_clio_init() */

/*---------------------------------------------------------------------------
 * Function:    H5FD__clio_term
 *
 * Purpose:     Shut down the VFD
 *
 * Returns:     SUCCEED (Can't fail)
 *
 *---------------------------------------------------------------------------
 */
static herr_t H5FD__clio_term(void) {
  herr_t ret_value = SUCCEED;

  /* Unregister from HDF5 error API (also frees the class's messages). */
  if (H5FDclio_err_class_g >= 0) {
    if (H5Eunregister_class(H5FDclio_err_class_g) < 0) {
      // TODO(llogan)
    }
    H5FDclio_err_class_g = H5I_INVALID_HID;
    H5FDclio_err_major_g = H5I_INVALID_HID;
    H5FDclio_err_minor_g = H5I_INVALID_HID;
  }

  /* Reset VFL ID */
  H5FD_CLIO_g = H5I_INVALID_HID;

  return ret_value;
} /* end H5FD__clio_term() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_open
 *
 * Purpose:     Create and/or open a file. The authoritative store is a real
 *              on-disk native HDF5 file; a CTE cache handle is opened alongside
 *              as groundwork for a future read/tiering cache.
 *
 * Return:      Success:    A pointer to a new file data structure.
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static H5FD_t *H5FD__clio_open(const char *name, unsigned flags,
                                 hid_t fapl_id, haddr_t maxaddr) {
  // Argument validation, sec2 parity. Without these a NULL name dereferences in
  // StripClioPrefix below, and an out-of-range maxaddr is accepted despite the
  // class advertising MAXADDR.
  if (!name || !*name) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("invalid file name (NULL or empty)");
    return nullptr;
  }
  if (0 == maxaddr || HADDR_UNDEF == maxaddr) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("invalid maxaddr");
    return nullptr;
  }
  if (maxaddr > MAXADDR) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("maxaddr exceeds the driver's addressable range");
    return nullptr;
  }

  // Driver-specific FAPL config: use the caller's policy if a driver-info block
  // was set (H5Pset_fapl_clio), else the default (cache on).
  const H5FD_clio_fapl_t *fa_in =
      (const H5FD_clio_fapl_t *)H5Pget_driver_info(fapl_id);
  H5FD_clio_fapl_t fa = fa_in ? *fa_in : H5FD_clio_fapl_default_g;

  /* Environment opt-out, applied before the runtime probe below so that
     CLIO_VFD_CACHE=0 also skips the attach attempt entirely (and therefore its
     retry timeout) rather than attaching and then not using it. */
  if (fa.cache_enabled && !H5FD__clio_cache_env_enabled()) {
    fa.cache_enabled = 0;
  }

  // Attach to the CLIO runtime ONLY when this file actually wants the cache
  // tier. Model A keeps the native file authoritative, so the native-only
  // configuration (H5Pset_fapl_clio(fapl, false)) is a complete, correct driver
  // on its own and must not require CLIO to be running -- previously this ran
  // unconditionally and with its result discarded, so a down/absent runtime
  // broke even the native-only path. If attach fails we degrade to native-only
  // for this file rather than failing the open: the authoritative store is
  // unaffected, and the cache is a performance tier, not a correctness one.
  if (fa.cache_enabled && !H5FD__clio_cache_available()) {
    HLOG(kWarning,
         "CLIO runtime unavailable; opening {} native-only (cache disabled)",
         name);
    fa.cache_enabled = 0;
  }

  /* Build the open flags */
  int o_flags = (H5F_ACC_RDWR & flags) ? O_RDWR : O_RDONLY;
  if (H5F_ACC_TRUNC & flags) {
    o_flags |= O_TRUNC;
  }
  if (H5F_ACC_CREAT & flags) {
    o_flags |= O_CREAT;
  }
  if (H5F_ACC_EXCL & flags) {
    o_flags |= O_EXCL;
  }

  // The AUTHORITATIVE store is a real on-disk native HDF5 file at the stripped
  // path (clio::/tmp/foo.h5 -> /tmp/foo.h5), so standard tools (h5dump/h5ls)
  // read it live.
  std::string native_path = clio::cae::StripClioPrefix(name);
  int posix_fd =
      open(native_path.c_str(), o_flags, H5FD_CLIO_POSIX_CREATE_MODE_RW);
  if (posix_fd < 0) {
    // Fail-closed: no authoritative file => the open fails. We do not proceed
    // with a cache-only file. Record errno on the driver error stack.
    H5FD_CLIO_ERROR("open() of authoritative native file failed");
    return nullptr;
  }

  // FUTURE (CTE tiering): open a CTE/CFS handle for this file so writes can
  // populate a fast cache tier and reads can eventually be served from it.
  // Today the handle is populated on write but NOT yet served on reads, so the
  // open is best-effort: a cache-open failure must not sink the open (the
  // authoritative native file already succeeded) -- fd == -1 just means "no
  // cache this session". Skipped entirely when the FAPL disables the cache
  // (which avoids the current write-amplification of the populate-only tier).
  int fd = -1;
  if (fa.cache_enabled) {
    fd = CLIO_CTE_CFS->Open(name, o_flags, H5FD_CLIO_POSIX_CREATE_MODE_RW);
    HLOG(kDebug, "");
  }

  /* Create the new file struct */
  H5FD_clio_t *file = (H5FD_clio_t *)calloc(1, sizeof(H5FD_clio_t));
  if (file == NULL) {
    // Out of memory: release the handles we already opened instead of leaking
    // them (and dereferencing a NULL file). calloc does not reliably set errno,
    // so set it explicitly for an accurate error message.
    errno = ENOMEM;
    H5FD_CLIO_ERROR("calloc() of VFD file struct failed");
    close(posix_fd);
    if (fd >= 0) {
      CLIO_CTE_CFS->Close(fd);
    }
    return nullptr;
  }

  // Identity + size from ONE fstat of the authoritative file. cmp() depends on
  // dev/ino, so a failed fstat is fail-closed: without identity the library
  // could not tell this file apart from another and might open it twice.
  struct stat st;
  if (fstat(posix_fd, &st) < 0) {
    H5FD_CLIO_ERROR("fstat() of authoritative native file failed");
    close(posix_fd);
    if (fd >= 0) {
      CLIO_CTE_CFS->Close(fd);
    }
    free(file);
    return nullptr;
  }

  /* Pack file */
  file->filename_ = strdup(name);
  if (!file->filename_) {
    errno = ENOMEM;
    H5FD_CLIO_ERROR("strdup() of file name failed");
    close(posix_fd);
    if (fd >= 0) {
      CLIO_CTE_CFS->Close(fd);
    }
    free(file);
    return nullptr;
  }
  file->fd = fd;
  file->posix_fd = posix_fd;
  file->flags = flags;
  file->fa = fa;
  file->st_dev = st.st_dev;
  file->st_ino = st.st_ino;

  // EOF is the authoritative on-disk size (durable across reopen/append), not a
  // session-local counter or the cache's logical size.
  file->eof = (haddr_t)st.st_size;

  return (H5FD_t *)file;
} /* end H5FD__clio_open() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_close
 *
 * Purpose:     Closes an HDF5 file.
 *
 * Return:      Success:    SUCCEED
 *              Failure:    FAIL, file not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_close(H5FD_t *_file) {
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  herr_t ret_value = SUCCEED; /* Return value */
  assert(file);
  // fsync + close the authoritative native file first -- a successful close is
  // a durability barrier (no pending dirty state), and the on-disk file is a
  // complete valid native HDF5 image afterward.
  if (file->posix_fd >= 0) {
    if (fsync(file->posix_fd) < 0) {
      H5FD_CLIO_ERROR("fsync() on close failed");
      ret_value = FAIL; /* fail-closed: a close that did not persist fails */
    }
    // close() itself can fail (EIO, and notably deferred write errors on NFS).
    // A close that reports an error has not necessarily persisted, so it is
    // fail-closed too -- the whole point of the barrier.
    if (close(file->posix_fd) < 0) {
      H5FD_CLIO_ERROR("close() of authoritative native file failed");
      ret_value = FAIL;
    }
  }
  // Release the CTE cache handle, if this session had one.
  if (file->fd >= 0) {
    CLIO_CTE_CFS->Close(file->fd);
    HLOG(kDebug, "");
  }
  if (file->filename_) {
    free(file->filename_);
  }
  free(file);
  return ret_value;
} /* end H5FD__clio_close() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_cmp
 *
 * Purpose:     Compares two files belonging to this driver using an arbitrary
 *              (but consistent) ordering.
 *
 * Return:      Success:    A value like strcmp()
 *              Failure:    never fails (arguments were checked by the
 *                          caller).
 *
 *-------------------------------------------------------------------------
 */
static int H5FD__clio_cmp(const H5FD_t *_f1, const H5FD_t *_f2) {
  const H5FD_clio_t *f1 = (const H5FD_clio_t *)_f1;
  const H5FD_clio_t *f2 = (const H5FD_clio_t *)_f2;
  // Compare filesystem IDENTITY (device + inode), not the filename string.
  // HDF5 uses this to decide whether an already-open file is the same file;
  // comparing names made "/tmp/f.h5", "./f.h5", "clio::/tmp/f.h5" and a symlink
  // to any of them look like four different files, so the library could open
  // one file several times with independent metadata caches and corrupt it.
  // sec2 compares dev/ino for exactly this reason.
  if (f1->st_dev < f2->st_dev) return -1;
  if (f1->st_dev > f2->st_dev) return 1;
  if (f1->st_ino < f2->st_ino) return -1;
  if (f1->st_ino > f2->st_ino) return 1;
  return 0;
} /* end H5FD__clio_cmp() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_query
 *
 * Purpose:     Set the flags that this VFL driver is capable of supporting.
 *              (listed in H5FDpublic.h)
 *
 * Return:      SUCCEED (Can't fail)
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_query(const H5FD_t *_file,
                                 unsigned long *flags /* out */) {
  (void)_file;
  if (flags) {
    /* Advertise the I/O-optimization subset of sec2's feature set. Model A
     * makes these honest: the backend is a real byte-addressable POSIX file, so
     * HDF5's metadata aggregation / accumulation / data-sieve / small-data
     * aggregation optimizations all apply. Returning 0 (the old behavior)
     * silently disabled all of them.
     *
     * NOT set: H5FD_FEAT_POSIX_COMPAT_HANDLE and H5FD_FEAT_DEFAULT_VFD_COMPATIBLE
     * -- the two "name-resolving" flags. Under either, H5F_open() resolves the
     * file's canonical name by stat()/realpath() on the filename it was handed
     * (H5F__build_actual_name), which for this driver is the clio::-marked path
     * (e.g. clio::/tmp/foo.h5), NOT a real filesystem path -- so stat() fails
     * and H5Fopen aborts. sec2's names are real paths; ours carry the marker, so
     * these two can only be advertised once the name HDF5 sees is the bare
     * (stripped) path. get_handle itself still works, and the on-disk image is a
     * plain native HDF5 file readable by standard tools. */
    *flags = 0;
    *flags |= H5FD_FEAT_AGGREGATE_METADATA;   /* metadata block aggregation */
    *flags |= H5FD_FEAT_ACCUMULATE_METADATA;  /* metadata accumulation      */
    *flags |= H5FD_FEAT_DATA_SIEVE;           /* data sieving               */
    *flags |= H5FD_FEAT_AGGREGATE_SMALLDATA;  /* small raw-data aggregation */
    *flags |= H5FD_FEAT_SUPPORTS_SWMR_IO;     /* flock + real file          */
  }
  return SUCCEED;
} /* end H5FD__clio_query() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_get_eoa
 *
 * Purpose:     Gets the end-of-address marker for the file. The EOA marker
 *              is the first address past the last byte allocated in the
 *              format address space.
 *
 * Return:      The end-of-address marker.
 *
 *-------------------------------------------------------------------------
 */
static haddr_t H5FD__clio_get_eoa(const H5FD_t *_file, H5FD_mem_t type) {
  (void)type;
  const H5FD_clio_t *file = (const H5FD_clio_t *)_file;
  return file->eoa;
} /* end H5FD__clio_get_eoa() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_set_eoa
 *
 * Purpose:     Set the end-of-address marker for the file. This function is
 *              called shortly after an existing HDF5 file is opened in order
 *              to tell the driver where the end of the HDF5 data is located.
 *
 * Return:      SUCCEED (Can't fail)
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_set_eoa(H5FD_t *_file, H5FD_mem_t type,
                                   haddr_t addr) {
  (void)type;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  file->eoa = addr;
  return SUCCEED;
} /* end H5FD__clio_set_eoa() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_get_eof
 *
 * Purpose:     Returns the end-of-file marker, which is the greater of
 *              either the filesystem end-of-file or the HDF5 end-of-address
 *              markers.
 *
 * Return:      End of file address, the first address past the end of the
 *              "file", either the filesystem file or the HDF5 file.
 *
 *-------------------------------------------------------------------------
 */
static haddr_t H5FD__clio_get_eof(const H5FD_t *_file, H5FD_mem_t type) {
  (void)type;
  const H5FD_clio_t *file = (const H5FD_clio_t *)_file;
  return file->eof;
} /* end H5FD__clio_get_eof() */

/*-------------------------------------------------------------------------
 * Shared byte-I/O primitives used by BOTH the scalar (read/write) and the
 * vectored (read_vector/write_vector) callbacks, so their semantics cannot
 * drift -- when the CTE read-cache tier eventually lands, it changes here once
 * and both paths inherit it.
 *
 *   H5FD__clio_do_read: read SIZE bytes at ADDR from the authoritative native
 *     file into BUF, zero-filling any tail past EOF (HDF5 treats the file as a
 *     flat byte array). A genuine read error is fail-closed; a short read is EOF.
 *     FUTURE (CTE read tier): consult the cache first and serve a hit from a fast
 *     tier, falling back to native on a miss. That needs per-file tracking of
 *     which byte ranges are populated -- the CFS chimod zero-fills holes and
 *     reports a full read, so a naive lookup could return stale zeros. Until
 *     then, reads come straight from the authoritative file.
 *
 *   H5FD__clio_do_write: write-through SIZE bytes at ADDR to the authoritative
 *     native file (fail-closed on short/failed write), best-effort populate the
 *     CTE cache tier when a handle exists (groundwork; not yet served on reads),
 *     and advance the session end-of-file.
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_do_read(H5FD_clio_t *file, haddr_t addr, size_t size,
                                 void *buf) {
  if (H5FD_CLIO_REGION_INVALID(addr, size)) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("read region is undefined or out of range");
    return FAIL;
  }
  char *dst = static_cast<char *>(buf);
  size_t remaining = size;
  off_t off = static_cast<off_t>(addr);

  // Loop rather than issue one pread. A short return is NOT proof of EOF: the
  // kernel caps a single transfer (0x7ffff000 on Linux) and a signal can cut
  // one short via EINTR. The previous single-shot version zero-filled whatever
  // it did not get, so a >2 GiB read -- or any interrupted one -- silently
  // handed back zeros in place of real data WITH a success status. Only a
  // pread returning exactly 0 is true EOF; HDF5 treats the file as a flat byte
  // array, so the tail past EOF is legitimately zero-filled.
  while (remaining > 0) {
    const size_t cap = H5FD__clio_max_io_bytes();
    size_t want = (remaining > cap) ? cap : remaining;
    ssize_t got = pread(file->posix_fd, dst, want, off);
    if (got < 0) {
      if (errno == EINTR) {
        continue; /* interrupted before transferring anything: retry */
      }
      H5FD_CLIO_ERROR("pread() of authoritative native file failed");
      return FAIL;
    }
    if (got == 0) {
      memset(dst, 0, remaining); /* genuine EOF: zero-fill the remainder */
      break;
    }
    dst += got;
    off += got;
    remaining -= static_cast<size_t>(got);
  }
  if (getenv("CLIO_VFD_DEBUG"))
    fprintf(stderr, "[vfd] READ  addr=%llu size=%llu\n",
            (unsigned long long)addr, (unsigned long long)size);
  return SUCCEED;
}

static herr_t H5FD__clio_do_write(H5FD_clio_t *file, haddr_t addr, size_t size,
                                  const void *buf) {
  if (H5FD_CLIO_REGION_INVALID(addr, size)) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("write region is undefined or out of range");
    return FAIL;
  }
  const char *src = static_cast<const char *>(buf);
  size_t remaining = size;
  off_t off = static_cast<off_t>(addr);

  // Same loop as the read path: chunk to stay under the kernel's per-call cap
  // and retry on EINTR. A short pwrite is a partial transfer to be continued,
  // not a failure -- unlooped, a >2 GiB write failed outright.
  while (remaining > 0) {
    const size_t cap = H5FD__clio_max_io_bytes();
    size_t want = (remaining > cap) ? cap : remaining;
    ssize_t put = pwrite(file->posix_fd, src, want, off);
    if (put < 0) {
      if (errno == EINTR) {
        continue;
      }
      H5FD_CLIO_ERROR("pwrite() to authoritative native file failed");
      return FAIL;
    }
    if (put == 0) {
      // No progress and no error: cannot complete the write. Fail closed
      // rather than spin.
      H5FD_CLIO_ERROR("pwrite() to authoritative native file made no progress");
      return FAIL;
    }
    src += put;
    off += put;
    remaining -= static_cast<size_t>(put);
  }
  if (getenv("CLIO_VFD_DEBUG"))
    fprintf(stderr, "[vfd] WRITE addr=%llu size=%llu\n",
            (unsigned long long)addr, (unsigned long long)size);

  // Populate the cache tier. Best-effort by design (the authoritative write
  // already succeeded), but NOT silent: a dropped populate is a range the tier
  // does not hold, which the future read tier must not mistake for resident
  // data. Count it and log once per failure so residency work has a signal.
  if (file->fd >= 0) {
    if (CLIO_CTE_CFS->Pwrite(file->fd, buf, size, static_cast<off_t>(addr)) < 0) {
      H5FDclio_cache_write_failures_g++;
      HLOG(kWarning,
           "CTE cache populate failed at addr={} size={} (native file is "
           "unaffected and remains authoritative)",
           (unsigned long long)addr, (unsigned long long)size);
    }
  }
  if ((haddr_t)(addr + size) > file->eof) {
    file->eof = (haddr_t)(addr + size);
  }
  return SUCCEED;
}

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_read
 *
 * Purpose:     Reads SIZE bytes of data from FILE beginning at address ADDR
 *              into buffer BUF. Reads come from the authoritative native file;
 *              a read of a region past the last byte ever written is
 *              zero-filled (HDF5 treats the file as a flat byte array).
 *
 * Return:      Success:    SUCCEED. Result is stored in caller-supplied
 *                          buffer BUF.
 *              Failure:    FAIL, Contents of buffer BUF are undefined.
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_read(H5FD_t *_file, H5FD_mem_t type, hid_t dxpl_id,
                                haddr_t addr, size_t size, void *buf) {
  (void)dxpl_id;
  (void)type;
  return H5FD__clio_do_read((H5FD_clio_t *)_file, addr, size, buf);
} /* end H5FD__clio_read() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_write
 *
 * Purpose:     Writes SIZE bytes of data from buffer BUF at file address ADDR.
 *              The write is committed synchronously to the authoritative native
 *              file; a short/failed write is fail-closed.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_write(H5FD_t *_file, H5FD_mem_t type, hid_t dxpl_id,
                                 haddr_t addr, size_t size, const void *buf) {
  (void)dxpl_id;
  (void)type;
  return H5FD__clio_do_write((H5FD_clio_t *)_file, addr, size, buf);
} /* end H5FD__clio_write() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_read_vector / H5FD__clio_write_vector
 *
 * Purpose:     Vectored I/O: service a whole vector of (addr, size, buf)
 *              elements in ONE driver call, rather than have HDF5 re-dispatch
 *              each element through the scalar read/write callbacks (the VFL
 *              emulation). Each element goes through the shared do_read/do_write
 *              helpers, so the semantics match the scalar paths exactly. Note the
 *              elements are still issued as individual pread/pwrites; coalescing
 *              file-contiguous elements into a single preadv/pwritev is a
 *              possible future optimization (worthwhile only for patterns with
 *              contiguous runs, which HDF5's scattered vector I/O rarely has).
 *
 *              The `sizes` array may be shortened: a 0 entry (for i > 0) means
 *              this and all subsequent elements reuse the last explicit size.
 *              `addrs` and `bufs` always have `count` entries.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_read_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                     H5FD_mem_t types[], haddr_t addrs[],
                                     size_t sizes[], void *bufs[]) {
  (void)dxpl;
  (void)types;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  H5FDclio_read_vector_calls_g++;
  if (count == 0) {
    return SUCCEED;
  }
  size_t size = sizes[0];
  bool size_fixed = false; /* once a 0 size is seen, reuse the last explicit */
  for (uint32_t i = 0; i < count; i++) {
    if (!size_fixed) {
      if (i > 0 && sizes[i] == 0) {
        size_fixed = true; /* size stays = the last explicit size */
      } else {
        size = sizes[i];
      }
    }
    if (H5FD__clio_do_read(file, addrs[i], size, bufs[i]) < 0) {
      return FAIL;
    }
  }
  return SUCCEED;
} /* end H5FD__clio_read_vector() */

static herr_t H5FD__clio_write_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                      H5FD_mem_t types[], haddr_t addrs[],
                                      size_t sizes[], const void *bufs[]) {
  (void)dxpl;
  (void)types;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  H5FDclio_write_vector_calls_g++;
  if (count == 0) {
    return SUCCEED;
  }
  size_t size = sizes[0];
  bool size_fixed = false;
  for (uint32_t i = 0; i < count; i++) {
    if (!size_fixed) {
      if (i > 0 && sizes[i] == 0) {
        size_fixed = true;
      } else {
        size = sizes[i];
      }
    }
    if (H5FD__clio_do_write(file, addrs[i], size, bufs[i]) < 0) {
      return FAIL;
    }
  }
  return SUCCEED;
} /* end H5FD__clio_write_vector() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_get_handle
 *
 * Purpose:     Returns the POSIX file descriptor of the authoritative native
 *              file, for consumers (tools, the core VFD) that expect a real OS
 *              handle. Behaves like sec2.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_get_handle(H5FD_t *_file, hid_t fapl,
                                    void **file_handle) {
  (void)fapl;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  if (!file_handle) {
    return FAIL;
  }
  *file_handle = &(file->posix_fd);
  return SUCCEED;
} /* end H5FD__clio_get_handle() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_flush
 *
 * Purpose:     Durability barrier: fsync the authoritative native file so a
 *              successful H5Fflush/H5Dflush leaves no pending dirty state on
 *              disk. Fail-closed if the fsync fails.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_flush(H5FD_t *_file, hid_t dxpl_id, bool closing) {
  (void)dxpl_id;
  (void)closing;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  // Persist the authoritative native file; fail-closed so a flush that did not
  // reach disk never reports success. Writes are write-through, so the native
  // file is the only store holding data to flush.
  if (file->posix_fd >= 0 && fsync(file->posix_fd) < 0) {
    H5FD_CLIO_ERROR("fsync() in flush failed");
    return FAIL;
  }
  return SUCCEED;
} /* end H5FD__clio_flush() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_truncate
 *
 * Purpose:     Truncate the authoritative native file to the end-of-address
 *              marker so close-to-EOA yields the correct on-disk file size
 *              (a byte-exact native image). Behaves like sec2.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_truncate(H5FD_t *_file, hid_t dxpl_id, bool closing) {
  (void)dxpl_id;
  (void)closing;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  if (file->eof != file->eoa) {
    if (file->posix_fd >= 0 &&
        ftruncate(file->posix_fd, (off_t)file->eoa) < 0) {
      H5FD_CLIO_ERROR("ftruncate() of authoritative native file failed");
      return FAIL;
    }
    // Keep the CTE cache's logical size in step (best-effort; populate-only
    // tier, see the write callback). Counted on failure for the same reason:
    // a tier that did not shrink still holds bytes past the new EOF.
    if (file->fd >= 0) {
      if (CLIO_CTE_CFS->FtruncateFd(file->fd, (off_t)file->eoa) < 0) {
        H5FDclio_cache_truncate_failures_g++;
        HLOG(kWarning,
             "CTE cache truncate to {} failed (native file is unaffected and "
             "remains authoritative)",
             (unsigned long long)file->eoa);
      }
    }
    file->eof = file->eoa;
  }
  return SUCCEED;
} /* end H5FD__clio_truncate() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_lock / H5FD__clio_unlock
 *
 * Purpose:     Advisory whole-file locking (flock) on the authoritative native
 *              fd, for file locking / SWMR / concurrent-tool safety. Behaves
 *              like sec2: non-blocking flock, and a filesystem that does not
 *              support locking (ENOSYS) is not treated as an error.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_lock(H5FD_t *_file, bool rw) {
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  if (file->posix_fd < 0) {
    return SUCCEED;
  }
  int lock_flags = (rw ? LOCK_EX : LOCK_SH) | LOCK_NB;
  if (flock(file->posix_fd, lock_flags) < 0) {
    if (errno == ENOSYS) {
      return SUCCEED; /* locking unsupported here: not an error (sec2 parity) */
    }
    // Push a diagnosable error like every other failure path. Lock contention
    // (EWOULDBLOCK: another process holds the file) is exactly the failure a
    // user needs named, and it was the one case the driver stayed silent on.
    H5FD_CLIO_ERROR("flock() of authoritative native file failed");
    return FAIL;
  }
  return SUCCEED;
} /* end H5FD__clio_lock() */

static herr_t H5FD__clio_unlock(H5FD_t *_file) {
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  if (file->posix_fd < 0) {
    return SUCCEED;
  }
  if (flock(file->posix_fd, LOCK_UN) < 0) {
    if (errno == ENOSYS) {
      return SUCCEED;
    }
    H5FD_CLIO_ERROR("flock(LOCK_UN) of authoritative native file failed");
    return FAIL;
  }
  return SUCCEED;
} /* end H5FD__clio_unlock() */

/*-------------------------------------------------------------------------
 * Driver-specific FAPL memory management.
 *
 * These make the driver's FAPL a first-class, storable/copyable property so
 * H5Pset_driver(fapl, driver, &config) round-trips the config (with
 * fapl_size/get/copy/free all NULL, no driver-info could be carried at all).
 * The struct is POD, so copy/free are trivial.
 *-------------------------------------------------------------------------
 */
static void *H5FD__clio_fapl_get(H5FD_t *_file) {
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  H5FD_clio_fapl_t *fa = (H5FD_clio_fapl_t *)malloc(sizeof(H5FD_clio_fapl_t));
  if (fa) {
    *fa = file->fa;
  }
  return fa;
}

static void *H5FD__clio_fapl_copy(const void *_old_fa) {
  H5FD_clio_fapl_t *fa = (H5FD_clio_fapl_t *)malloc(sizeof(H5FD_clio_fapl_t));
  if (fa && _old_fa) {
    *fa = *(const H5FD_clio_fapl_t *)_old_fa;
  }
  return fa;
}

static herr_t H5FD__clio_fapl_free(void *_fa) {
  free(_fa);
  return SUCCEED;
}

/*-------------------------------------------------------------------------
 * Function:    H5Pset_fapl_clio
 *
 * Purpose:     Select the CLIO VFD on a file access property list and attach
 *              the driver-specific tiering policy. The supported, HDF5-idiomatic
 *              way to configure the driver (vs. H5Pset_driver with the raw id).
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t H5Pset_fapl_clio(hid_t fapl_id, hbool_t cache_enabled) {
  if (H5Pisa_class(fapl_id, H5P_FILE_ACCESS) <= 0) {
    H5FD_CLIO_ERROR("H5Pset_fapl_clio: not a file access property list");
    return FAIL;
  }
  hid_t driver = H5FD_clio_init();
  if (driver < 0) {
    return FAIL;
  }
  H5FD_clio_fapl_t fa;
  fa.cache_enabled = cache_enabled;
  return H5Pset_driver(fapl_id, driver, &fa);
}

/*-------------------------------------------------------------------------
 * Function:    H5Pget_fapl_clio
 *
 * Purpose:     Read back the driver-specific tiering policy from a FAPL that
 *              selects this driver. The symmetric counterpart to
 *              H5Pset_fapl_clio -- HDF5 convention pairs every H5Pset_fapl_*
 *              with a getter, and without one a caller cannot inspect (or
 *              round-trip) the config it set.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t H5Pget_fapl_clio(hid_t fapl_id, hbool_t *cache_enabled /*out*/) {
  if (H5Pisa_class(fapl_id, H5P_FILE_ACCESS) <= 0) {
    H5FD_CLIO_ERROR("H5Pget_fapl_clio: not a file access property list");
    return FAIL;
  }
  if (!cache_enabled) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("H5Pget_fapl_clio: NULL out parameter");
    return FAIL;
  }
  if (H5Pget_driver(fapl_id) != H5FD_clio_init()) {
    H5FD_CLIO_ERROR("H5Pget_fapl_clio: FAPL does not select the CLIO VFD");
    return FAIL;
  }
  // No driver-info block means the file would open with the default policy
  // (H5Pset_driver(fapl, driver, NULL)); report that same default so the getter
  // always describes what an open would actually do.
  const H5FD_clio_fapl_t *fa =
      (const H5FD_clio_fapl_t *)H5Pget_driver_info(fapl_id);
  *cache_enabled =
      fa ? fa->cache_enabled : H5FD_clio_fapl_default_g.cache_enabled;
  return SUCCEED;
}

/*-------------------------------------------------------------------------
 * Function:    H5FD__clio_del
 *
 * Purpose:     Delete the file NAME (H5Fdelete). Removes BOTH stores so neither
 *              is left orphaned: the authoritative on-disk native file
 *              (fail-closed, like sec2) and the CTE cache tag (best-effort -- the
 *              cache may have been disabled or never populated, in which case its
 *              removal is a harmless no-op). Called without an open handle, so it
 *              (re)initializes the CTE client and works purely by name.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_del(const char *name, hid_t fapl) {
  (void)fapl;
  if (!name || !*name) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("H5Fdelete: invalid file name (NULL or empty)");
    return FAIL;
  }

  // Drop the CTE cache tag first so it can never be orphaned behind a deleted
  // native file. Best-effort: keyed by the same full name open() used, and
  // absence (cache off / never populated) is a harmless no-op. If the runtime
  // is not reachable there is no cache entry to orphan, so deleting the native
  // file alone is still correct -- the delete must not fail just because CLIO
  // is down (same reasoning as open()).
  if (H5FD__clio_cache_available()) {
    CLIO_CTE_CFS->RemovePath(name);
  }

  // Remove the authoritative native file at the stripped path. Fail-closed on
  // error (sec2 parity) so a failed delete is reported, not masked.
  std::string native_path = clio::cae::StripClioPrefix(name);
  if (unlink(native_path.c_str()) < 0) {
    H5FD_CLIO_ERROR("unlink() of authoritative native file failed");
    return FAIL;
  }
  return SUCCEED;
} /* end H5FD__clio_del() */

/*
 * Entry points for dynamic plugin loading.
 */
H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_VFD; }

const void *H5PLget_plugin_info(void) { return &H5FD_clio_g; }

} // extern C
