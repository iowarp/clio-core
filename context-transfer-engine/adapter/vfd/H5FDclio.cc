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
#include <sys/stat.h>
#include <sys/types.h>

/* HDF5 header for dynamic plugin loading */
#include "H5FDclio.h" /* Clio file driver     */
#include "H5FDclio_compat.h" /* POSIX/Win32 platform layer */
#include "H5PLextern.h"
#include "adapter/clio_config_str.h"
#include "adapter/clio_require_runtime.h"
#include "adapter/clio_coherence_stamp.h"
#include "H5FDclio_trace.h"
#include <clio_cte/filesystem/filesystem_client.h>
#include "clio_cte/core/core_client.h"
#include <clio_ctp/util/logging.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <vector>

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

/* Observability: the largest span serviced as ONE coalesced I/O, in bytes.
 * The coalescing window is a promise about memory -- the span sizes a scratch
 * buffer -- and a promise nothing can see is one that rots. Single-element
 * groups are not counted: those are serviced directly, allocate nothing, and
 * are legitimately larger than the window. Exported (not static) so the suite
 * can assert the window is actually enforced. */
unsigned long H5FDclio_vec_max_span_g = 0;

/* Observability: CTE cache-tier operations that FAILED. Survivable while the
 * cache is populate-only, but a dropped Pwrite is a range the cache believes
 * it holds and does not -- the stale-data hazard any read tier has to contain.
 * Exported (not static) so tests can assert on them. */
unsigned long H5FDclio_cache_write_failures_g = 0;
/* Reads served from the tier, and reads that fell through to the native file.
 * Both are needed: a cache that is off and one that never hits are otherwise
 * indistinguishable -- same bytes, same success. */
unsigned long H5FDclio_cache_read_hits_g = 0;
unsigned long H5FDclio_cache_read_misses_g = 0;
/* Cached copies dropped because the file changed underneath them. Distinct
 * from a read miss: a workload that invalidates on every open is paying to
 * populate a cache it can never use. */
unsigned long H5FDclio_cache_stale_invalidations_g = 0;

/* Also governs the coherence stamp: coherence exists only to make the read
 * tier safe, so with the tier off, validating and stamping is pure cost. Safe
 * to skip -- a session that never stamps leaves none, and the next open that
 * cares fails closed. */
static bool H5FD__clio_read_tier_on() {
  static const bool on = [] {
    const char *e = getenv("CLIO_VFD_READ_TIER");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

/* An xattr on the TIER's copy, so it travels with the cached bytes and is
   dropped with them. Never on the native file: this driver must leave the
   authoritative image exactly as a native writer would. */
static constexpr const char *H5FD_CLIO_STAMP_XATTR = "user.clio.coherence";

/* "1:" is this connector's cache-layout version -- bump it when the CFS page
   layout changes, so old cached bytes are invalidated. Empty when the file
   cannot be stat'd, which callers treat as "no verdict". */
static std::string H5FD__clio_stamp_of(const char *native_path) {
  const std::string id = clio::adapter::stamp::FileIdentity(native_path);
  if (id.empty()) return std::string();
  return std::string("1:") + id;
}

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

/* Whether the CTE cache tier can be compiled at all.
 *
 * The tier is driven through clio::cte::filesystem::Client's descriptor API
 * (OpenFd/PwriteFd/CloseFd/FtruncateFd/RemovePath), and that API is POSIX-only
 * upstream -- filesystem_client.h puts it inside `#if !defined(_WIN32)`
 * because it is specified in terms of ssize_t/off_t and POSIX descriptor
 * semantics. Until it is ported, the Windows build is native-only: the
 * authoritative on-disk file is written exactly as on every other platform,
 * and the cache tier is off.
 *
 * This is the same degradation the driver already applies at run time when the
 * CLIO runtime is unreachable (see H5FD__clio_cache_available below); on
 * Windows the answer is simply known at compile time. */
#define H5FD_CLIO_HAVE_CACHE_TIER 1

#define MAXADDR (((haddr_t)1 << (8 * sizeof(clio_vfd_off_t) - 1)) - 1)
#define SUCCEED 0
#define FAIL (-1)

/* Largest byte count handed to a single pread/pwrite. Linux caps a single
 * transfer at 0x7ffff000 and returns a SHORT count above it, so an unlooped
 * call would silently transfer less than asked. sec2 chunks for the same
 * reason (H5_POSIX_MAX_IO_BYTES); 1 GiB is comfortably under every platform
 * limit and keeps the loop count trivial for realistic HDF5 requests.
 *
 * CLIO_VFD_MAX_IO_BYTES overrides it so the multi-pass path can be exercised
 * with kilobyte-sized transfers -- the splitting/resume logic is identical at
 * any threshold, and a test needing 2 GiB of disk does not get run. */
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

/*---------------------------------------------------------------------------
 * Process-exit guard.
 *
 * HDF5 installs atexit(H5_term_library) the first time the library is
 * initialized, and H5_term_library closes every file, datatype and group the
 * application left open -- which reaches this driver's write, flush, truncate
 * and close callbacks LONG AFTER main() has returned.
 *
 * The CLIO client, by contrast, is a set of process-global objects whose
 * teardown runs as ordinary static destructors: the shared ZeroMQ context
 * (whose destructor zmq_ctx_shutdown()s every socket, so RecvZmqClientThread
 * exits), the deferred-write registry, the IPC manager's receive threads.
 * Every one of those is registered LATER than HDF5's atexit handler -- the
 * client is first constructed on the first H5Fopen, long after H5open() -- and
 * exit handlers run last-registered-first. So by the time H5_term_library asks
 * this driver to close a file, CLIO is already gone. Calling into it then does
 * one of two things, both observed in the netCDF-C suite:
 *
 *   - SIGSEGV inside Client::DeferRegisterWrite, dereferencing the destroyed
 *     DeferRegistry hash table (h5_test/tst_h_dimscales, at exit, after the
 *     test itself printed "Tests successful!");
 *   - an unkillable hang in Client::CloseFd, waiting on a future that no
 *     surviving receive thread will ever complete (h5_test/tst_h_compounds2,
 *     tst_h_strings2, and every ncdump shell test whose child processes never
 *     exit).
 *
 * The fix is to stop calling into CLIO once teardown has begun. The guard is
 * armed from every entry point HDF5 can reach this driver through --
 * H5PLget_plugin_info() for the HDF5_DRIVER=clio_vfd path, H5FD_clio_init()
 * for an application that registers the driver itself, and open() as a
 * backstop -- all of which run AFTER H5open() has installed H5_term_library,
 * so LIFO ordering guarantees it runs BEFORE H5_term_library: the flag is
 * always set by the time the late callbacks arrive. Nothing is lost by
 * skipping the tier there -- it is populate-only and
 * best-effort, and every byte has already been written through to the
 * authoritative native file, so the on-disk image is complete and valid either
 * way.
 *
 * Not atomic on purpose: it is written once by the thread running exit
 * handlers, at a point where the C runtime has already serialized teardown.
 *---------------------------------------------------------------------------*/
static bool H5FDclio_exiting_g = false;

static void H5FD__clio_note_exit(void) { H5FDclio_exiting_g = true; }

/* Register the guard once, from driver init. Separate from the flag so the
 * registration cannot be duplicated by a re-register of the driver. */
static void H5FD__clio_install_exit_guard(void) {
  static bool installed = false;
  if (!installed) {
    installed = true;
    atexit(H5FD__clio_note_exit);
  }
}

/* True when it is still safe to call into the CTE cache tier: this file has a
 * handle AND the process has not begun running exit handlers. Every cache-tier
 * call site goes through this -- a call site that tests `fd >= 0` alone is the
 * bug described above. */
static inline bool H5FD__clio_cache_live(int fd) {
  return fd >= 0 && !H5FDclio_exiting_g;
}

/* Attach to the CLIO runtime, at most once per process, and remember the
 * answer. Attaching to an absent runtime retries for CLIO_CLIENT_RETRY_TIMEOUT
 * seconds (60 by default), so asking per H5Fopen would make a down runtime
 * cost that timeout per file and a hundred-file workload appear to hang.
 *
 * Accepted consequence: a runtime that starts AFTER the first open is not
 * picked up for the life of the process. Files stay native-only, which is
 * correct, just unaccelerated. */
static bool H5FD__clio_cache_available(void) {
  /* Checked before the cached answer: a file opened while exit handlers are
     running (HDF5 reopens nothing, but H5FD__clio_del does open a FAPL path)
     must not attach to a client that is being torn down. */
  if (H5FDclio_exiting_g) return false;
  static const bool available = clio::cte::core::CLIO_CTE_CLIENT_INIT();
  return available;
}

/* Read the CLIO_VFD_CACHE opt-out. Default on; "0"/"off"/"false"/"no" disable
 * the CTE tier and make the driver pure write-through to the native file.
 * Same name-shape and accepted values as the VOL's CLIO_VOL_CACHE, so the two
 * connectors take the same muscle memory to switch off.
 *
 * Precedence: this is an opt-OUT only. It can force the cache off, but never
 * on over an explicit H5Pset_fapl_clio(fapl, false) -- either source is
 * sufficient to DISABLE, which is the fail-closed direction, and a caller who
 * asked for native-only in code must not have the cache switched back on
 * underneath by an environment it did not set.
 */
static bool H5FD__clio_cache_env_enabled(void) {
  const char *v = getenv("CLIO_VFD_CACHE");
  if (!v || !*v) return true;
  return !(strcmp(v, "0") == 0 || strcmp(v, "off") == 0 ||
           strcmp(v, "false") == 0 || strcmp(v, "no") == 0);
}

/* Read the CLIO_VFD_FSYNC opt-IN for the flush callback. Default off; any of
 * "1"/"on"/"true"/"yes" turns it on for every file in the process. The mirror
 * image of CLIO_VFD_CACHE: that one can only DISABLE what the FAPL asked for,
 * this one can only ENABLE what the FAPL left off -- in both cases the
 * environment moves the knob in the safe direction (fail-closed for the cache,
 * more-durable for fsync), so neither can silently weaken an explicit choice
 * made in code. */
static bool H5FD__clio_fsync_env_forced(void) {
  const char *v = getenv("CLIO_VFD_FSYNC");
  if (!v || !*v) return false;
  return strcmp(v, "1") == 0 || strcmp(v, "on") == 0 ||
         strcmp(v, "true") == 0 || strcmp(v, "yes") == 0;
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
  hbool_t fsync_on_flush; /* fsync(2) the native file in flush (default off) */
  size_t sieve_max;      /* vector-I/O coalescing window, bytes (0 = off) */
} H5FD_clio_fapl_t;

/* Coalescing window for vector I/O. 64 KiB matches HDF5's own default sieve
 * buffer (H5Pset_sieve_buf_size), which is the mechanism this replaces for
 * drivers that implement vector I/O -- see H5FD__clio_write_vector. */
#define H5FD_CLIO_SIEVE_MAX_DEF ((size_t)(64 * 1024))

/* Largest accepted coalescing window. The window bounds a scratch buffer this
 * driver allocates per vector call, so an absurd value is not a slow
 * configuration but an out-of-memory one -- and 1 GiB is already four orders of
 * magnitude past the useful range (HDF5's own sieve default is 64 KiB). Having
 * a stated maximum is also what lets every entry point reject a nonsense value
 * with the same message instead of each one inventing its own bound. */
#define H5FD_CLIO_SIEVE_MAX_LIM ((unsigned long long)1 << 30)

/* 0 (coalescing off) through the limit above. Applied to every source of the
 * value: the config string, and a driver-info block an application built by
 * hand and passed to H5Pset_driver. */
static inline bool H5FD__clio_sieve_valid(size_t v) {
  return (unsigned long long)v <= H5FD_CLIO_SIEVE_MAX_LIM;
}

/* Default policy when a file is opened without a driver-specific FAPL
 * (e.g. H5Pset_driver(fapl, driver, NULL)): cache on, coalescing on. */
static const H5FD_clio_fapl_t H5FD_clio_fapl_default_g = {
    /*cache_enabled*/ 1, /*fsync_on_flush*/ 0,
    /*sieve_max*/ H5FD_CLIO_SIEVE_MAX_DEF};

/*
 * Apply the driver config string, if the FAPL carries one.
 *
 * Reached via HDF5_DRIVER_CONFIG=... or H5Pset_driver_by_name(fapl,
 * "clio_vfd", "..."). The grammar is the shared CLIO one -- key=value pairs
 * separated by ';' -- which is the dialect the registered HDF5 VOL connectors
 * already use, so a user spells CLIO the way they spell the rest of the stack.
 *
 * Recognised keys:
 *   cache=0|1|on|off|true|false|yes|no   the CTE tier
 *   fsync=0|1|on|off|true|false|yes|no   fsync(2) inside the flush callback
 *                                        (default off; see H5FD__clio_flush)
 *   sieve=<bytes>                        vector-I/O coalescing window
 *                                        (0 disables; default 65536)
 *
 * An unrecognised key is an ERROR, not a shrug. A config string is something a
 * person typed, and a parser that ignores what it does not understand converts
 * a typo into "the knob you set did nothing" -- discovered, if ever, as a
 * performance mystery. Returns false with the reason on HDF5's error stack.
 */
static bool H5FD__clio_apply_config_str(hid_t fapl_id, H5FD_clio_fapl_t *fa) {
  const ssize_t len = H5Pget_driver_config_str(fapl_id, nullptr, 0);
  if (len <= 0) return true;  /* absent or empty: nothing to apply */

  std::string raw(static_cast<size_t>(len) + 1, '\0');
  if (H5Pget_driver_config_str(fapl_id, &raw[0], raw.size()) < 0) {
    H5FD_CLIO_ERROR("could not read the driver config string from the FAPL");
    return false;
  }
  raw.resize(static_cast<size_t>(len));

  std::map<std::string, std::string> kv;
  std::string err;
  if (!clio::cte::adapter::ParseConfigStr(raw, &kv, &err)) {
    H5FD_CLIO_ERROR(("driver config string: " + err).c_str());
    return false;
  }
  for (const auto &e : kv) {
    if (e.first == "cache") {
      bool on = true;
      if (!clio::cte::adapter::ConfigParseBool(e.second, &on)) {
        H5FD_CLIO_ERROR(("driver config: cache='" + e.second +
                         "' is not a boolean (use 0/1/on/off/true/false)")
                            .c_str());
        return false;
      }
      fa->cache_enabled = on ? 1 : 0;
    } else if (e.first == "fsync") {
      bool on = false;
      if (!clio::cte::adapter::ConfigParseBool(e.second, &on)) {
        H5FD_CLIO_ERROR(("driver config: fsync='" + e.second +
                         "' is not a boolean (use 0/1/on/off/true/false)")
                            .c_str());
        return false;
      }
      fa->fsync_on_flush = on ? 1 : 0;
    } else if (e.first == "sieve") {
      /* strtoull WRAPS a negative instead of rejecting it -- "-1" parses as
         ULLONG_MAX with errno untouched, which would install a SIZE_MAX
         window. Refuse the sign up front; detecting the wrap afterwards is not
         possible, since ULLONG_MAX is also a legitimate spelling of a value
         this driver would reject for being too large anyway. */
      const std::string &sv = e.second;
      const size_t first = sv.find_first_not_of(" \t");
      if (first == std::string::npos || sv[first] == '-') {
        H5FD_CLIO_ERROR(("driver config: sieve='" + sv +
                         "' is not a byte count (must be >= 0)")
                            .c_str());
        return false;
      }
      errno = 0;
      char *endp = nullptr;
      const unsigned long long v = strtoull(sv.c_str(), &endp, 0);
      if (errno != 0 || endp == sv.c_str() || (endp && *endp != '\0')) {
        H5FD_CLIO_ERROR(("driver config: sieve='" + sv +
                         "' is not a byte count")
                            .c_str());
        return false;
      }
      /* Bound before the narrowing cast: on a 32-bit size_t the cast alone
         would silently truncate, and a window larger than the limit is
         refused on every platform for the same reason. */
      if (v > H5FD_CLIO_SIEVE_MAX_LIM) {
        H5FD_CLIO_ERROR(("driver config: sieve='" + sv +
                         "' exceeds the maximum coalescing window (1 GiB)")
                            .c_str());
        return false;
      }
      fa->sieve_max = (size_t)v;
    } else {
      H5FD_CLIO_ERROR(("driver config: unknown key '" + e.first +
                       "' (this driver accepts: cache, fsync, sieve)")
                          .c_str());
      return false;
    }
  }
  return true;
}


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
  /* Opaque per-platform identity: dev/ino on POSIX, volume serial + file
   * index on Windows, where st_ino is always 0. See H5FDclio_compat.h. */
  clio_vfd_file_id_t file_id;
  clio::vfdtrace::FileTrace *trace; /* byte-altitude telemetry; null when off */
  /* May the tier answer for this file? Decided once at open; default-refuse. */
  bool tier_coherent;
  /* The tier copy may be incomplete (a populate, truncate, invalidation or
     cache close failed), so it must not be stamped. */
  bool cache_degraded;
  /* Identity at open, NULL if unstattable. Compared at close so a file that
     changed underneath the session is not stamped over stale cached bytes. */
  char *open_stamp_;
} H5FD_clio_t;

/* Was this file opened with write intent? H5F_ACC_RDWR is what open() keyed
 * its O_RDWR/O_RDONLY choice off, so it is the same question, asked later. */
#define H5FD_CLIO_WRITABLE(f) (((f)->flags & H5F_ACC_RDWR) != 0)

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

  /* The ordering argument that makes the guard work is that this atexit()
     call follows the one H5open() already made for H5_term_library. Every
     entry point that arms it is reached after H5open(), so arming it at all
     of them is safe and none of them can be the one that gets missed. */
  H5FD__clio_install_exit_guard();

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
  /* Belt and braces. Whatever route selected this driver, a file open is
     unconditionally after H5open(), so the guard is armed no later than the
     first file that could still be open when exit handlers start running. */
  H5FD__clio_install_exit_guard();

  // Argument validation, sec2 parity. Without these a NULL name dereferences in
  // HasClioPrefix below, and an out-of-range maxaddr is accepted despite the
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

  /* Driver config string. HDF5 does not hand this to a callback the way it does
     for a VOL connector -- H5Pset_driver_by_name / HDF5_DRIVER_CONFIG copy the
     string onto the FAPL and the driver is expected to fetch it. So the pull
     happens here, at the one point that sees the FAPL for a real open.
     This is the half of "configurable without source edits" that was missing:
     HDF5_DRIVER=clio_vfd already worked, but there was no way to say anything
     ABOUT the driver without calling H5Pset_fapl_clio from the application. */
  if (!H5FD__clio_apply_config_str(fapl_id, &fa)) {
    /* Message already on the error stack. A config string we cannot understand
       fails the open rather than being ignored: the caller asked for something
       specific, and silently giving them the default is how a user ends up
       believing a knob is set when it is not. */
    return nullptr;
  }

  /* Validate the coalescing window HERE, after both sources have had their say.
     The config string vets its own input, but a driver-info block does not go
     through it at all: H5Pset_driver(fapl, id, &fa) takes whatever struct the
     application filled in, so a value that never passed a parser can still
     reach this point. sieve_max sizes a scratch allocation, which makes a
     nonsense one an allocation failure rather than a slow open -- and this is
     the single place every FAPL path converges on. Fail closed, as the config
     string does. */
  if (!H5FD__clio_sieve_valid(fa.sieve_max)) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("driver FAPL: the vector-I/O coalescing window "
                    "(sieve_max) exceeds the maximum of 1 GiB");
    return nullptr;
  }

  /* Environment opt-out, applied AFTER the config string and last, so that
     CLIO_VFD_CACHE=0 can always force the tier off no matter what else asked
     for it -- the documented "either is sufficient to DISABLE" rule. Applied
     before the runtime probe below so it also skips the attach attempt (and
     therefore its retry timeout) rather than attaching and then not using it. */
  if (fa.cache_enabled && !H5FD__clio_cache_env_enabled()) {
    fa.cache_enabled = 0;
  }

  /* Same shape, opposite direction: the environment can turn the flush-time
     durability barrier ON for a file whose FAPL did not ask for it. */
  if (!fa.fsync_on_flush && H5FD__clio_fsync_env_forced()) {
    fa.fsync_on_flush = 1;
  }

  /* A read-only open has nothing to gain from the cache tier, so do not attach
     one. The tier is write-populated and not yet served on reads:
     H5FD__clio_do_read goes to the authoritative descriptor unconditionally,
     and H5FD__clio_do_write -- the only writer -- cannot run on a file HDF5
     opened without H5F_ACC_RDWR. The handle would therefore be opened, never
     used, and closed.
     It is not free: the attach waits on a CFS Open task here and close() waits
     on a Close, so a read-only open/close cycle pays two blocking round trips --
     and the open one is serviced by Runtime::Open, which for an open without
     O_CREAT awaits a TagQuery and then a GetTagSize inside the runtime. A
     workload that opens and closes files in a loop spends its entire time there:
     netCDF-C's nc_test4/tst_files4 does 32768 read-only open/close cycles, i.e.
     ~66k client round trips for a tier no byte is ever read from.
     Decided before the H5FD__clio_cache_available() probe below so a read-only
     open also skips the runtime attach and its retry timeout. */
  if (fa.cache_enabled && !(H5F_ACC_RDWR & flags)) {
    fa.cache_enabled = 0;
  }

  // Attach to the CLIO runtime ONLY when this file wants the cache tier: the
  // native file is authoritative, so the native-only configuration is a
  // complete driver on its own and must not require CLIO to be running.
  // A failed attach degrades this file to native-only rather than failing the
  // open -- the cache is a performance tier, not a correctness one.
  if (fa.cache_enabled && !H5FD__clio_cache_available()) {
    // ...unless the caller set CLIO_REQUIRE_RUNTIME, which says a silent
    // native-only open is worse than a failed one. Only checked when the cache
    // was WANTED: an explicit opt-out is a choice, not a failure.
    if (clio::adapter::RequireRuntime()) {
      H5FD_CLIO_ERROR(clio::adapter::RequireRuntimeMessage());
      HLOG(kError, "{} -- {}", clio::adapter::RequireRuntimeMessage(), name);
      return nullptr;
    }
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

  // A clio::-marked name is NOT accepted: the marker is CLIO-internal, and
  // every name HDF5 holds must be a real path it can stat() -- which is what
  // lets query() advertise POSIX_COMPAT_HANDLE / DEFAULT_VFD_COMPATIBLE.
  // Refused rather than silently stripped, because HDF5 would keep the marked
  // name in its own bookkeeping even if we stripped it for open(2).
  if (clio::cte::filesystem::HasClioPrefix(name)) {
    H5FD_CLIO_ERROR("the clio:: prefix is not accepted by this driver; pass "
                    "the plain filesystem path (CLIO is selected by "
                    "HDF5_DRIVER=clio_vfd or H5Pset_fapl_clio, not by the "
                    "filename)");
    return nullptr;
  }

  // The AUTHORITATIVE store is a real on-disk native HDF5 file at this exact
  // path, so standard tools (h5dump/h5ls) read it live.
  // clio_vfd_open, not open(2): MSVC has no POSIX open, so the Windows port
  // routes every descriptor call through the compat shims.
  std::string native_path = name;
  int posix_fd = clio_vfd_open(native_path.c_str(), o_flags,
                               H5FD_CLIO_POSIX_CREATE_MODE_RW);
  if (posix_fd < 0) {
    // Fail-closed: no authoritative file => the open fails. We do not proceed
    // with a cache-only file. Record errno on the driver error stack.
    H5FD_CLIO_ERROR("open() of authoritative native file failed");
    return nullptr;
  }

  // CTE cache handle. Best-effort: the authoritative file is already open, so
  // a cache-open failure must not sink this one -- fd == -1 just means no
  // cache this session.
  //
  // O_RDWR|O_CREAT regardless of the application's flags: the tier copy is
  // ours, and a read-only HDF5 open must still be able to create it, stamp it,
  // and populate on a miss. O_TRUNC is preserved.
  int fd = -1;
#if H5FD_CLIO_HAVE_CACHE_TIER
  if (fa.cache_enabled) {
    int cache_flags = (o_flags & ~O_ACCMODE) | O_RDWR | O_CREAT;
    fd = CLIO_CFS_CLIENT->OpenFd(name, cache_flags,
                                 H5FD_CLIO_POSIX_CREATE_MODE_RW);
    HLOG(kDebug, "");
  }
#endif

  // Coherence gate. The tier may only answer for this file if the stamp taken
  // at its last close still describes the file as it is NOW. Anything else --
  // mismatch, no stamp, unstattable -- means the cached copy cannot be vouched
  // for, so it is dropped and this session starts with an empty tier.
  //
  // Gated on a cache handle actually existing. Every step here is a CFS RPC,
  // and where no filesystem pool is composed those go to a pool that is not
  // there: issuing them before knowing a handle could be had is what wedged
  // the compat suite, whose runtime composes only bdev + cte_core. fd >= 0 is
  // the proof that the pool answered.
  //
  // A truncating open skips the check: O_TRUNC already emptied the tier copy,
  // so there is nothing left to be stale.
  bool tier_coherent = false;
  bool cache_degraded = false;
#if H5FD_CLIO_HAVE_CACHE_TIER
  if (fd >= 0 && H5FD__clio_read_tier_on()) {
    if (o_flags & O_TRUNC) {
      // OpenFd fires its own truncate for O_TRUNC but discards the result and
      // still hands back a valid fd, so "the tier copy is empty" was an
      // assumption. Do it explicitly and check: if the previous file's pages
      // survive, marking the tier coherent would serve them.
      if (CLIO_CFS_CLIENT->FtruncateFd(fd, 0) == 0) {
        tier_coherent = true;
      } else {
        cache_degraded = true;
      }
    } else {
      const std::string want = H5FD__clio_stamp_of(native_path.c_str());
      auto got = CLIO_CFS_CLIENT->AsyncGetxattr(name, H5FD_CLIO_STAMP_XATTR);
      got.Wait();
      const bool have = got->GetReturnCode() == 0 && got->found_ == 1;
      // want.empty() means the file could not be stat'd: no verdict is
      // possible, so refuse rather than compare against nothing.
      tier_coherent = have && !want.empty() && got->value_.str() == want;
      if (!tier_coherent) {
        // Drop the cached copy and take a fresh handle on the empty one. This
        // removes only the tier's pages and xattrs -- CFS is a blob-backed
        // namespace with no native backing, so it cannot touch the user's
        // file.
        //
        // The drop is CHECKED, not assumed. If it fails the stale pages
        // survive, and while this session is safe (it will not read them), its
        // close must not stamp -- a stamp would tell the NEXT session that a
        // tier still holding pre-change pages describes the file.
        CLIO_CFS_CLIENT->CloseFd(fd);
        const int rc = CLIO_CFS_CLIENT->RemovePath(name);
        H5FDclio_cache_stale_invalidations_g++;
        fd = CLIO_CFS_CLIENT->OpenFd(name,
                                     (o_flags & ~O_ACCMODE) | O_RDWR | O_CREAT,
                                     H5FD_CLIO_POSIX_CREATE_MODE_RW);
        if (rc == 0 && fd >= 0) {
          // The copy really is gone and the handle really is fresh, so the
          // tier is empty and consistent with the file: this session may use
          // it. Without this the session paid read-through on every miss with
          // no possibility of a hit -- and since a file's first open never
          // has a stamp, that was the common case.
          tier_coherent = true;
        } else {
          cache_degraded = true;
        }
      }
    }
  }
#endif

  /* Create the new file struct */
  H5FD_clio_t *file = (H5FD_clio_t *)calloc(1, sizeof(H5FD_clio_t));
  if (file == NULL) {
    // Out of memory: release the handles we already opened instead of leaking
    // them (and dereferencing a NULL file). calloc does not reliably set errno,
    // so set it explicitly for an accurate error message.
    errno = ENOMEM;
    H5FD_CLIO_ERROR("calloc() of VFD file struct failed");
    clio_vfd_close(posix_fd);
#if H5FD_CLIO_HAVE_CACHE_TIER
    if (fd >= 0) {
      CLIO_CFS_CLIENT->CloseFd(fd);
    }
#endif
    return nullptr;
  }

  // Byte-altitude telemetry. Observe-only and null unless CLIO_VFD_TRACE is
  // set, so the disabled path costs one cached bool.
  file->trace = clio::vfdtrace::OpenFile(native_path);

  // Identity + size from ONE fstat of the authoritative file. cmp() depends on
  // dev/ino, so a failed fstat is fail-closed: without identity the library
  // could not tell this file apart from another and might open it twice.
  clio_vfd_file_id_t file_id;
  clio_vfd_off_t file_size = 0;
  if (clio_vfd_fstat(posix_fd, &file_id, &file_size) < 0) {
    H5FD_CLIO_ERROR("fstat() of authoritative native file failed");
    clio_vfd_close(posix_fd);
#if H5FD_CLIO_HAVE_CACHE_TIER
    if (fd >= 0) {
      CLIO_CFS_CLIENT->CloseFd(fd);
    }
#endif
    free(file);
    return nullptr;
  }

  /* Pack file */
  file->filename_ = strdup(name);
  file->tier_coherent = tier_coherent;
  file->cache_degraded = cache_degraded;
  {
    const std::string ident = H5FD__clio_stamp_of(native_path.c_str());
    file->open_stamp_ = ident.empty() ? nullptr : strdup(ident.c_str());
  }
  if (!file->filename_) {
    errno = ENOMEM;
    H5FD_CLIO_ERROR("strdup() of file name failed");
    clio_vfd_close(posix_fd);
#if H5FD_CLIO_HAVE_CACHE_TIER
    if (fd >= 0) {
      CLIO_CFS_CLIENT->CloseFd(fd);
    }
#endif
    free(file);
    return nullptr;
  }
  file->fd = fd;
  file->posix_fd = posix_fd;
  file->flags = flags;
  file->fa = fa;
  file->file_id = file_id;

  // EOF is the authoritative on-disk size (durable across reopen/append), not a
  // session-local counter or the cache's logical size.
  file->eof = (haddr_t)file_size;

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

  // Write the summary before the handles go. No-op when tracing is off.
  clio::vfdtrace::CloseFile(file->trace);
  file->trace = nullptr;
  // fsync + close the authoritative native file first -- a successful close is
  // a durability barrier (no pending dirty state), and the on-disk file is a
  // complete valid native HDF5 image afterward.
  if (file->posix_fd >= 0) {
    /* Only a file we could have WRITTEN has anything to persist, and asking to
     * persist a read-only one is not portable: fsync(2) tolerates a read-only
     * descriptor, but the Windows equivalent (_commit -> FlushFileBuffers)
     * requires write access. The debug CRT asserts outright there ("Invalid
     * file descriptor", ucrt commit.cpp), and the release CRT quietly returns
     * EBADF -- which this fail-closed branch would then turn into a failed
     * H5Fclose on a file that was never dirty. Reproduced by the no-runtime
     * test's read-only reopen, which is the second open of the same path. */
    if (H5FD_CLIO_WRITABLE(file) && clio_vfd_fsync(file->posix_fd) < 0) {
      H5FD_CLIO_ERROR("fsync() on close failed");
      ret_value = FAIL; /* fail-closed: a close that did not persist fails */
    }
    // close() itself can fail (EIO, and notably deferred write errors on NFS).
    // A close that reports an error has not necessarily persisted, so it is
    // fail-closed too -- the whole point of the barrier.
    if (clio_vfd_close(file->posix_fd) < 0) {
      H5FD_CLIO_ERROR("close() of authoritative native file failed");
      ret_value = FAIL;
    }
  }
  // Release the CTE cache handle, if this session had one -- unless the
  // process is already running its exit handlers, in which case the client
  // that would service the close no longer has a receive thread and the wait
  // never returns. The handle goes down with the process.
  //
  // Closed before the stamp is decided: CloseFd is the only place a deferred
  // CFS write failure surfaces, since deferred writes report success at submit.
#if H5FD_CLIO_HAVE_CACHE_TIER
  const bool cache_live = H5FD__clio_cache_live(file->fd);
  if (cache_live) {
    if (CLIO_CFS_CLIENT->CloseFd(file->fd) != 0) {
      file->cache_degraded = true;
    }
    HLOG(kDebug, "");
  }
  file->fd = -1;

  // Stamp only what this session can vouch for. Withheld when the cached copy
  // may be incomplete (degraded), when the file moved underneath the session --
  // stamping the new identity over a tier holding the old bytes is the exact
  // staleness this prevents -- or when mtime is too young to discriminate a
  // later write. Taken after the native fsync/close above, so the mtime it
  // embeds is the one the file will keep.
  //
  // Skipped entirely once exit handlers are running: these are blocking CFS
  // calls, and a client being torn down can no longer answer them.
  if (cache_live && file->filename_ != nullptr && H5FD__clio_read_tier_on()) {
    const std::string now = H5FD__clio_stamp_of(file->filename_);
    const bool changed = file->open_stamp_ == nullptr || now.empty() ||
                         now != std::string(file->open_stamp_);
    const std::string stamp =
        (file->cache_degraded || changed ||
         clio::adapter::stamp::Ambiguous(file->filename_))
            ? std::string()
            : now;
    if (stamp.empty()) {
      auto rm = CLIO_CFS_CLIENT->AsyncRemovexattr(
          std::string(file->filename_), H5FD_CLIO_STAMP_XATTR);
      rm.Wait();
    } else {
      // flags 0 = create-or-replace: a file opened, closed and reopened must
      // overwrite its own previous stamp rather than fail with EEXIST.
      auto set = CLIO_CFS_CLIENT->AsyncSetxattr(
          std::string(file->filename_), H5FD_CLIO_STAMP_XATTR, stamp, 0);
      set.Wait();
      if (set->GetReturnCode() != 0) {
        HLOG(kWarning,
             "clio-vfd: coherence stamp for {} failed to store (rc={}); the "
             "cached copy will be dropped on the next open",
             file->filename_, set->GetReturnCode());
      }
    }
  }
#endif
  if (file->open_stamp_) {
    free(file->open_stamp_);
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
  return clio_vfd_cmp_file_id(&f1->file_id, &f2->file_id);
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
    /* Advertise sec2's feature set. The backend is a real byte-addressable
     * POSIX file, so HDF5's metadata aggregation / accumulation / data-sieve /
     * small-data aggregation all apply; returning 0 silently disables them.
     *
     * POSIX_COMPAT_HANDLE and DEFAULT_VFD_COMPATIBLE both require that every
     * name HDF5 holds be a real path (H5F__build_actual_name stat()s it) and
     * that the image be a plain native file. Both hold only because a
     * clio::-marked name is refused at open.
     *
     * OPEN QUESTION: a failed open under H5P_DEFAULT + HDF5_DRIVER=clio_vfd
     * once showed H5FD__sec2_open in the error stack, i.e. HDF5 serviced it
     * with the default driver. Never reproduced with an explicit driver FAPL,
     * and the flag was never toggled to confirm. Harmless while the tier is
     * populate-only (reads go to the native file either way), but a read tier
     * HDF5 may bypass at its discretion cannot answer for the file -- measure
     * which paths this flag diverts before relying on it then. */
    *flags = 0;
    *flags |= H5FD_FEAT_AGGREGATE_METADATA;   /* metadata block aggregation */
    *flags |= H5FD_FEAT_ACCUMULATE_METADATA;  /* metadata accumulation      */
    *flags |= H5FD_FEAT_DATA_SIEVE;           /* data sieving               */
    *flags |= H5FD_FEAT_AGGREGATE_SMALLDATA;  /* small raw-data aggregation */
    *flags |= H5FD_FEAT_SUPPORTS_SWMR_IO;     /* flock + real file          */
    *flags |= H5FD_FEAT_POSIX_COMPAT_HANDLE;  /* get_handle yields a real fd */
    *flags |= H5FD_FEAT_DEFAULT_VFD_COMPATIBLE; /* image is a plain sec2 file */
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
 *     A future read tier must track which byte ranges are actually populated:
 *     the CFS chimod zero-fills holes and reports a full read, so a naive
 *     lookup would return stale zeros as data.
 *
 *   H5FD__clio_do_write: write-through SIZE bytes at ADDR to the authoritative
 *     native file (fail-closed on short/failed write), best-effort populate the
 *     CTE cache tier when a handle exists, and advance the session EOF.
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_do_read(H5FD_clio_t *file, haddr_t addr, size_t size,
                                 void *buf) {
  if (H5FD_CLIO_REGION_INVALID(addr, size)) {
    errno = EINVAL;
    H5FD_CLIO_ERROR("read region is undefined or out of range");
    return FAIL;
  }
  // Serve from the tier only when it holds the WHOLE range and this session's
  // coherence check passed.
  //
  // TryReadShmResident rather than a plain CFS read: CFS zero-fills holes and
  // reports a full read. Right for a filesystem; wrong here, where a range the
  // tier does not hold is not zeros but bytes living in the native file.
  //
  // All-or-nothing per request: splitting a read between tier and file would
  // mean tracking which half came from where on every failure path.
#if H5FD_CLIO_HAVE_CACHE_TIER
  if (H5FD__clio_read_tier_on() && file->tier_coherent && file->fd >= 0 &&
      file->filename_ != nullptr) {
    const ssize_t served = CLIO_CFS_CLIENT->TryReadShmResident(
        file->filename_, static_cast<clio::run::u64>(addr), buf, size);
    if (served == static_cast<ssize_t>(size)) {
      H5FDclio_cache_read_hits_g++;
      if (getenv("CLIO_VFD_DEBUG"))
        fprintf(stderr, "[vfd] READ(tier) addr=%llu size=%llu\n",
                (unsigned long long)addr, (unsigned long long)size);
      return SUCCEED;
    }
    // A SHORT read is refused too, not stitched: the tier held only part of
    // the range, and the rest is the file's. Fall through and take it all from
    // the file rather than track a split.
    H5FDclio_cache_read_misses_g++;
  }
#endif

  char *dst = static_cast<char *>(buf);
  size_t remaining = size;
  clio_vfd_off_t off = static_cast<clio_vfd_off_t>(addr);

  // Loop rather than issue one pread. A short return is NOT proof of EOF: the
  // kernel caps a single transfer (0x7ffff000 on Linux) and a signal can cut
  // one short via EINTR. Zero-filling whatever a single pread did not deliver
  // hands back zeros in place of real data WITH a success status. Only a pread
  // returning exactly 0 is true EOF; HDF5 treats the file as a flat byte
  // array, so the tail past EOF is legitimately zero-filled.
  while (remaining > 0) {
    const size_t cap = H5FD__clio_max_io_bytes();
    size_t want = (remaining > cap) ? cap : remaining;
    clio_vfd_ssize_t got = clio_vfd_pread(file->posix_fd, dst, want, off);
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
  // Populate on a miss. Not an optimisation: a writing session's own close
  // flushes the file, so its mtime is always fresh and its stamp always
  // withheld -- a tier filled only by writes is never stamped, and so never
  // readable. The open either matched the stamp or dropped the copy, so what
  // is written here came from the file as it now stands.
#if H5FD_CLIO_HAVE_CACHE_TIER
  if (H5FD__clio_read_tier_on() && file->fd >= 0) {
    if (CLIO_CFS_CLIENT->PwriteFd(file->fd, buf, size,
                                  static_cast<off_t>(addr)) < 0) {
      H5FDclio_cache_write_failures_g++;
      // A populate that failed leaves a hole the tier does not know about, so
      // this copy can no longer be vouched for as a whole. Counting it is not
      // enough: without latching, close would still stamp it.
      file->cache_degraded = true;
    }
  }
#endif

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
  clio_vfd_off_t off = static_cast<clio_vfd_off_t>(addr);

  // Same loop as the read path: chunk to stay under the kernel's per-call cap
  // and retry on EINTR. A short pwrite is a partial transfer to be continued,
  // not a failure -- unlooped, a >2 GiB write failed outright.
  while (remaining > 0) {
    const size_t cap = H5FD__clio_max_io_bytes();
    size_t want = (remaining > cap) ? cap : remaining;
    clio_vfd_ssize_t put = clio_vfd_pwrite(file->posix_fd, src, want, off);
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
  //
  // Gated on the read tier, like every other site that touches the tier: the
  // open coherence check, the read that serves from it, the read-through
  // populate below it, and the close that stamps it. This one was the
  // exception, and with CLIO_VFD_READ_TIER unset -- the default -- that made it
  // pure cost. Nothing in the process can read what it writes, because
  // tier_coherent is only ever set inside the same gate; and nothing later can
  // either, because close only stamps inside that gate, so the next session
  // that does enable reads finds no stamp and drops the copy. The read path's
  // own comment already states the invariant: "a tier filled only by writes is
  // never stamped, and so never readable."
  //
  // Measured on nc_perf_tst_attsperf (macOS arm64, HDF5 write callbacks are
  // frequently a few bytes, one PwriteFd each): 705 s with the populate, 10.8 s
  // without, against a 10.8 s native baseline. nc_perf_tst_files3: 223 s vs
  // 12.9 s vs 13.8 s.
#if H5FD_CLIO_HAVE_CACHE_TIER
  if (H5FD__clio_cache_live(file->fd) && H5FD__clio_read_tier_on()) {
    if (CLIO_CFS_CLIENT->PwriteFd(file->fd, buf, size, static_cast<off_t>(addr)) < 0) {
      H5FDclio_cache_write_failures_g++;
      HLOG(kWarning,
           "CTE cache populate failed at addr={} size={} (native file is "
           "unaffected and remains authoritative)",
           (unsigned long long)addr, (unsigned long long)size);
    }
  }
#endif
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

/* Time one byte-level access and hand it to the telemetry producer. Observe
 * only: the return code is the callee's, untouched. Kept as one helper so the
 * scalar and vector paths cannot drift in what they record -- a vectored
 * workload tracing empty would be a silent hole in the data, not an obvious
 * one. */
/* extern "C++": this file is inside an extern "C" block for the HDF5 callback
   ABI, and a template cannot have C linkage. The callbacks themselves stay C. */
extern "C++" {
template <typename Fn>
static herr_t H5FD__clio_traced(H5FD_clio_t *file, clio::vfdtrace::Op op,
                                int mem_type, haddr_t addr, size_t size,
                                Fn &&fn) {
  if (!file || !file->trace) return fn();
  const auto t0 = std::chrono::steady_clock::now();
  const herr_t rc = fn();
  const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  clio::vfdtrace::Record(file->trace, op, mem_type,
                         static_cast<uint64_t>(addr),
                         static_cast<uint64_t>(size),
                         static_cast<uint64_t>(dt < 0 ? 0 : dt));
  return rc;
}
}  // extern "C++"

static herr_t H5FD__clio_read(H5FD_t *_file, H5FD_mem_t type, hid_t dxpl_id,
                                haddr_t addr, size_t size, void *buf) {
  (void)dxpl_id;
  H5FD_clio_t *f = (H5FD_clio_t *)_file;
  return H5FD__clio_traced(f, clio::vfdtrace::Op::kRead, (int)type, addr, size,
                           [&] { return H5FD__clio_do_read(f, addr, size, buf); });
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
  H5FD_clio_t *f = (H5FD_clio_t *)_file;
  return H5FD__clio_traced(f, clio::vfdtrace::Op::kWrite, (int)type, addr, size,
                           [&] { return H5FD__clio_do_write(f, addr, size, buf); });
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
/* Shorthand state for a vector's sizes[]/types[] arrays.
 *
 * Both arrays may be "shortened": a 0 size (H5FD_MEM_NOLIST type) at i > 0
 * means this and every later element reuse the last explicit value. The rule is
 * monotone -- once shortened, always shortened -- so the state can be
 * snapshotted at the first element of a group and replayed over that group,
 * which is what the coalescing loops below need in order to walk a group twice
 * (once to size it, once to copy it). */
typedef struct H5FD_clio_vecst_t {
  size_t size;
  bool size_fixed;
  H5FD_mem_t type;
  bool type_fixed;
} H5FD_clio_vecst_t;

static inline void H5FD__clio_vecst_init(H5FD_clio_vecst_t *st,
                                         const size_t sizes[],
                                         const H5FD_mem_t types[]) {
  st->size = sizes[0];
  st->size_fixed = false;
  st->type = types ? types[0] : H5FD_MEM_DEFAULT;
  st->type_fixed = false;
}

/* Advance the state to element k; afterwards st->size/st->type are k's. */
static inline void H5FD__clio_vecst_step(H5FD_clio_vecst_t *st, uint32_t k,
                                         const size_t sizes[],
                                         const H5FD_mem_t types[]) {
  if (!st->size_fixed) {
    if (k > 0 && sizes[k] == 0)
      st->size_fixed = true;
    else
      st->size = sizes[k];
  }
  if (types && !st->type_fixed) {
    if (k > 0 && types[k] == H5FD_MEM_NOLIST)
      st->type_fixed = true;
    else
      st->type = types[k];
  }
}

/* Find the longest run of elements starting at `i` whose spanning region stays
 * within `cap` bytes. Returns the element index one past the run, and stores
 * the span end and the sum of the elements' sizes.
 *
 * Elements must be ascending for a run to form; HDF5's vectors are, but a
 * non-monotone entry simply closes the group rather than being mishandled. */
static uint32_t H5FD__clio_vec_group(uint32_t i, uint32_t count,
                                     const haddr_t addrs[],
                                     const size_t sizes[],
                                     const H5FD_mem_t types[],
                                     H5FD_clio_vecst_t st, size_t cap,
                                     haddr_t *end_out, size_t *payload_out) {
  haddr_t end = addrs[i] + st.size;
  size_t payload = st.size;
  uint32_t j = i + 1;
  /* An element that is already larger than the window stays a group of one.
     Coalescing around it cannot honour the window -- the scratch buffer would
     be sized by that element, not by `cap` -- and there is nothing to gain:
     the element is serviced as a single I/O either way. */
  if (cap > 0 && (uint64_t)(end - addrs[i]) <= (uint64_t)cap) {
    while (j < count) {
      H5FD_clio_vecst_t nxt = st;
      H5FD__clio_vecst_step(&nxt, j, sizes, types);
      if (addrs[j] < addrs[i]) break;                 /* not ascending */
      const haddr_t e = addrs[j] + nxt.size;
      if (e <= addrs[i]) break;                       /* degenerate */
      /* Test the span the group WOULD have, not this element's own end. They
         differ whenever an element falls inside the span already accumulated
         (a shorter element after a longer one), and testing the element's end
         there would admit it while the span -- and with it the scratch
         allocation -- stayed above the window. */
      const haddr_t nend = (e > end) ? e : end;
      if ((uint64_t)(nend - addrs[i]) > (uint64_t)cap) break;
      st = nxt;
      end = nend;
      payload += nxt.size;
      j++;
    }
  }
  *end_out = end;
  *payload_out = payload;
  return j;
}

/* Scratch for one coalesced group. Bounded by the sieve window, so this is a
 * 64 KiB allocation by default and it is reused for the whole vector call.
 *
 * The allocation is the one step here that can fail, and it must fail as this
 * driver's failure rather than as an exception: these callbacks are reached
 * from HDF5's C frames, and unwinding a C++ exception through them is
 * undefined. Convert it to FAIL, with the reason on HDF5's error stack, and
 * let the caller return FAIL to the library. */
static herr_t H5FD__clio_vec_scratch(std::vector<char> *buf, size_t need) {
  if (buf->size() >= need) {
    return SUCCEED;
  }
  try {
    buf->resize(need);
  } catch (const std::exception &ex) {
    errno = ENOMEM;
    H5FD_CLIO_ERROR(
        ("could not allocate the " + std::to_string(need) +
         "-byte vector-I/O scratch buffer: " + ex.what()).c_str());
    return FAIL;
  } catch (...) {
    errno = ENOMEM;
    H5FD_CLIO_ERROR("could not allocate the vector-I/O scratch buffer");
    return FAIL;
  }
  return SUCCEED;
}

/*-------------------------------------------------------------------------
 * Coalescing vector I/O.
 *
 * HDF5 hands a driver that implements these callbacks the WHOLE selection in
 * one call -- for a strided slab of a contiguous dataset that is tens of
 * thousands of 4-byte elements (measured: count=65536, size=4, stride 1024).
 * Servicing them one at a time costs one pread/pwrite AND one CTE populate
 * round-trip per element, which made such writes ~200x slower than sec2
 * (iowarp/clio-core#980).
 *
 * The library's own answer to this pattern is the data sieve buffer, but
 * implementing vector I/O is exactly what turns selection I/O on and takes the
 * sieve path away (H5Dio.c, H5D__ioinfo_adjust) -- so the driver has to do it
 * itself. Consecutive elements are grouped while their spanning region stays
 * within the sieve window and serviced as one I/O:
 *
 *   - a group whose elements exactly tile the span needs no read: gather and
 *     write once;
 *   - otherwise read the span, patch the elements in, write the span back.
 *     The span ends at the last element's end, so nothing outside the range the
 *     unmerged path would have written is touched.
 *
 * `sieve=0` in the driver config restores the element-at-a-time behaviour.
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_read_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                     H5FD_mem_t types[], haddr_t addrs[],
                                     size_t sizes[], void *bufs[]) {
  (void)dxpl;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  H5FDclio_read_vector_calls_g++;
  if (count == 0) {
    return SUCCEED;
  }
  const size_t cap = file->fa.sieve_max;
  H5FD_clio_vecst_t st;
  H5FD__clio_vecst_init(&st, sizes, types);
  std::vector<char> scratch;

  uint32_t i = 0;
  while (i < count) {
    H5FD_clio_vecst_t gst = st;
    H5FD__clio_vecst_step(&gst, i, sizes, types);
    haddr_t end;
    size_t payload;
    const uint32_t j = H5FD__clio_vec_group(i, count, addrs, sizes, types, gst,
                                            cap, &end, &payload);
    if (j == i + 1) {
      if (H5FD__clio_traced(file, clio::vfdtrace::Op::kRead, (int)gst.type,
                            addrs[i], gst.size, [&] {
                              return H5FD__clio_do_read(file, addrs[i],
                                                        gst.size, bufs[i]);
                            }) < 0) {
        return FAIL;
      }
    } else {
      const size_t span = (size_t)(end - addrs[i]);
      if ((unsigned long)span > H5FDclio_vec_max_span_g)
        H5FDclio_vec_max_span_g = (unsigned long)span;
      if (H5FD__clio_vec_scratch(&scratch, span) < 0) return FAIL;
      if (H5FD__clio_traced(file, clio::vfdtrace::Op::kRead, (int)gst.type,
                            addrs[i], span, [&] {
                              return H5FD__clio_do_read(file, addrs[i], span,
                                                        scratch.data());
                            }) < 0) {
        return FAIL;
      }
      H5FD_clio_vecst_t cp = gst;
      for (uint32_t k = i; k < j; k++) {
        if (k > i) H5FD__clio_vecst_step(&cp, k, sizes, types);
        memcpy(bufs[k], scratch.data() + (size_t)(addrs[k] - addrs[i]), cp.size);
      }
    }
    for (uint32_t k = i; k < j; k++) H5FD__clio_vecst_step(&st, k, sizes, types);
    i = j;
  }
  return SUCCEED;
} /* end H5FD__clio_read_vector() */

static herr_t H5FD__clio_write_vector(H5FD_t *_file, hid_t dxpl, uint32_t count,
                                      H5FD_mem_t types[], haddr_t addrs[],
                                      size_t sizes[], const void *bufs[]) {
  (void)dxpl;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  H5FDclio_write_vector_calls_g++;
  if (count == 0) {
    return SUCCEED;
  }
  const size_t cap = file->fa.sieve_max;
  H5FD_clio_vecst_t st;
  H5FD__clio_vecst_init(&st, sizes, types);
  std::vector<char> scratch;

  uint32_t i = 0;
  while (i < count) {
    H5FD_clio_vecst_t gst = st;
    H5FD__clio_vecst_step(&gst, i, sizes, types);
    haddr_t end;
    size_t payload;
    const uint32_t j = H5FD__clio_vec_group(i, count, addrs, sizes, types, gst,
                                            cap, &end, &payload);
    if (j == i + 1) {
      if (H5FD__clio_traced(file, clio::vfdtrace::Op::kWrite, (int)gst.type,
                            addrs[i], gst.size, [&] {
                              return H5FD__clio_do_write(file, addrs[i],
                                                         gst.size, bufs[i]);
                            }) < 0) {
        return FAIL;
      }
    } else {
      const size_t span = (size_t)(end - addrs[i]);
      if ((unsigned long)span > H5FDclio_vec_max_span_g)
        H5FDclio_vec_max_span_g = (unsigned long)span;
      if (H5FD__clio_vec_scratch(&scratch, span) < 0) return FAIL;
      /* Elements that exactly tile the span leave no bytes to preserve, so the
         read-modify part is skipped and this is a pure gather. */
      if (payload < span) {
        if (H5FD__clio_traced(file, clio::vfdtrace::Op::kRead, (int)gst.type,
                              addrs[i], span, [&] {
                                return H5FD__clio_do_read(file, addrs[i], span,
                                                          scratch.data());
                              }) < 0) {
          return FAIL;
        }
      }
      H5FD_clio_vecst_t cp = gst;
      for (uint32_t k = i; k < j; k++) {
        if (k > i) H5FD__clio_vecst_step(&cp, k, sizes, types);
        memcpy(scratch.data() + (size_t)(addrs[k] - addrs[i]), bufs[k], cp.size);
      }
      if (H5FD__clio_traced(file, clio::vfdtrace::Op::kWrite, (int)gst.type,
                            addrs[i], span, [&] {
                              return H5FD__clio_do_write(file, addrs[i], span,
                                                         scratch.data());
                            }) < 0) {
        return FAIL;
      }
    }
    for (uint32_t k = i; k < j; k++) H5FD__clio_vecst_step(&st, k, sizes, types);
    i = j;
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
 * Purpose:     Push this driver's buffers out to the file. There are none:
 *              every write is written through to the authoritative native file
 *              by pwrite(2) before the write callback returns, so by the time
 *              flush is called the data is already in the page cache and
 *              visible to every other process on the machine. That is what
 *              HDF5 asks a flush callback for, and it is what sec2 -- the
 *              reference driver this one is byte-for-byte compatible with --
 *              provides by having no flush callback at all.
 *
 *              This used to fsync(2) unconditionally, which is a DURABILITY
 *              barrier (survive a power cut), not a visibility one, and is
 *              strictly stronger than the callback's contract. It is also
 *              ruinously expensive at netCDF-4's call rate: netCDF calls
 *              H5Fflush from nc_enddef, so nc_test4/tst_atts -- which
 *              re-enters define mode 3 x 2^16 times -- issued ~200k fsyncs and
 *              went from 0.9s on sec2 to over 300s here (ctest timeout).
 *              tst_h_strings2, tst_h_compounds2 and the ncdump shell tests
 *              were the same shape.
 *
 *              The barrier is still available, because "my writes are on
 *              stable storage when H5Fflush returns" is a real requirement for
 *              some callers -- it is just opt-in now: fsync=1 in the driver
 *              config string (HDF5_DRIVER_CONFIG="fsync=1"), or
 *              CLIO_VFD_FSYNC=1 in the environment for a process that cannot
 *              reach the FAPL. close() fsyncs unconditionally either way, so
 *              a file that is closed normally is always durable; only the
 *              per-flush barrier is a choice.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t H5FD__clio_flush(H5FD_t *_file, hid_t dxpl_id, bool closing) {
  (void)dxpl_id;
  (void)closing;
  H5FD_clio_t *file = (H5FD_clio_t *)_file;
  if (!file->fa.fsync_on_flush) {
    return SUCCEED;
  }
  // Opt-in durability barrier; fail-closed so a flush that did not reach disk
  // never reports success. Writes are write-through, so the native file is the
  // only store holding data to flush.
  //
  // H5FD_CLIO_WRITABLE is a separate condition from the opt-in above, and both
  // are needed: fsync(2) tolerates a read-only descriptor, but the Windows
  // equivalent does not -- _commit() is FlushFileBuffers(), which requires
  // write access. The debug CRT asserts and the release CRT returns EBADF,
  // which the fail-closed branch below would turn into a failed flush on a
  // file that was never dirty.
  if (file->posix_fd >= 0 && H5FD_CLIO_WRITABLE(file) &&
      clio_vfd_fsync(file->posix_fd) < 0) {
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
        clio_vfd_ftruncate(file->posix_fd, (clio_vfd_off_t)file->eoa) < 0) {
      H5FD_CLIO_ERROR("ftruncate() of authoritative native file failed");
      return FAIL;
    }
    // Keep the CTE cache's logical size in step (best-effort; populate-only
    // tier, see the write callback). Counted on failure for the same reason:
    // a tier that did not shrink still holds bytes past the new EOF.
#if H5FD_CLIO_HAVE_CACHE_TIER
    if (H5FD__clio_cache_live(file->fd)) {
      if (CLIO_CFS_CLIENT->FtruncateFd(file->fd, (off_t)file->eoa) < 0) {
        H5FDclio_cache_truncate_failures_g++;
        HLOG(kWarning,
             "CTE cache truncate to {} failed (native file is unaffected and "
             "remains authoritative)",
             (unsigned long long)file->eoa);
      }
    }
#endif
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
  if (clio_vfd_lock(file->posix_fd, rw ? 1 : 0) < 0) {
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
  if (clio_vfd_unlock(file->posix_fd) < 0) {
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
  /* Start from the defaults so EVERY field is initialized, including the ones
     this entry point does not expose. H5Pset_driver copies the struct verbatim
     onto the FAPL, so a field left unset here would reach H5FD__clio_open as
     stack garbage -- for sieve_max, as the coalescing window and therefore as
     the size of a scratch allocation. Assigning the default struct rather than
     naming fields keeps that true for whatever is added next. */
  H5FD_clio_fapl_t fa = H5FD_clio_fapl_default_g;
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
#if H5FD_CLIO_HAVE_CACHE_TIER
    CLIO_CFS_CLIENT->RemovePath(name);
#endif
  }

  // Remove the authoritative native file. The name is a plain path -- the
  // marked form is refused at open() -- so no stripping is needed. Fail-closed
  // on error (sec2 parity) so a failed delete is reported, not masked.
  // clio_vfd_unlink, not unlink(2): MSVC has no POSIX unlink.
  if (clio_vfd_unlink(name) < 0) {
    H5FD_CLIO_ERROR("unlink() of authoritative native file failed");
    return FAIL;
  }
  return SUCCEED;
} /* end H5FD__clio_del() */

/*
 * Entry points for dynamic plugin loading.
 */
/* H5PLUGIN_DLL, not a bare definition: H5PLextern.h declares both entry points
 * with it, which is __declspec(dllexport) on Windows, so defining them without
 * it is a linkage mismatch (MSVC C2375) -- and an unexported entry point is one
 * HDF5's plugin loader cannot find. Elsewhere it expands to default visibility,
 * which is what a plugin entry point wants in any case. */
H5PLUGIN_DLL H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_VFD; }

H5PLUGIN_DLL const void *H5PLget_plugin_info(void) {
  /* The plugin path does NOT go through H5FD_clio_init(): HDF5 asks for the
     class struct here and registers the driver itself. So this is the entry
     point that must arm the process-exit guard -- arming it only in
     H5FD_clio_init() left the guard disarmed for every HDF5_DRIVER=clio_vfd
     application, which is all of them. */
  H5FD__clio_install_exit_guard();
  return &H5FD_clio_g;
}

} // extern C
