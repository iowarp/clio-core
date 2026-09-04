/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Coherence-stamp primitives shared by the HDF5 VOL and VFD connectors.
 *
 * Coherence asks whether the tier's copy still corresponds to the authoritative
 * native file -- a different question from residency, which asks only whether
 * the bytes are present. Only an adapter can answer it, because only it knows
 * which POSIX file a cache entry stands for.
 *
 * Shared because two copies would drift, and drift here does not fail loudly:
 * it leaves a tier answering for a file it can no longer vouch for.
 *
 * The layout-version prefix is NOT shared. It guards "the tier's representation
 * changed", and the two connectors store bytes differently, so each prefixes
 * the identity with its own and bumps it when its layout changes. That is the
 * one staleness file identity cannot catch: a tier populated under a different
 * layout reads back hole-zeros while the file is unchanged.
 */
#ifndef CLIO_ADAPTER_COHERENCE_STAMP_H_
#define CLIO_ADAPTER_COHERENCE_STAMP_H_

#include <sys/stat.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace clio::adapter::stamp {

/* Modification time, split into seconds and nanoseconds.
 *
 * `struct stat` disagrees across platforms: glibc names the timespec member
 * st_mtim, Darwin st_mtimespec, and MSVC has no sub-second member at all. The
 * halves stay separate because the stamp embeds them as "<sec>.<nsec>", and
 * that text is compared against stamps already stored in a tier. */
inline void StatMtime(const struct stat &st, long long *sec, long long *nsec) {
#if defined(_WIN32)
  /* No sub-second field exists; see GranularityNs, which widens the ambiguity
     window to match this coarser clock. */
  *sec = static_cast<long long>(st.st_mtime);
  *nsec = 0;
#elif defined(__APPLE__)
  *sec = static_cast<long long>(st.st_mtimespec.tv_sec);
  *nsec = static_cast<long long>(st.st_mtimespec.tv_nsec);
#else
  *sec = static_cast<long long>(st.st_mtim.tv_sec);
  *nsec = static_cast<long long>(st.st_mtim.tv_nsec);
#endif
}

/* Wall-clock now, in nanoseconds. system_clock rather than clock_gettime:
 * MSVC has neither the function nor the macro, and this cannot fail, so
 * callers need no "clock unreadable" arm. */
inline long long RealtimeNowNs() {
  return static_cast<long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/* Width of the window in which mtime cannot discriminate a later write.
 *
 * Filesystem timestamps are coarse -- the kernel stamps an inode on a tick, so
 * two writes inside one tick get identical mtimes. There is no portable way to
 * ask a filesystem for this number, so it is a bound, not a measurement: too
 * large costs cache misses, too small costs correctness. The default covers
 * HZ=1000 and HZ=250, and does NOT cover second-granularity timestamps (some
 * NFS mounts, FAT); raise it there.
 *
 * A property of the filesystem rather than of a connector, so one knob governs
 * both. CLIO_VOL_STAMP_GRANULARITY_NS is still honoured as the documented
 * name. */
inline uint64_t GranularityNs() {
  static const uint64_t g = []() -> uint64_t {
    const char *e = std::getenv("CLIO_STAMP_GRANULARITY_NS");
    if (e == nullptr || *e == '\0') {
      e = std::getenv("CLIO_VOL_STAMP_GRANULARITY_NS");
    }
    if (e != nullptr && *e != '\0') {
      char *end = nullptr;
      unsigned long long v = std::strtoull(e, &end, 10);
      if (end != e && *end == '\0') return static_cast<uint64_t>(v);
    }
#if defined(_WIN32)
    /* Windows' stat() reports mtime in whole seconds (and only to two on a
       FAT-formatted volume), so a 10 ms granule would call a file unambiguous
       whose very next write lands in the same reported second -- exactly the
       case this check exists to refuse. One second is the smallest default that
       still fails closed there. */
    return 1000ull * 1000ull * 1000ull; /* 1 s */
#else
    return 10ull * 1000ull * 1000ull; /* 10 ms */
#endif
  }();
  return g;
}

/* Identity + state of the native file, WITHOUT a layout-version prefix -- the
   caller prepends its own.

   dev/ino catch the file being replaced (a new inode at the same path:
   h5repack, rsync, mv); size and mtime catch modification in place. Empty when
   the file cannot be stat'd, which callers must treat as "cannot vouch for
   this" rather than as a comparable value. */
inline std::string FileIdentity(const char *path) {
  struct stat st;
  if (!path || stat(path, &st) != 0) return std::string();
  long long mtime_sec = 0, mtime_nsec = 0;
  StatMtime(st, &mtime_sec, &mtime_nsec);
  return std::to_string(static_cast<unsigned long long>(st.st_dev)) + ":" +
         std::to_string(static_cast<unsigned long long>(st.st_ino)) + ":" +
         std::to_string(static_cast<unsigned long long>(st.st_size)) + ":" +
         std::to_string(mtime_sec) + "." + std::to_string(mtime_nsec);
}

/* Can this file's mtime still discriminate a LATER modification?
 *
 * For an in-place, same-size edit mtime is the only signal -- dev, ino and size
 * are unchanged by definition. So a file whose mtime is younger than one
 * granule would produce an identical stamp after a write landing right now.
 *
 * True means "cannot tell": the caller must WITHHOLD the stamp so the next open
 * fails closed. */
inline bool Ambiguous(const char *path) {
  struct stat st;
  if (!path || stat(path, &st) != 0) return true;
  long long mtime_sec = 0, mtime_nsec = 0;
  StatMtime(st, &mtime_sec, &mtime_nsec);
  const int64_t now_ns = static_cast<int64_t>(RealtimeNowNs());
  const int64_t mtime_ns = static_cast<int64_t>(mtime_sec) * 1000000000LL +
                           static_cast<int64_t>(mtime_nsec);
  /* A negative age means the mtime is in the future (clock skew, or a network
     filesystem stamping from a different host). Nothing can be concluded from
     it, so it is ambiguous too. */
  const int64_t age_ns = now_ns - mtime_ns;
  return age_ns < 0 || static_cast<uint64_t>(age_ns) < GranularityNs();
}

}  // namespace clio::adapter::stamp

#endif  // CLIO_ADAPTER_COHERENCE_STAMP_H_
