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
#include <thread>
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

/* How long a close may wait for the mtime to settle. DEFAULTS TO 0 -- no wait,
 * which makes SettleForStamp behave exactly like Ambiguous() and preserves the
 * fail-closed rule as it stands.
 *
 * Waiting is opt-in because it does NOT close the window it appears to. The
 * stamp is written after the under-VOL has closed the file (its size and mtime
 * are not final until then), so the file is UNLOCKED for the whole wait: a
 * writer that modifies it in the same timestamp tick as our last write leaves
 * mtime identical, the re-stat sees nothing, and we would stamp a file whose
 * contents no longer match what was staged -- the stale-cache bug the stamp
 * exists to prevent, and the one vol/instant_reopen pins.
 *
 * Timing cannot fix that: mtime at tick granularity cannot distinguish two
 * writes in one tick, and a content digest would have to READ the whole file at
 * open, which is the read the cache exists to avoid. Settling under the file
 * lock (stamping before the native close) would make it sound, but the mtime is
 * not final there.
 *
 * So this is a knob for a deployment that has decided the race is acceptable --
 * a single-writer pipeline, say -- in exchange for write-side staging that
 * survives to the next open. Set it to about two granules to enable.
 *
 * A property of the filesystem, like GranularityNs, so one knob governs both
 * connectors. */
inline uint64_t SettleMaxWaitNs() {
  static const uint64_t v = []() -> uint64_t {
    const char *e = std::getenv("CLIO_STAMP_SETTLE_MAX_NS");
    if (e != nullptr && *e != '\0') {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && *end == '\0') return static_cast<uint64_t>(n);
    }
    return 0;
  }();
  return v;
}

/* Wait until this file's mtime CAN discriminate a later write, then report
 * whether it does.
 *
 * Ambiguous() answers "can mtime tell?" and the connectors' close paths used to
 * withhold the stamp whenever it said no. But at a writing close the answer is
 * ALWAYS no -- the session just wrote the file, so its mtime is fresh by
 * construction -- which meant a writer never stamped, the next open found no
 * stamp, failed closed, and dropped the very image the write had just staged.
 * The staging was therefore pure cost in the commonest pattern there is: write
 * a file, then read it back.
 *
 * Waiting fixes it at the source. Sleep only until the file's mtime is one
 * granule old, then re-stat. Once `now > mtime + granularity`, any subsequent
 * write must land on a later tick and so must change mtime -- which is exactly
 * the property a stamp needs. The wait is bounded by one granule (10 ms by
 * default, and typically less, since some of it has already elapsed), paid once
 * per close that has something staged to vouch for.
 *
 * Returns false if the file cannot be stat'd, or if it changed underneath us
 * while we waited -- in both cases the caller must withhold the stamp exactly
 * as before. A change during the wait means another writer is active, which is
 * the multi-process case and stays undefined.
 *
 * `max_wait_ns` bounds the sleep so a pathological granularity setting cannot
 * hang a close; 0 means "do not wait", restoring the old withhold-always
 * behaviour. */
inline bool SettleForStamp(const char *path, uint64_t max_wait_ns) {
  struct stat st;
  if (!path || stat(path, &st) != 0) return false;
  long long sec = 0, nsec = 0;
  StatMtime(st, &sec, &nsec);
  const int64_t mtime_ns =
      static_cast<int64_t>(sec) * 1000000000LL + static_cast<int64_t>(nsec);
  const int64_t ready_ns =
      mtime_ns + static_cast<int64_t>(GranularityNs()) + 1;
  const int64_t now_ns = static_cast<int64_t>(RealtimeNowNs());
  if (now_ns < ready_ns) {
    int64_t wait_ns = ready_ns - now_ns;
    /* An mtime in the future (clock skew, or a network filesystem stamping
       from another host) would ask for an unbounded sleep. Refuse rather than
       wait: nothing can be concluded from such a timestamp anyway. */
    if (wait_ns > static_cast<int64_t>(max_wait_ns)) return false;
    std::this_thread::sleep_for(std::chrono::nanoseconds(wait_ns));
  }
  /* Re-stat: the file must not have moved while we waited, or the stamp we are
     about to take describes a state that is already gone. */
  struct stat after;
  if (stat(path, &after) != 0) return false;
  long long sec2 = 0, nsec2 = 0;
  StatMtime(after, &sec2, &nsec2);
  if (sec2 != sec || nsec2 != nsec || after.st_size != st.st_size ||
      after.st_ino != st.st_ino || after.st_dev != st.st_dev) {
    return false;
  }
  return !Ambiguous(path);
}

}  // namespace clio::adapter::stamp

#endif  // CLIO_ADAPTER_COHERENCE_STAMP_H_
