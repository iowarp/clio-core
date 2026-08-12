/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Clio HDF5 VOL — access telemetry (observability, Part B).
 *
 * Observe-only, header-only. Records per-access HDF5-semantic information that
 * only the VOL sees (dataset path, datatype, selection shape, transfer size, and
 * — the CLIO-specific signal — whether the transfer was served from the CTE tier
 * or from native). Two outputs, matching the report's WRP-trace -> OMNI-summary
 * model:
 *   - per-access JSONL:   <dir>/<file>.access.jsonl   (one line per read/write)
 *   - aggregated summary: <dir>/<file>.access.json    (written at file close)
 * The summary is the artifact a CLIO-using agent reads to advise tuning:
 * hot datasets, cache hit rate, repeated selections (cache/prefetch candidates),
 * and the transfer-size distribution (small-read / per-object-cost detection).
 *
 * Gated entirely by the CLIO_VOL_TRACE environment variable (a directory). When
 * unset, enabled() is a single cached bool check and nothing else runs — zero
 * effect on data path or performance. Telemetry NEVER alters semantics: records
 * are taken after each operation completes.
 */
#ifndef CLIO_HDF5_VOL_TRACE_H_
#define CLIO_HDF5_VOL_TRACE_H_

#include "adapter/clio_trace_schema.h"
#include <hdf5.h>

#ifdef _WIN32
#include <process.h>  /* _getpid — per-process trace filenames */
#else
#include <unistd.h>  /* getpid — per-process trace filenames */
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace clio {
namespace trace {

/* getpid() is POSIX; MSVC spells it _getpid() (<process.h>). Portable shim for
   the per-process trace-file suffix in open_file() below. */
inline long trace_pid() {
#ifdef _WIN32
  return static_cast<long>(_getpid());
#else
  return static_cast<long>(getpid());
#endif
}

enum class Op { kRead, kWrite };
enum class Sel { kWhole, kHyperslab, kPoint, kOther };
/* How a transfer was satisfied. kCache = served from the CTE tier; kNative =
   delegated to the native VOL (cacheable but a miss, or a selection miss);
   kUncacheable = the datatype/transfer is never cached (compound/array/vlen/
   collective) so it always goes native.

   kStaged is a WRITE-only state and exists because the honest answer at write
   time is neither "cached" nor "not cached". Staging puts are asynchronous: at
   the moment the write record is taken they have been submitted and nothing has
   confirmed they landed. This state says exactly that -- submitted, outcome
   resolved later at drain.

   It replaces reporting writes as kCache, which was decided from whether the
   NATIVE write succeeded. That made "mirrored" mean "the native write was fine",
   a fact about the authoritative file rather than about the tier, so any
   admission measurement built on it was measuring the wrong thing. A write is
   never kCache: a cache serves reads, and nothing is served by writing. */
enum class Served { kCache, kNative, kUncacheable, kStaged };

/* One read/write access. */
struct Access {
  Op op;
  Sel sel;
  Served served;
  std::string dataset;
  std::string dtype;       /* datatype class name */
  size_t elem_size = 0;
  int ndims = 0;
  long long nelem_sel = 0;
  size_t bytes = 0;
  size_t staged_bytes = 0; /* writes: bytes actually submitted to the tier */
  double dur_us = 0.0;
  uint64_t sel_sig = 0;    /* selection-bounds signature for repeat detection */
  bool chunked = false;    /* dataset storage layout is chunked */
  int chunk_aligned = -1;  /* -1 = n/a (contiguous/point), 0 = misaligned, 1 = aligned */
  std::string chunk_dims;  /* e.g. "[4,3]" for chunked datasets */
};

/* Per-dataset rollup. Read serving and write mirroring are tracked separately so
   the cache hit rate is a READ-only signal (a write mirrored to the tier is not a
   "hit"). */
struct DsetStat {
  uint64_t reads = 0, writes = 0;
  uint64_t bytes_read = 0, bytes_written = 0;
  uint64_t whole = 0, hyperslab = 0, point = 0, other = 0;
  /* reads: how each was served */
  uint64_t read_cache = 0, read_native = 0, read_uncacheable = 0;
  uint64_t read_bytes_cache = 0, read_bytes_native = 0;
  /* read latency split by how it was served (tier vs native) */
  double read_cache_us = 0, read_native_us = 0;
  /* writes: submitted to the tier vs native-only (uncacheable).
     `write_staged`/`bytes_staged` count what was SUBMITTED; `staged_discarded`
     counts bytes whose staging was later thrown away, either because a put
     failed at drain or because something invalidated the image (a partial
     write, a set_extent, a failed native write).

     WHAT THESE DO AND DO NOT SAY. bytes_staged is the cost of admission --
     bytes written a second time, into the tier -- and it is the honest
     replacement for using application write volume, which counts writes that
     were never eligible for the tier at all. bytes_staged minus discarded is
     what is still SERVABLE at close, and that is NOT a measure of usefulness:
     staged data can be served and then invalidated by a later partial write,
     so a dataset can show resident=0 having answered reads all session. The
     benefit side is read_bytes_from_cache. An admission ratio needs both, and
     needs a workload whose sessions the measurer controls -- which is why no
     ratio is emitted here. */
  uint64_t write_staged = 0, write_uncacheable = 0;
  uint64_t bytes_staged = 0, staged_discarded = 0;
  /* Bytes pulled OUT of the tier to satisfy reads, as opposed to bytes handed
     to the application (read_bytes_from_cache). The gap between them is read
     amplification: the cache stores a linear image, so serving a selection
     fetches whole chunks around it. §4(A) narrows the fetch to the selection's
     bounding box; this is the number that shows whether it worked, and the one
     a rechunking recommendation would rest on. */
  uint64_t bytes_fetched = 0;
  size_t min_bytes = SIZE_MAX, max_bytes = 0;
  double sum_bytes = 0;
  uint64_t n_sized = 0;
  double read_us = 0, write_us = 0;
  std::string dtype;
  size_t elem_size = 0;
  int ndims = 0;
  /* storage layout + chunk-alignment of reads (rechunk / read-amplification signal) */
  bool chunked = false;
  std::string chunk_dims;
  uint64_t read_aligned = 0, read_misaligned = 0;
  std::unordered_map<uint64_t, uint32_t> sel_counts;  /* signature -> count */
};

/* What the coherence stamp said, at file altitude. Emitted because every one
   of these outcomes except kMatched ends in the same place -- reads that count
   as `native` -- and a hit rate alone cannot say WHY the tier went unused. A
   cold file, an evicted stamp, a file genuinely modified between sessions and
   a file we simply declined to trust are four different findings, and a
   measurement that cannot separate them will attribute a self-inflicted miss
   to the workload.

   kAmbiguous is the one to watch in a benchmark. It means the file was NOT
   known to have changed: its mtime was too recent for mtime to rule out a
   later same-granule write, so the stamp was withheld and the next open pays a
   miss on a file that is probably fine. That is a deliberate trade of hit rate
   for correctness (see clio_stamp_ambiguous in clio_vol.cc), and if it is ever
   a material fraction of opens, the answer is to widen the workload's
   close-to-reopen gap or revisit the granularity bound -- not to wonder why
   the cache "randomly" underperforms. */
enum class Stamp { kMatched, kMismatched, kAbsent, kAmbiguous };

/* Per-file aggregation, held by clio_file_t and finalized at file close. */
struct FileTrace {
  std::string file_name;   /* basename used for output filenames */
  std::mutex mtx;
  std::unordered_map<std::string, DsetStat> dsets;
  std::ofstream jsonl;     /* per-access stream (open only when enabled) */
  /* Coherence-stamp outcomes for this session. Counters rather than a single
     value because one session checks at open and writes at close, and a future
     re-validation would add more; a scalar would silently keep only the last. */
  uint64_t stamp_matched = 0, stamp_mismatched = 0;
  uint64_t stamp_absent = 0, stamp_ambiguous = 0;
};

/* ---- internals ---- */

inline const std::string &trace_dir() {
  static std::string dir = []() {
    const char *e = std::getenv("CLIO_VOL_TRACE");
    return std::string(e ? e : "");
  }();
  return dir;
}

/* True when telemetry is enabled (CLIO_VOL_TRACE set). Cached; cheap. */
inline bool enabled() { return !trace_dir().empty(); }

inline const char *op_str(Op o) { return o == Op::kRead ? "read" : "write"; }
inline const char *sel_str(Sel s) {
  switch (s) {
    case Sel::kWhole: return "whole";
    case Sel::kHyperslab: return "hyperslab";
    case Sel::kPoint: return "point";
    default: return "other";
  }
}
inline const char *served_str(Served s) {
  switch (s) {
    case Served::kCache: return "cache";
    case Served::kNative: return "native";
    case Served::kStaged: return "staged";
    default: return "uncacheable";
  }
}

/* Classify a file-space selection into a shape and a bounds-based signature.
   H5S_ALL (or an all-points selection) is "whole". The signature hashes the
   selection type, rank, and bounding box so repeated identical selections
   collide (the caching/prefetch candidate signal). */
inline Sel classify(hid_t file_space_id, uint64_t *sig) {
  *sig = 0;
  if (file_space_id == H5S_ALL) { *sig = 1469598103934665603ull; return Sel::kWhole; }
  H5S_sel_type st = H5Sget_select_type(file_space_id);
  int rank = H5Sget_simple_extent_ndims(file_space_id);
  uint64_t h = 1469598103934665603ull;  /* FNV-1a */
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  mix(static_cast<uint64_t>(st));
  mix(static_cast<uint64_t>(rank));
  Sel sel = Sel::kOther;
  if (st == H5S_SEL_ALL) {
    sel = Sel::kWhole;
  } else if (st == H5S_SEL_HYPERSLABS || st == H5S_SEL_POINTS) {
    sel = (st == H5S_SEL_POINTS) ? Sel::kPoint : Sel::kHyperslab;
    if (rank > 0 && rank <= 32) {
      hsize_t start[32], end[32];
      if (H5Sget_select_bounds(file_space_id, start, end) >= 0) {
        for (int i = 0; i < rank; ++i) { mix(start[i]); mix(end[i]); }
      }
    }
    hssize_t np = H5Sget_select_npoints(file_space_id);
    if (np > 0) mix(static_cast<uint64_t>(np));
  }
  *sig = h;
  return sel;
}

/* Aggregate one access and, if a JSONL stream is open, append a record. Caller
   holds nothing; this locks the file's mutex. */
/* Report bytes submitted to the tier for `dataset`.
   THE single accounting point for admission cost, called from every path that
   stages -- the write path and the read-miss path both. Staging is its own
   event: it is not an attribute of the read or write that happened to trigger
   it, and treating it as one is why read-miss staging went uncounted. That
   mattered the moment an admission policy existed, because under
   CLIO_VOL_ADMIT=read-miss every staged byte arrives on the read path, and a
   measurement that could not see them would have scored the policy as staging
   nothing at all. */
/* Bytes read from the tier for `dataset`. Called by the image fetch, which is
   the only place that talks to the tier on the read path. */
/* Record one coherence-stamp outcome. Safe with a null FileTrace (telemetry
   off), like every other record_* here. */
inline void record_stamp(FileTrace *ft, Stamp s) {
  if (!ft) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  switch (s) {
    case Stamp::kMatched: ft->stamp_matched++; break;
    case Stamp::kMismatched: ft->stamp_mismatched++; break;
    case Stamp::kAbsent: ft->stamp_absent++; break;
    case Stamp::kAmbiguous: ft->stamp_ambiguous++; break;
  }
}

inline void record_fetch(FileTrace *ft, const std::string &dataset,
                         uint64_t bytes) {
  if (!ft || bytes == 0) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  ft->dsets[dataset].bytes_fetched += bytes;
}

inline void record_stage(FileTrace *ft, const std::string &dataset,
                         uint64_t bytes) {
  if (!ft || bytes == 0) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  ft->dsets[dataset].bytes_staged += bytes;
}

/* Report that everything staged for `dataset` has stopped being servable.
   Idempotent by construction -- discarded is clamped to what was staged, so a
   dataset invalidated twice does not report negative surviving bytes. */
inline void record_discard(FileTrace *ft, const std::string &dataset) {
  if (!ft) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  auto it = ft->dsets.find(dataset);
  if (it == ft->dsets.end()) return;
  it->second.staged_discarded = it->second.bytes_staged;
}

inline void record(FileTrace *ft, const Access &a) {
  if (!ft) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  DsetStat &s = ft->dsets[a.dataset];
  if (s.dtype.empty()) { s.dtype = a.dtype; s.elem_size = a.elem_size; }
  if (a.ndims > s.ndims) s.ndims = a.ndims;  /* whole ops report 0; keep true rank */
  if (a.chunked) { s.chunked = true; if (s.chunk_dims.empty()) s.chunk_dims = a.chunk_dims; }
  if (a.op == Op::kRead) {
    s.reads++; s.bytes_read += a.bytes; s.read_us += a.dur_us;
    if (a.chunk_aligned == 1) s.read_aligned++;
    else if (a.chunk_aligned == 0) s.read_misaligned++;
    switch (a.served) {
      case Served::kCache:
        s.read_cache++; s.read_bytes_cache += a.bytes; s.read_cache_us += a.dur_us; break;
      case Served::kNative:
        s.read_native++; s.read_bytes_native += a.bytes; s.read_native_us += a.dur_us; break;
      default:
        s.read_uncacheable++; s.read_bytes_native += a.bytes; s.read_native_us += a.dur_us; break;
    }
  } else {
    s.writes++; s.bytes_written += a.bytes; s.write_us += a.dur_us;
    /* Counters only. The BYTES are accounted by record_stage, which every
       staging path calls; adding a.staged_bytes here too would double-count
       the write path and still miss the read-miss path. */
    if (a.served == Served::kStaged) s.write_staged++;
    else s.write_uncacheable++;
  }
  switch (a.sel) {
    case Sel::kWhole: s.whole++; break;
    case Sel::kHyperslab: s.hyperslab++; break;
    case Sel::kPoint: s.point++; break;
    default: s.other++; break;
  }
  if (a.bytes > 0) {
    s.min_bytes = a.bytes < s.min_bytes ? a.bytes : s.min_bytes;
    s.max_bytes = a.bytes > s.max_bytes ? a.bytes : s.max_bytes;
    s.sum_bytes += static_cast<double>(a.bytes);
    s.n_sized++;
  }
  /* Repeat detection applies to reads (the caching/prefetch signal). */
  if (a.op == Op::kRead) s.sel_counts[a.sel_sig]++;

  if (ft->jsonl.is_open()) {
    ft->jsonl << "{" << clio::trace::EnvelopeJson(clio::trace::kAltitudeSemantic)
              << ",\"op\":\"" << op_str(a.op) << "\",\"dataset\":\"" << a.dataset
              << "\",\"dtype\":\"" << a.dtype << "\",\"elem_size\":" << a.elem_size
              << ",\"ndims\":" << a.ndims << ",\"sel\":\"" << sel_str(a.sel)
              << "\",\"sel_sig\":" << a.sel_sig << ",\"nelem\":" << a.nelem_sel
              << ",\"bytes\":" << a.bytes << ",\"served\":\"" << served_str(a.served)
              << "\"";
    /* Absent means absent: a read has no staged_bytes, so the key is not
       emitted for one rather than emitted as 0. */
    if (a.op == Op::kWrite && a.served == Served::kStaged)
      ft->jsonl << ",\"staged_bytes\":" << a.staged_bytes;
    ft->jsonl << ",\"chunked\":" << (a.chunked ? "true" : "false")
              << ",\"chunk_aligned\":" << a.chunk_aligned
              << ",\"dur_us\":" << a.dur_us << "}\n";
  }
}

/* Filesystem-safe basename of an HDF5 file path (used for output filenames). */
inline std::string safe_base(const std::string &path) {
  size_t slash = path.find_last_of('/');
  std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
  for (char &c : base) if (c == ' ' || c == ':') c = '_';
  return base.empty() ? std::string("file") : base;
}

/* Open the per-file streams. Call once when the file is created/opened (only when
   enabled()). */
inline FileTrace *open_file(const std::string &file_name) {
  if (!enabled()) return nullptr;
  auto *ft = new FileTrace();
  /* Suffix with the pid so concurrent processes (e.g. MPI ranks) opening the
     same file path do not clobber each other's trace output -- and with a
     session number, because the pid alone is NOT unique across successive opens
     within one process. "create, write, close; open, read, close" is the most
     ordinary shape there is, and without the session suffix the read session
     truncated the write session's artifacts: the summary then reported zero
     writes for a workload that wrote the whole file. */
  ft->file_name = safe_base(file_name) + "." + std::to_string(trace_pid()) +
                  ".s" + std::to_string(clio::trace::NextSessionSeq());
  ft->jsonl.open(trace_dir() + "/" + ft->file_name + ".access.jsonl",
                 std::ios::out | std::ios::trunc);
  return ft;
}

/* Write the aggregated summary JSON and free the FileTrace. Call at file close. */
inline void close_file(FileTrace *ft) {
  if (!ft) return;
  std::lock_guard<std::mutex> lk(ft->mtx);
  if (ft->jsonl.is_open()) ft->jsonl.close();

  std::ofstream o(trace_dir() + "/" + ft->file_name + ".access.json",
                  std::ios::out | std::ios::trunc);
  if (o.is_open()) {
    uint64_t g_reads = 0, g_writes = 0, g_rc = 0, g_rn = 0, g_ru = 0;
    uint64_t g_br = 0, g_bw = 0, g_rbc = 0, g_rbn = 0;
    uint64_t g_bs = 0, g_bd = 0;
    /* Same envelope the VFD emits, so the two producers are one contract at
       different altitudes rather than two formats. `capabilities` states what
       this producer can and cannot see, machine-readably, so a report can
       declare its own limits without a human remembering to. */
    o << "{\n  " << clio::trace::EnvelopeJson(clio::trace::kAltitudeSemantic)
      << ",\n  \"producer\": \"clio_hdf5_vol\",\n  "
      << clio::trace::CapabilitiesJson(clio::trace::kAltitudeSemantic)
      << ",\n  \"file\": \"" << ft->file_name << "\",\n  \"datasets\": {\n";
    bool first = true;
    for (auto &kv : ft->dsets) {
      const DsetStat &s = kv.second;
      g_reads += s.reads; g_writes += s.writes;
      g_rc += s.read_cache; g_rn += s.read_native; g_ru += s.read_uncacheable;
      g_br += s.bytes_read; g_bw += s.bytes_written;
      g_bs += s.bytes_staged; g_bd += s.staged_discarded;
      g_rbc += s.read_bytes_cache; g_rbn += s.read_bytes_native;
      /* Read hit rate = fraction of reads served from the tier (writes excluded). */
      double hit_rate = s.reads ? (double)s.read_cache / (double)s.reads : 0.0;
      uint32_t max_repeat = 0;
      for (auto &c : s.sel_counts) max_repeat = c.second > max_repeat ? c.second : max_repeat;
      double mean_bytes = s.n_sized ? s.sum_bytes / (double)s.n_sized : 0.0;
      uint64_t read_nc = s.read_native + s.read_uncacheable;  /* non-cache reads */
      double cache_lat = s.read_cache ? s.read_cache_us / (double)s.read_cache : 0.0;
      double native_lat = read_nc ? s.read_native_us / (double)read_nc : 0.0;
      if (!first) o << ",\n";
      first = false;
      o << "    \"" << kv.first << "\": {"
        << "\"dtype\": \"" << s.dtype << "\", \"elem_size\": " << s.elem_size
        << ", \"ndims\": " << s.ndims
        << ", \"reads\": " << s.reads << ", \"writes\": " << s.writes
        << ", \"bytes_read\": " << s.bytes_read << ", \"bytes_written\": " << s.bytes_written
        << ", \"sel\": {\"whole\": " << s.whole << ", \"hyperslab\": " << s.hyperslab
        << ", \"point\": " << s.point << ", \"other\": " << s.other << "}"
        << ", \"read_served\": {\"cache\": " << s.read_cache
        << ", \"native\": " << s.read_native
        << ", \"uncacheable\": " << s.read_uncacheable << "}"
        << ", \"cache_hit_rate\": " << hit_rate
        << ", \"read_bytes_from_cache\": " << s.read_bytes_cache
        << ", \"bytes_fetched_from_tier\": " << s.bytes_fetched
        << ", \"read_bytes_from_native\": " << s.read_bytes_native
        << ", \"write_staged\": {\"staged\": " << s.write_staged
        << ", \"native_only\": " << s.write_uncacheable
        << ", \"bytes_staged\": " << s.bytes_staged
        << ", \"bytes_discarded\": " << s.staged_discarded
        << ", \"bytes_resident\": "
        << (s.bytes_staged > s.staged_discarded
                ? s.bytes_staged - s.staged_discarded : 0) << "}"
        << ", \"layout\": {\"chunked\": " << (s.chunked ? "true" : "false")
        << ", \"chunk_dims\": \"" << s.chunk_dims << "\", \"read_aligned\": "
        << s.read_aligned << ", \"read_misaligned\": " << s.read_misaligned << "}"
        << ", \"read_latency_us\": {\"cache_mean\": " << cache_lat
        << ", \"native_mean\": " << native_lat << "}"
        << ", \"xfer_bytes\": {\"min\": " << (s.n_sized ? s.min_bytes : 0)
        << ", \"max\": " << s.max_bytes << ", \"mean\": " << mean_bytes << "}"
        << ", \"distinct_read_selections\": " << s.sel_counts.size()
        << ", \"max_repeated_selection\": " << max_repeat << "}";
    }
    double g_hit = g_reads ? (double)g_rc / (double)g_reads : 0.0;
    o << "\n  },\n  \"totals\": {"
      << "\"reads\": " << g_reads << ", \"writes\": " << g_writes
      << ", \"bytes_read\": " << g_br << ", \"bytes_written\": " << g_bw
      << ", \"read_served\": {\"cache\": " << g_rc << ", \"native\": " << g_rn
      << ", \"uncacheable\": " << g_ru << "}"
      << ", \"cache_hit_rate\": " << g_hit
      << ", \"read_bytes_from_cache\": " << g_rbc
      << ", \"read_bytes_from_native\": " << g_rbn
      /* The admission inputs, as raw totals. Deliberately NOT emitted as a
         ratio: read_bytes_from_cache can include bytes staged by an EARLIER
         session of the same file, so dividing the two within one summary would
         look like "fraction of what we staged that got read back" while
         actually being something slightly different. The measurement harness
         that controls both sessions can form the ratio honestly; a field here
         could not say which it meant. */
      << ", \"bytes_staged\": " << g_bs
      << ", \"bytes_staged_discarded\": " << g_bd
      << ", \"bytes_staged_resident\": " << (g_bs > g_bd ? g_bs - g_bd : 0)
      << "}"
      /* File altitude, not per-dataset: the stamp describes the FILE, and a
         session checks it once before any dataset is open. See enum Stamp. */
      << ",\n  \"coherence\": {\"matched\": " << ft->stamp_matched
      << ", \"mismatched\": " << ft->stamp_mismatched
      << ", \"absent\": " << ft->stamp_absent
      << ", \"ambiguous\": " << ft->stamp_ambiguous
      << "}\n}\n";
  }
  delete ft;
}

}  // namespace trace
}  // namespace clio

#endif  // CLIO_HDF5_VOL_TRACE_H_
