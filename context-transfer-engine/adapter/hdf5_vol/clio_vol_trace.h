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
   collective) so it always goes native. */
enum class Served { kCache, kNative, kUncacheable };

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
  /* writes: mirrored to the tier vs native-only (uncacheable) */
  uint64_t write_cache = 0, write_uncacheable = 0;
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

/* Per-file aggregation, held by clio_file_t and finalized at file close. */
struct FileTrace {
  std::string file_name;   /* basename used for output filenames */
  std::mutex mtx;
  std::unordered_map<std::string, DsetStat> dsets;
  std::ofstream jsonl;     /* per-access stream (open only when enabled) */
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
    if (a.served == Served::kCache) s.write_cache++;
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
    ft->jsonl << "{\"op\":\"" << op_str(a.op) << "\",\"dataset\":\"" << a.dataset
              << "\",\"dtype\":\"" << a.dtype << "\",\"elem_size\":" << a.elem_size
              << ",\"ndims\":" << a.ndims << ",\"sel\":\"" << sel_str(a.sel)
              << "\",\"sel_sig\":" << a.sel_sig << ",\"nelem\":" << a.nelem_sel
              << ",\"bytes\":" << a.bytes << ",\"served\":\"" << served_str(a.served)
              << "\",\"chunked\":" << (a.chunked ? "true" : "false")
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
  /* Suffix with the pid so concurrent processes (e.g. MPI ranks) opening the same
     file path do not clobber each other's trace output. */
  ft->file_name = safe_base(file_name) + "." + std::to_string(trace_pid());
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
    o << "{\n  \"file\": \"" << ft->file_name << "\",\n  \"datasets\": {\n";
    bool first = true;
    for (auto &kv : ft->dsets) {
      const DsetStat &s = kv.second;
      g_reads += s.reads; g_writes += s.writes;
      g_rc += s.read_cache; g_rn += s.read_native; g_ru += s.read_uncacheable;
      g_br += s.bytes_read; g_bw += s.bytes_written;
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
        << ", \"read_bytes_from_native\": " << s.read_bytes_native
        << ", \"write_served\": {\"mirrored\": " << s.write_cache
        << ", \"native_only\": " << s.write_uncacheable << "}"
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
      << "}\n}\n";
  }
  delete ft;
}

}  // namespace trace
}  // namespace clio

#endif  // CLIO_HDF5_VOL_TRACE_H_
