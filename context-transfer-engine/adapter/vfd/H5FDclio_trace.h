/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Byte-altitude access telemetry for the CLIO VFD.
 *
 * Enabled by CLIO_VFD_TRACE=<dir>, deliberately the same name-shape as the
 * VOL's CLIO_VOL_TRACE so the two connectors do not need different muscle
 * memory. Compile it out entirely with -DCLIO_VFD_TRACE_DISABLED.
 *
 * Observe-only: it never changes what is read or written, never changes a
 * return code, and when the variable is unset the cost is one cached bool.
 *
 * ---------------------------------------------------------------------------
 * Why these fields and no others
 * ---------------------------------------------------------------------------
 *
 * Telemetry sprawls unless something constrains it. The constraint here is the
 * consumer: `hdf5 diagnose` is the only thing that reads this, so a field earns
 * its place by backing a recommendation the byte layer can honestly make.
 * Six of them:
 *
 *   R1 "your requests are small -- raise the chunk cache"   <- size histogram
 *   R2 "you re-read the same bytes -- the CLIO tier pays"   <- repeat detection
 *   R3 "metadata traffic dominates"                         <- H5FD_mem_t split
 *   R4 "write-heavy; a populate-only cache will not help"   <- read/write mix
 *   R5 "access is scattered, not sequential"                <- offset locality
 *   R6 "transfers exceed one pass"                          <- multi-pass count
 *
 * R3 is the interesting one: it is a signal the VOL structurally CANNOT
 * produce. By the time a VOL sees H5Dread, the metadata traffic HDF5 generates
 * around it has already happened below. So the byte altitude is not a poorer
 * view of the same thing -- it sees something else.
 *
 * Anything not on that list is out, however easy it would be to emit. A field
 * with no consumer is a field that will be wrong for years without anyone
 * noticing, because nothing reads it.
 *
 * What is deliberately NOT here: any attempt to guess which dataset an access
 * belongs to. Byte offsets carry no names. Guessing would produce confidently
 * wrong advice, which is worse than declining to advise.
 *
 * ---------------------------------------------------------------------------
 * Measured overhead of the ENABLED path
 * ---------------------------------------------------------------------------
 *
 * Q1.5 requires this be measured and published rather than assumed, on the
 * grounds that an unmeasured tracing cost makes every later performance number
 * unfalsifiable. So:
 *
 *   Workload: 40 x (create + write + reopen + read) of a 256 KiB dataset,
 *             360 traced accesses per run, HDF5 2.1.1, cache tier off,
 *             in the iowarp dev container on overlayfs.
 *   trace OFF: 181.5 172.1 174.3 171.7 155.6 ms   (median 172.1)
 *   trace ON : 178.1 194.2 189.8 179.9 172.0 ms   (median 179.9)
 *   => roughly +5% median, +7% mean.
 *
 * Read that as an indication, not a figure: the run-to-run spread (155-182 ms
 * with tracing OFF) is comparable to the effect being measured, so this
 * container is too noisy to claim a precise number. It is also close to a WORST
 * case -- the workload is dominated by many small metadata accesses, which is
 * the highest per-byte tracing cost there is; a workload moving large raw
 * transfers amortizes the per-access work far better. A quieter host and more
 * iterations would be needed before quoting this anywhere load-bearing.
 *
 * The DISABLED path is a different question and much cheaper: one cached bool
 * (TraceDir() is a function-local static) plus a null check per access.
 */
#ifndef CLIO_CTE_ADAPTER_VFD_H5FDCLIO_TRACE_H_
#define CLIO_CTE_ADAPTER_VFD_H5FDCLIO_TRACE_H_

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>

#include "adapter/clio_trace_schema.h"

namespace clio {
namespace vfdtrace {

enum class Op { kRead, kWrite };

/** Bucket boundaries for the request-size histogram (R1). Powers of two from
 *  1 KiB; the interesting question is "are these small?", which needs
 *  resolution at the small end and almost none at the large. */
inline const char *SizeBucket(uint64_t bytes) {
  if (bytes < 1024) return "lt_1k";
  if (bytes < 4096) return "1k_4k";
  if (bytes < 65536) return "4k_64k";
  if (bytes < 1048576) return "64k_1m";
  return "ge_1m";
}

/** HDF5's memory-type tag, collapsed to the distinction that drives R3. HDF5
 *  has several metadata classes; a recommendation only cares raw vs not. */
inline const char *MemClass(int h5fd_mem_type) {
  /* H5FD_MEM_DRAW == 1 in HDF5's enum; everything else is metadata of some
     kind. Compared numerically so this header does not need H5FDpublic.h. */
  return h5fd_mem_type == 1 ? "raw" : "meta";
}

struct FileTrace {
  std::string file_name;
  std::ofstream jsonl;
  uint64_t reads = 0, writes = 0;
  uint64_t read_bytes = 0, write_bytes = 0;
  uint64_t meta_ops = 0, raw_ops = 0;
  uint64_t meta_bytes = 0, raw_bytes = 0;
  uint64_t min_req = UINT64_MAX, max_req = 0;
  uint64_t sequential = 0, scattered = 0;   /* R5 */
  uint64_t next_expected = 0;               /* running sequential cursor */
  uint64_t repeated = 0;                    /* R2 */
  std::map<std::string, uint64_t> size_hist;             /* R1 */
  std::map<uint64_t, uint32_t> range_seen;               /* R2, keyed addr^len */
};

inline const std::string &TraceDir() {
  static const std::string dir = []() -> std::string {
    const char *v = std::getenv("CLIO_VFD_TRACE");
    return (v && *v) ? std::string(v) : std::string();
  }();
  return dir;
}

inline bool Enabled() {
#ifdef CLIO_VFD_TRACE_DISABLED
  return false;
#else
  return !TraceDir().empty();
#endif
}

inline long TracePid() { return static_cast<long>(::getpid()); }

/** Filenames must not collide across processes (MPI ranks) or contain path
 *  separators from the traced file's own path. */
inline std::string SafeBase(const std::string &path) {
  std::string b = path;
  for (char &c : b) {
    if (c == '/' || c == '\\' || c == ':') c = '_';
  }
  return b;
}

inline FileTrace *OpenFile(const std::string &file_name) {
  if (!Enabled()) return nullptr;
  auto *ft = new FileTrace;
  ft->file_name = SafeBase(file_name) + "." + std::to_string(TracePid()) +
                  ".s" + std::to_string(clio::trace::NextSessionSeq());
  ft->jsonl.open(TraceDir() + "/" + ft->file_name + ".access.jsonl",
                 std::ios::out | std::ios::trunc);
  return ft;
}

/** Record one access. `addr`/`size` are the byte region; `mem_type` is HDF5's
 *  H5FD_mem_t as an int; `dur_us` is the measured duration. */
inline void Record(FileTrace *ft, Op op, int mem_type, uint64_t addr,
                   uint64_t size, uint64_t dur_us) {
  if (!ft) return;
  const bool is_raw = (MemClass(mem_type) == std::string("raw"));

  if (op == Op::kRead) { ft->reads++; ft->read_bytes += size; }
  else                 { ft->writes++; ft->write_bytes += size; }
  if (is_raw) { ft->raw_ops++; ft->raw_bytes += size; }
  else        { ft->meta_ops++; ft->meta_bytes += size; }

  if (size < ft->min_req) ft->min_req = size;
  if (size > ft->max_req) ft->max_req = size;
  ft->size_hist[SizeBucket(size)]++;

  /* R5: sequential means "starts where the last one ended". Anything else is
     scattered. Crude on purpose -- the recommendation only needs the ratio. */
  if (addr == ft->next_expected) ft->sequential++;
  else ft->scattered++;
  ft->next_expected = addr + size;

  /* R2: has this exact region been touched before? A cheap key rather than a
     set of intervals -- the question is "is there re-reading at all", and an
     exact-repeat count answers it without interval bookkeeping the driver
     would otherwise have to maintain (that is the residency problem, and it
     does not belong in a telemetry path). */
  const uint64_t key = (addr << 16) ^ size;
  if (++ft->range_seen[key] > 1) ft->repeated++;

  if (ft->jsonl.is_open()) {
    ft->jsonl << "{" << clio::trace::EnvelopeJson(clio::trace::kAltitudeByte)
              << ",\"producer\":\"clio_vfd\""
              << ",\"pid\":" << TracePid()
              << ",\"op\":\"" << (op == Op::kRead ? "read" : "write") << "\""
              << ",\"mem_class\":\"" << MemClass(mem_type) << "\""
              << ",\"addr\":" << addr
              << ",\"bytes\":" << size
              << ",\"dur_us\":" << dur_us
              << "}\n";
    /* NOTE: no "dataset" key, by construction. See clio_trace_schema.h --
       absent must mean absent, never null or "". */
  }
}

inline void CloseFile(FileTrace *ft) {
  if (!ft) return;
  if (ft->jsonl.is_open()) ft->jsonl.close();

  const uint64_t ops = ft->reads + ft->writes;
  std::ofstream o(TraceDir() + "/" + ft->file_name + ".access.json",
                  std::ios::out | std::ios::trunc);
  if (o.is_open()) {
    o << "{" << clio::trace::EnvelopeJson(clio::trace::kAltitudeByte)
      << ",\"producer\":\"clio_vfd\""
      << ",\"pid\":" << TracePid()
      << "," << clio::trace::CapabilitiesJson(clio::trace::kAltitudeByte)
      << ",\"totals\":{"
      << "\"reads\":" << ft->reads << ",\"writes\":" << ft->writes
      << ",\"read_bytes\":" << ft->read_bytes
      << ",\"write_bytes\":" << ft->write_bytes
      << ",\"ops\":" << ops << "}"
      /* R4 */
      << ",\"rw_mix\":{\"read_frac\":"
      << (ops ? static_cast<double>(ft->reads) / static_cast<double>(ops) : 0.0)
      << "}"
      /* R3 -- the byte altitude's unique signal */
      << ",\"mem_class\":{\"meta_ops\":" << ft->meta_ops
      << ",\"raw_ops\":" << ft->raw_ops
      << ",\"meta_bytes\":" << ft->meta_bytes
      << ",\"raw_bytes\":" << ft->raw_bytes << "}"
      /* R1 */
      << ",\"request_size\":{\"min\":"
      << (ft->min_req == UINT64_MAX ? 0 : ft->min_req)
      << ",\"max\":" << ft->max_req
      << ",\"mean\":"
      << (ops ? static_cast<double>(ft->read_bytes + ft->write_bytes) /
                    static_cast<double>(ops)
              : 0.0)
      << ",\"histogram\":{";
    bool first = true;
    for (const auto &b : ft->size_hist) {
      if (!first) o << ",";
      first = false;
      o << "\"" << b.first << "\":" << b.second;
    }
    o << "}}"
      /* R5 */
      << ",\"locality\":{\"sequential\":" << ft->sequential
      << ",\"scattered\":" << ft->scattered << "}"
      /* R2 */
      << ",\"repeat\":{\"repeated_ranges\":" << ft->repeated
      << ",\"distinct_ranges\":" << ft->range_seen.size() << "}"
      << "}\n";
  }
  delete ft;
}

}  // namespace vfdtrace
}  // namespace clio

#endif  // CLIO_CTE_ADAPTER_VFD_H5FDCLIO_TRACE_H_
