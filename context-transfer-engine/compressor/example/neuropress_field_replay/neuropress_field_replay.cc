/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_field_replay.cc
 * @brief Replay simulation field dumps through Clio's compressor.
 *
 * The LAMMPS example (../neuropress_lammps_lib) drives a live simulation and
 * hands its arrays to Clio. Some codes cannot be embedded that way -- Nyx is
 * an AMReX application with no library interface, and upstream NeuroPress's
 * own Nyx benchmark does not embed it either: it patches Nyx to dump raw
 * float32 fields per FArrayBox per component, then sweeps those files
 * offline. This driver is the Clio side of that second shape.
 *
 * It reads a directory of flat binary field files, cuts each into chunks, and
 * submits every chunk to the compressor exactly as the LAMMPS driver does --
 * same Context, same AsyncDynamicSchedule, same selection knobs from compose.
 * What changes is only where the bytes come from.
 *
 * Files are replayed in lexical order, which the dumper makes chronological
 * by zero-padding the timestep (plt00000_..., plt00001_...). That matters
 * for the learning and exploration modes, whose whole point is that a
 * decision depends on the chunks that came before it.
 *
 * ELEMENT TYPE. Default is float32 (Context::data_type_ = 1), because that is
 * what the dumps hold and what the selection model was trained on -- and,
 * unlike the float64 LAMMPS workload, it is the width NeuroPress's byte
 * shuffle actually assumes. Use --f64 for float64 dumps.
 *
 * Run through the scripts in paper-benchmark/nyx.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
// DeviceAwareMemcpy and gpu::IpcManager::MemKind, for the baseline's H2D
// staging: --no-compress hands the bdev a DEVICE pointer so it pays the same
// device-to-host copy every compressed arm pays.
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string dir;                 // directory of field files
  std::string ext = ".f32";        // which files to replay
  size_t chunk = 4u << 20;         // bytes per compressor call; 0 = whole file
  std::string tag = "field_replay";
  std::string report;              // per-chunk CSV
  size_t max_files = 0;            // 0 = all
  bool f64 = false;                // dumps are float64 rather than float32
  bool verify = false;
  bool readback = false;           // read a previous report back, no files
  std::string dump_dir;            // write decompressed bytes here (readback)
  // Check |original - decoded| <= error bound instead of a bit-exact digest.
  // Only meaningful with a positive CLIO_NEUROPRESS_ERROR_BOUND: under lossy
  // compression the digest MUST differ, so --verify alone can only report a
  // failure that is not one. This re-reads each chunk from its source file
  // and compares element-wise, which is the only check that actually tests
  // the guarantee the error bound makes.
  bool check_bound = false;
  bool no_compress = false;  // raw PutBlob/GetBlob, no codec (baseline)
  // Read-back requests kept in flight at once. 1 is the sequential read every
  // earlier campaign used; more is a consumer that prefetches, which lets the
  // runtime's workers decode several chunks concurrently.
  size_t read_inflight = 1;
  // Decode into device memory (a GPU consumer's buffer) instead of host
  // shared memory; the runtime then skips the D2H of every decoded chunk.
  bool read_to_gpu = false;
  // With --verify: read everything back this many times, each timed. Needed
  // for a memory-backed tier, whose contents do not outlive this process.
  size_t read_repeat = 1;
  // With --verify: read back only the blobs whose name matches (ECMAScript
  // regex), as a consumer that reads some of the fields a producer wrote.
  std::string read_match;
};

void Usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --dir DIR [options]\n"
      << "  --dir DIR        directory of flat binary field files\n"
      << "  --ext .f32       file extension to replay [.f32]\n"
      << "  --check-bound    with a positive CLIO_NEUROPRESS_ERROR_BOUND,\n"
      << "                   verify |original-decoded| <= bound element-wise\n"
      << "                   against the source files instead of comparing a\n"
      << "                   digest, which lossy compression must fail\n"
      << "  --chunk BYTES    bytes per compressor call, 0 = whole file "
         "[4194304]\n"
      << "  --max-files N    replay only the first N files (lexical order)\n"
      << "  --f64            files hold float64 (default float32)\n"
      << "  --tag NAME       CTE tag [field_replay]\n"
      << "  --report CSV     per-chunk outcome\n"
      << "  --verify         read every blob back and compare\n"
      << "  --no-compress    baseline: store every chunk raw with PutBlob, no\n"
      << "                   codec selection or compression\n"
      << "  --readback CSV   no files: read the blobs a previous run listed\n"
      << "  --read-inflight N  keep N read-back requests in flight [1]\n"
      << "  --read-to-gpu    read back into device memory (a GPU consumer)\n"
      << "  --read-repeat N  with --verify: N timed read-backs; --check-bound\n"
      << "                   then runs one more, untimed, to check the bound\n"
      << "  --read-match RE  with --verify: read back only blobs matching RE\n"
      << "  --dump-decompressed DIR  with --readback, write each decompressed\n"
      << "                   blob to DIR so an EXTERNAL tool can compare it\n"
      << "                   against the simulation's own output files\n";
}

bool ParseArgs(int argc, char **argv, Options *o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << a << " needs " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--dir") o->dir = need("DIR");
    else if (a == "--ext") o->ext = need("EXT");
    else if (a == "--chunk") o->chunk = std::strtoull(need("BYTES"), nullptr, 10);
    else if (a == "--max-files") o->max_files = std::strtoull(need("N"), nullptr, 10);
    else if (a == "--tag") o->tag = need("NAME");
    else if (a == "--report") o->report = need("CSV");
    else if (a == "--readback") { o->readback = true; o->report = need("CSV"); }
    else if (a == "--dump-decompressed") o->dump_dir = need("DIR");
    else if (a == "--check-bound") o->check_bound = true;
    else if (a == "--f64") o->f64 = true;
    else if (a == "--verify") o->verify = true;
    else if (a == "--no-compress") o->no_compress = true;
    else if (a == "--read-to-gpu") o->read_to_gpu = true;
    else if (a == "--read-match") o->read_match = need("REGEX");
    else if (a == "--read-repeat") {
      o->read_repeat = std::strtoull(need("N"), nullptr, 10);
      if (o->read_repeat == 0) o->read_repeat = 1;
    }
    else if (a == "--read-inflight") {
      o->read_inflight = std::strtoull(need("N"), nullptr, 10);
      if (o->read_inflight == 0) o->read_inflight = 1;
    }
    else if (a == "-h" || a == "--help") { Usage(argv[0]); std::exit(0); }
    else { std::cerr << "unknown option " << a << "\n"; Usage(argv[0]); return false; }
  }
  if (o->dir.empty() && !o->readback) { Usage(argv[0]); return false; }
  return true;
}

/* FNV-1a-64, the digest the sibling examples use, so results are comparable. */
uint64_t Fnv1a(const void *p, size_t n) {
  const auto *b = static_cast<const unsigned char *>(p);
  uint64_t h = 14695981039346656037ull;  // 0xcbf29ce484222325
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

/**
 * Absolute error bound for error-bounded lossy compression, from
 * CLIO_NEUROPRESS_ERROR_BOUND. 0 (the default) means LOSSLESS and is what this
 * driver did unconditionally before -- NeuroPress's ranking masks all 16
 * quantize actions to -INFINITY at 0, so the search covers the 16 lossless
 * configurations only. A positive bound makes the other half reachable.
 *
 * Same name and same meaning as the knob neuropress_explore_sweep.cc already
 * reads, and as gpucompress_config_t::error_bound upstream.
 */
double ErrorBoundFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  if (e == nullptr || *e == '\0') return 0.0;
  const double v = std::atof(e);
  return v > 0.0 ? v : 0.0;
}

/**
 * Value-range-relative bound, from CLIO_NEUROPRESS_REL_BOUND: with eps > 0 the
 * quantizer bounds each chunk by eps x its (max - min) rather than by the
 * absolute CLIO_NEUROPRESS_ERROR_BOUND (QuantizeDevice). 0 = absolute mode.
 *
 * @return eps, or 0
 */
double RelativeBoundFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_REL_BOUND");
  if (e == nullptr || *e == '\0') return 0.0;
  const double v = std::atof(e);
  return v > 0.0 ? v : 0.0;
}

/**
 * The bound one chunk was quantized to, for the element-wise check.
 *
 * @param src   the chunk's original bytes
 * @param bytes their size
 * @param f64   element width is 8 bytes
 * @param eb    the absolute bound
 * @param rel   the relative bound, 0 in absolute mode
 * @return rel x the chunk's value range (NaN ignored) in relative mode, the
 *   absolute bound otherwise and for a constant chunk, which is exact anyway
 */
double ChunkBound(const char *src, size_t bytes, bool f64, double eb, double rel) {
  if (rel <= 0.0) return eb;
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  const size_t n = bytes / (f64 ? sizeof(double) : sizeof(float));
  for (size_t i = 0; i < n; ++i) {
    const double v = f64 ? reinterpret_cast<const double *>(src)[i]
                         : static_cast<double>(reinterpret_cast<const float *>(src)[i]);
    if (std::isnan(v)) continue;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  return (hi > lo) ? rel * (hi - lo) : eb;
}

/**
 * Quality floor for lossy selection, in dB, from CLIO_NEUROPRESS_TARGET_PSNR.
 * With a positive error bound, NeuroPress then refuses to quantize a chunk
 * whose analytical PSNR (its value range against the bound) is below the
 * floor, and stores it losslessly instead. 0 (the default) disables the floor.
 *
 * @return the floor in whole dB, or 0
 */
unsigned TargetPsnrFromEnv() {
  const char *e = std::getenv("CLIO_NEUROPRESS_TARGET_PSNR");
  if (e == nullptr || *e == '\0') return 0;
  const double v = std::atof(e);
  return v > 0.0 ? static_cast<unsigned>(std::lround(v)) : 0;
}

/**
 * Source file and offset a blob came from, recovered from its NAME.
 *
 * Blobs are named "<frame>/<stem>/chunk_<i>" (see where records are built), so
 * the original bytes are at `<dir>/<frame>/<stem><ext>` + i*chunk. Deriving it
 * rather than storing it keeps the cold read-back path working too: that one
 * rebuilds its records from blobs.csv, which carries the name and nothing
 * about where the data came from.
 *
 * Returns false when the name does not have that shape (a caller that named
 * its blobs differently), which the bound check reports rather than skips.
 */
bool SourceOfBlob(const std::string &name, const std::string &dir,
                  const std::string &ext, size_t chunk,
                  std::string *path, size_t *offset) {
  const size_t c = name.rfind("/chunk_");
  if (c == std::string::npos) return false;
  const size_t slash = name.rfind('/', c - 1);
  if (slash == std::string::npos) return false;
  const std::string frame = name.substr(0, slash);
  const std::string stem = name.substr(slash + 1, c - slash - 1);
  const size_t idx = std::strtoull(name.c_str() + c + 7, nullptr, 10);
  *path = dir + "/" + frame + "/" + stem + ext;
  *offset = idx * chunk;
  return true;
}

/**
 * fdatasync every file this run made durable, inside the timed window.
 *
 * NOTHING in the write path fsyncs (bdev writes are buffered), so "durable"
 * used to mean different things per arm: on the node's xfs an arm's whole
 * output stayed dirty in memory past the timer and drained later, untimed,
 * while Lustre pushed the bytes out inside the loop. An arm on local NVMe was
 * therefore timed against the page cache and one on the PFS was not, which is
 * most of what the tiering ablation appeared to show.
 *
 * CLIO_REPLAY_FSYNC is a comma-separated list of tier paths as composed; each
 * is synced along with the per-node files the runtime actually created
 * (<path>_node<N>), so the caller does not have to know the node suffix.
 *
 * THE BARRIER MUST FAIL LOUDLY. This used to skip an unopenable file and
 * ignore a failed fdatasync, so a run could report a time without ever having
 * made its bytes durable -- the one thing the barrier exists to guarantee.
 * Now: every fdatasync error is fatal, and a tier path that syncs NOTHING is
 * fatal, because a tier that produced no file at all means the arm did not
 * write where the harness believes it did. The composed path itself is allowed
 * to be absent: the runtime writes <path>_node<N>, so the bare path usually
 * does not exist and only the siblings do.
 *
 * @param spec comma-separated tier paths, or empty to do nothing.
 * @param out_files receives how many files were synced.
 * @param out_error receives why the barrier did not hold; empty on success.
 * @return seconds spent in fdatasync, 0 when the list is empty.
 */
double FsyncTiers(const std::string &spec, size_t *out_files,
                  std::string *out_error) {
  *out_files = 0;
  out_error->clear();
  if (spec.empty()) return 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  size_t start = 0;
  while (start <= spec.size()) {
    const size_t comma = spec.find(',', start);
    std::string path = spec.substr(start, comma == std::string::npos
                                              ? std::string::npos
                                              : comma - start);
    start = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
    if (path.empty()) continue;
    // The composed path, plus <path>_node<N> that the runtime writes.
    std::vector<std::string> candidates = {path};
    const fs::path dir = fs::path(path).parent_path();
    const std::string stem = fs::path(path).filename().string();
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      if (name.size() > stem.size() && name.compare(0, stem.size(), stem) == 0) {
        candidates.push_back(e.path().string());
      }
    }
    size_t synced_here = 0;
    for (const auto &c : candidates) {
      const int fd = ::open(c.c_str(), O_RDONLY);
      if (fd < 0) {
        // ENOENT on the composed path is the normal case, not a failure; any
        // other errno is a file that exists and could not be opened.
        if (errno != ENOENT && out_error->empty()) {
          *out_error = "cannot open " + c + ": " + std::strerror(errno);
        }
        continue;
      }
      const bool ok = ::fdatasync(fd) == 0;
      const int err = errno;
      ::close(fd);
      if (!ok) {
        if (out_error->empty()) {
          *out_error = "fdatasync failed on " + c + ": " + std::strerror(err);
        }
        continue;
      }
      ++synced_here;
      ++*out_files;
    }
    if (synced_here == 0 && out_error->empty()) {
      *out_error = "no file to sync for tier path " + path +
                   " (neither it nor any " + stem + "_node* sibling exists)";
    }
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

struct BlobRecord {
  std::string name;
  size_t bytes = 0;
  uint64_t digest = 0;
  // Where the chunk came from, so its digest can be taken after the timed
  // loop. Points into the driver's file list; null for read-back records.
  const fs::path *src_file = nullptr;
  size_t src_off = 0;
  int lib = 0;
  double ratio = 0.0;
  size_t stored = 0;
  double ms = 0.0;
  // Quantize + byte-shuffle kernel time for this chunk, separate from `ms`
  // (the codec launch alone). Reported so a caller can measure the work that
  // happens AFTER the data is device-resident: preproc_ms + compress_ms.
  double preproc_ms = 0.0;
  // Host-to-device staging for this chunk; 0 when the caller was already
  // device-resident (in situ) or staging was not requested.
  double h2d_ms = 0.0;
  // MEASURED decompression time (CUDA events around the codec call alone), or
  // <0 when nothing measured one. Needs CLIO_NEUROPRESS_EXPLORE_MEASURE_DT.
  // Same field, same clock and same caveat as the LAMMPS driver's dt_ms: NOT
  // comparable with a read-path Decompress(), which times the whole path.
  double dt_ms = -1.0;
  bool ok = false;
};

/**
 * Fill each record's FNV-1a digest from the source bytes on disk.
 *
 * The digest only feeds the round-trip check and blobs.csv, so it is
 * bookkeeping, not part of the write path. Hashing inside the timed loop
 * charged 5-7 s per 4 GiB arm to every arm -- lossy ones too, which never
 * verify -- so it is computed here, after the timer has stopped, by re-reading
 * each chunk from the file and offset recorded when it was written.
 *
 * @param records blobs written this run; `bytes`, `src_file` and `src_off`
 *                must be set.
 * @return number of records whose source could not be re-read; those keep
 *         digest 0, so a later round-trip check reports them as mismatches.
 */
size_t FillDigestsFromSource(std::vector<BlobRecord> *records) {
  size_t missing = 0;
  std::vector<char> buf;
  for (auto &r : *records) {
    if (r.src_file == nullptr) { ++missing; continue; }
    std::ifstream in(*r.src_file, std::ios::binary);
    buf.resize(r.bytes);
    in.seekg(static_cast<std::streamoff>(r.src_off));
    in.read(buf.data(), static_cast<std::streamsize>(r.bytes));
    if (!in || in.gcount() != static_cast<std::streamsize>(r.bytes)) {
      ++missing;
      continue;
    }
    r.digest = Fnv1a(buf.data(), r.bytes);
  }
  return missing;
}

struct Pending {
  clio::run::Future<clio::cte::compressor::DynamicScheduleTask> fut;
  clio::run::Future<clio::cte::core::PutBlobTask> put;  // --no-compress
  ctp::ipc::FullPtr<char> buf;
  /** --no-compress only: the device copy the bdev writes from, freed on drain.
      Null on every other path, where the compressor owns its own staging. */
  ctp::ipc::AllocatorId dev_alloc;
  size_t record;
};

/* Group a dump filename by its physical field, so the report can say which
   fields compress. The dumper writes fab%04d_comp%02d_<name>.f32, so the
   field is whatever follows the component index. Anything that does not
   match that shape is grouped under its stem, which is still useful. */
std::string FieldOf(const std::string &stem) {
  const size_t c = stem.find("_comp");
  if (c == std::string::npos) return stem;
  const size_t u = stem.find('_', c + 5);
  if (u == std::string::npos) return stem;
  return stem.substr(u + 1);
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 2;
  // Process start. Runtime bring-up, field discovery and the payload du all
  // happen before the workload proper and are IDENTICAL for every arm, so they
  // are reported as `setup` and EXCLUDED from `total` (t_work, below): leaving
  // them in added ~2 s of the same work to every bar and diluted every
  // comparison the figure makes.
  const auto t_proc = std::chrono::steady_clock::now();

  // ---- Clio: attach to the runtime, which CLIO_WITH_RUNTIME=1 brings up
  // in THIS process, composed from CLIO_SERVER_CONF. Same call the HDF5 VOL
  // makes lazily; here it is the first thing main does.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed -- is CLIO_SERVER_CONF set and "
                 "CLIO_WITH_RUNTIME=1?\n";
    return 1;
  }
  auto *cte_client = CLIO_CTE_CLIENT;

  unsigned pool_major = 512, pool_minor = 0;
  if (const char *e = std::getenv("CLIO_REPLAY_COMPRESSOR_POOL"); e && *e) {
    if (std::sscanf(e, "%u.%u", &pool_major, &pool_minor) < 1) {
      std::cerr << "bad CLIO_REPLAY_COMPRESSOR_POOL\n";
      return 1;
    }
  }
  clio::cte::compressor::Client compressor(
      clio::run::PoolId(pool_major, pool_minor));

  auto tag = cte_client->AsyncGetOrCreateTag(opt.tag);
  tag.Wait();
  if (tag->GetReturnCode() != 0) { std::cerr << "GetOrCreateTag failed\n"; return 1; }
  const auto tag_id = tag->tag_id_;

  // Error bound in force for this run, so verification can pick the check that
  // actually applies: bit-exact for lossless, |orig-decoded| <= eb for lossy.
  const double verify_eb = ErrorBoundFromEnv();
  const double verify_rel = RelativeBoundFromEnv();
  size_t bound_checked = 0, bound_exceeded = 0, bound_unreadable = 0;
  double bound_worst = 0.0;

  // One read-back record's verdict from its decoded bytes: the digest, or
  // under a positive bound "decompressed without error" plus, with
  // --check-bound, the element-wise comparison against the source file.
  // Driver work, never inside a read's timed interval.
  std::vector<char> srcbuf;
  auto check_record = [&](const BlobRecord &r, const char *data,
                          int rc_get) -> bool {
    // Which check applies is decided by the error bound, not by a separate
    // flag: under lossy compression the decoded bytes are NOT the input
    // bytes by construction, so a digest comparison there reports a failure
    // that is not one. --check-bound asks for the element-wise comparison
    // that tests the guarantee the bound actually makes.
    bool ok;
    if (verify_eb > 0.0) {
      // A positive bound rules the digest out on its own: nothing lossy can
      // reproduce the input bytes, so decompressing without error is the
      // whole round-trip check available here. The element-wise test against
      // the source files is what --check-bound adds, and it is deliberately
      // separate: it re-reads every source chunk, which a TIMED read-back
      // must not do.
      ok = rc_get == 0;
      std::string src;
      size_t off = 0;
      if (opt.check_bound && ok &&
          SourceOfBlob(r.name, opt.dir, opt.ext, opt.chunk, &src, &off)) {
        std::ifstream in(src, std::ios::binary);
        if (in) {
          srcbuf.resize(r.bytes);
          in.seekg(static_cast<std::streamoff>(off));
          in.read(srcbuf.data(), static_cast<std::streamsize>(r.bytes));
          if (in.gcount() == static_cast<std::streamsize>(r.bytes)) {
            // Compare at the element width the run declared. A float64
            // replay quantizes as float64; reading it as float32 would
            // compare halves of values against each other.
            double worst = 0.0;
            if (opt.f64) {
              const auto *a = reinterpret_cast<const double *>(srcbuf.data());
              const auto *b = reinterpret_cast<const double *>(data);
              for (size_t i = 0; i < r.bytes / sizeof(double); ++i)
                worst = std::max(worst, std::fabs(a[i] - b[i]));
            } else {
              const auto *a = reinterpret_cast<const float *>(srcbuf.data());
              const auto *b = reinterpret_cast<const float *>(data);
              for (size_t i = 0; i < r.bytes / sizeof(float); ++i)
                worst = std::max(worst,
                                 std::fabs(static_cast<double>(a[i]) -
                                           static_cast<double>(b[i])));
            }
            ++bound_checked;
            bound_worst = std::max(bound_worst, worst);
            const double bound =
                ChunkBound(srcbuf.data(), r.bytes, opt.f64, verify_eb, verify_rel);
            if (worst > bound) {
              ++bound_exceeded;
              // Scientific, and NOT the stream's inherited precision:
              // the summary block below sets fixed/3dp for ratios ("5.370x"),
              // which is sticky and would round an error of 9.49e-04 to
              // "0.001" -- indistinguishable from the bound it is being
              // checked against.
              std::cerr << std::scientific << std::setprecision(6)
                        << "  BOUND EXCEEDED " << r.name << " max|err|="
                        << worst << " > eb=" << bound << "\n";
            }
          } else {
            ++bound_unreadable;
          }
        } else {
          ++bound_unreadable;
        }
      } else if (opt.check_bound && ok) {
        ++bound_unreadable;
      }
    } else {
      ok = rc_get == 0 && Fnv1a(data, r.bytes) == r.digest;
    }
    if (!ok) std::cerr << "  MISMATCH " << r.name << " rc=" << rc_get << "\n";
    /* Optionally hand the decompressed bytes to an external checker. The
       digest above is computed by the same program that computed the
       original one, so it proves the round trip is self-consistent; writing
       the bytes out lets something else compare them against the
       simulation's own output and remove this program from the loop. */
    if (!opt.dump_dir.empty() && rc_get == 0) {
      std::string fn = r.name;
      for (auto &ch : fn) if (ch == '/') ch = '_';
      std::ofstream(opt.dump_dir + "/" + fn + ".bin", std::ios::binary)
          .write(data, static_cast<std::streamsize>(r.bytes));
    }
    return ok;
  };

  // One read-back destination: host shared memory, or with --read-to-gpu a
  // device buffer registered with the runtime (a GPU consumer's memory; the
  // runtime decodes straight into it, and there is no D2H of the result).
  struct ReadBuf {
    ctp::ipc::FullPtr<char> host;
    ctp::ipc::AllocatorId dev_alloc;
    char *dev = nullptr;
    ctp::ipc::ShmPtr<void> shm;
  };
  // A destination for `bytes`, cleared unless `clear` is false; false when it
  // cannot be allocated.
  auto alloc_read_buf = [&](size_t bytes, ReadBuf *b, bool clear) -> bool {
    if (opt.read_to_gpu) {
      b->dev_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, bytes,
          &b->dev);
      if (b->dev_alloc.IsNull() || b->dev == nullptr) return false;
      if (clear) ctp::GpuApi::Memset(b->dev, 0, bytes);
      // off_ carries the raw device address, which ToFullPtr resolves for the
      // process that minted the id (ipc_manager.h, "Case 4").
      b->shm = ctp::ipc::ShmPtr<void>(b->dev_alloc, reinterpret_cast<size_t>(b->dev));
      return true;
    }
    b->host = CLIO_IPC->AllocateBuffer(bytes);
    if (b->host.IsNull()) return false;
    if (clear) std::memset(b->host.ptr_, 0, bytes);
    b->shm = b->host.shm_.template Cast<void>();
    return true;
  };
  // The decoded bytes on the host for check_record: the buffer itself, or a
  // copy of the device buffer when a check or the dump needs the data.
  std::vector<char> host_copy;
  auto host_view = [&](const ReadBuf &b, size_t bytes) -> const char * {
    if (!opt.read_to_gpu) return b.host.ptr_;
    const bool need = verify_eb <= 0.0 || opt.check_bound || !opt.dump_dir.empty();
    if (!need) return nullptr;
    host_copy.resize(bytes);
    ctp::DeviceAwareMemcpy(host_copy.data(), b.dev, bytes);
    return host_copy.data();
  };
  auto free_read_buf = [&](ReadBuf &b) {
    if (opt.read_to_gpu) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, b.dev_alloc);
    else CLIO_IPC->FreeBuffer(b.host);
  };
  // Issue one read into `b`: a raw GetBlob for the baseline, else the
  // compressor's decompress.
  auto issue_get = [&](const BlobRecord &r, const ReadBuf &b) {
    return compressor.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), tag_id, r.name, 0, r.bytes, 0, b.shm,
        cte_client->pool_id_);
  };
  auto issue_raw = [&](const BlobRecord &r, const ReadBuf &b) {
    return cte_client->AsyncGetBlob(tag_id, r.name, 0, r.bytes, 0, b.shm,
                                    clio::run::PoolQuery::Local());
  };
  // A record's verdict once its read finished, plus the buffer's release.
  // Driver work: callers keep it out of the timed interval.
  auto finish_record = [&](const BlobRecord &r, ReadBuf &b, int rc,
                           size_t *bad, size_t *got_bytes) {
    if (rc == 0) *got_bytes += r.bytes;
    const char *data = host_view(b, r.bytes);
    // With the data on the device and nothing to check it against, only the
    // return code is left -- which is all a lossy run checks anyway.
    const bool ok = data != nullptr ? check_record(r, data, rc) : rc == 0;
    if (!ok) {
      ++*bad;
      if (data == nullptr) std::cerr << "  MISMATCH " << r.name << " rc=" << rc << "\n";
    }
    free_read_buf(b);
  };

  // Read-back with up to --read-inflight requests outstanding: a consumer
  // that prefetches, so the runtime's workers decode chunks concurrently.
  //
  // A TIMING mode. Its time is the wall clock of the whole loop -- issuing,
  // waiting, and allocating and releasing each buffer, as a prefetching
  // consumer would -- with nothing subtracted: the reads still in flight keep
  // progressing during any driver work, so subtracting that work would remove
  // read time with it. For the same reason nothing heavy runs inside the loop:
  // buffers are not cleared and only each read's return code is checked. The
  // data are verified by a sequential read (the harness's first), and a
  // windowed read that must dump or check them says its time includes that.
  auto read_windowed = [&](const std::vector<BlobRecord> &recs, size_t *bad,
                           double *get_ms, size_t *got_bytes) -> bool {
    const bool inspect = !opt.dump_dir.empty() || opt.check_bound;
    if (inspect) {
      std::cerr << "note: windowed read with --dump-decompressed/--check-bound;"
                   " its time includes that work\n";
    }
    auto run = [&](auto issue) -> bool {
      using Fut = decltype(issue(recs[0], std::declval<const ReadBuf &>()));
      struct Slot {
        size_t idx;
        ReadBuf buf;
        Fut fut;
      };
      std::deque<Slot> window;
      auto finish = [&](Slot &slot) {
        slot.fut.Wait();
        const int rc = slot.fut->GetReturnCode();
        const BlobRecord &r = recs[slot.idx];
        if (inspect) {
          finish_record(r, slot.buf, rc, bad, got_bytes);
          return;
        }
        if (rc == 0) *got_bytes += r.bytes;
        else {
          ++*bad;
          std::cerr << "  MISMATCH " << r.name << " rc=" << rc << "\n";
        }
        free_read_buf(slot.buf);
      };
      const auto t_start = std::chrono::steady_clock::now();
      for (size_t i = 0; i < recs.size(); ++i) {
        if (window.size() >= opt.read_inflight) {
          finish(window.front());
          window.pop_front();
        }
        ReadBuf buf;
        if (!alloc_read_buf(recs[i].bytes, &buf, inspect)) {
          std::cerr << "read buffer allocation failed\n";
          return false;
        }
        Fut fut = issue(recs[i], buf);
        window.push_back(Slot{i, buf, std::move(fut)});
      }
      while (!window.empty()) {
        finish(window.front());
        window.pop_front();
      }
      *get_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_start).count();
      return true;
    };
    return opt.no_compress ? run(issue_raw) : run(issue_get);
  };

  // Read-back one request at a time. The time is the read path alone: the
  // GetBlob / decompress call and its wait. The digest, the bound check and
  // the optional dump are verification, so a cold read-back's time is not
  // inflated by them.
  auto read_sequential = [&](const std::vector<BlobRecord> &recs, size_t *bad,
                             double *get_ms, size_t *got_bytes) -> bool {
    for (const auto &r : recs) {
      ReadBuf buf;
      if (!alloc_read_buf(r.bytes, &buf, /*clear=*/true)) {
        std::cerr << "read buffer allocation failed\n";
        return false;
      }
      int rc_get;
      const auto t_get = std::chrono::steady_clock::now();
      if (opt.no_compress) {
        auto get = issue_raw(r, buf);
        get.Wait();
        rc_get = get->GetReturnCode();
      } else {
        auto get = issue_get(r, buf);
        get.Wait();
        rc_get = get->GetReturnCode();
      }
      *get_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_get).count();
      finish_record(r, buf, rc_get, bad, got_bytes);
    }
    return true;
  };

  auto verify_records = [&](const std::vector<BlobRecord> &recs) -> bool {
    size_t bad = 0;
    double get_ms = 0.0;
    size_t got_bytes = 0;
    const bool read_ok =
        opt.read_inflight > 1
            ? read_windowed(recs, &bad, &get_ms, &got_bytes)
            : read_sequential(recs, &bad, &get_ms, &got_bytes);
    if (!read_ok) return false;
    // Say which check ran. In bound mode the bytes are NOT expected to match
    // bit for bit, so claiming they did would be false on every lossy run --
    // `bad` there counts decompression failures only, and the verdict on the
    // data is the BOUND line below.
    // Any positive bound, --check-bound or not: claiming a bit-exact round trip
    // on a lossy run would be false whether or not the bound was also checked.
    const bool bound_mode = verify_eb > 0.0;
    std::cout << (bad == 0 ? "VERIFIED: " : "FAILED: ") << (recs.size() - bad)
              << " of " << recs.size()
              << (bound_mode
                      ? (opt.check_bound
                             ? " blobs decompressed without error (bound"
                               " checked separately below)"
                             : " blobs decompressed without error (lossy: no"
                               " bit-exact check, and --check-bound was not"
                               " asked for)")
                      : " blobs round-tripped bit-exact through the"
                        " decompressor")
              << std::endl;
    std::cout << "  get+decompress: " << get_ms << " ms for " << got_bytes
              << " B in " << recs.size() << " blob(s)" << std::endl;
    return bad == 0;
  };

  // One line the harness greps, in the same shape as VERIFIED:/FAILED: above.
  // A chunk whose source could not be re-read is reported, never counted as a
  // pass -- the whole point of this check is that it not be silently vacuous.
  auto report_bound = [&]() {
    if (!(opt.check_bound && verify_eb > 0.0)) return true;
    const bool ok = bound_exceeded == 0 && bound_checked > 0;
    // AN ERROR MAGNITUDE IS NOT A RATIO. The summary block sets
    // fixed/setprecision(3) for ratio display and that setting is sticky, so
    // this line inherited it and printed a worst error of 9.494e-04 as
    // "0.001" -- exactly the bound, so a cross-check could not tell agreement
    // from a violation, and at eb=1e-3 the rounding alone was 5% of the value.
    // Errors here span 1e-7 to 1e0; scientific with 6 significant digits
    // covers that range and the eb beside it.
    const std::ios_base::fmtflags saved_flags = std::cout.flags();
    const std::streamsize saved_prec = std::cout.precision();
    std::cout << std::scientific << std::setprecision(6);
    std::cout << (ok ? "BOUND OK: " : "BOUND FAILED: ") << bound_checked
              << " chunk(s) checked against eb=" << verify_eb
              << ", " << bound_exceeded << " exceeded, worst max|err|="
              << bound_worst;
    if (bound_unreadable) std::cout << ", " << bound_unreadable
                                    << " source(s) unreadable";
    std::cout << std::endl;
    std::cout.flags(saved_flags);
    std::cout.precision(saved_prec);
    return ok;
  };

  // ---- Cold read-back: no files touched, the tier is the only source. ----
  if (opt.readback) {
    std::ifstream csv(opt.report);
    if (!csv) { std::cerr << "cannot open " << opt.report << "\n"; return 1; }
    std::vector<BlobRecord> recs;
    std::string line;
    std::getline(csv, line);  // header
    while (std::getline(csv, line)) {
      size_t p1 = line.find(','), p2 = line.find(',', p1 + 1),
             p3 = line.find(',', p2 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) continue;
      BlobRecord r;
      r.name = line.substr(0, p1);
      r.bytes = std::strtoull(line.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr, 10);
      r.digest = std::strtoull(line.substr(p2 + 1, p3 - p2 - 1).c_str(), nullptr, 16);
      recs.push_back(r);
    }
    std::cout << "cold read-back of " << recs.size() << " blob(s) listed in "
              << opt.report << std::endl;
    const bool ok = verify_records(recs) && report_bound();
    clio::run::CLIO_RUNTIME_FINALIZE();
    return ok ? 0 : 1;
  }

  // ---- Collect the field files, in replay order. ----
  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto &e : fs::recursive_directory_iterator(opt.dir, ec)) {
    if (e.is_regular_file() && e.path().extension() == opt.ext) {
      files.push_back(e.path());
    }
  }
  if (ec || files.empty()) {
    std::cerr << "no " << opt.ext << " files under " << opt.dir << "\n";
    return 1;
  }
  // Lexical order == chronological, because the dumper zero-pads the
  // timestep. Sorting the FULL path keeps each timestep's files together
  // even when they live in per-timestep subdirectories.
  std::sort(files.begin(), files.end());
  if (opt.max_files && files.size() > opt.max_files) files.resize(opt.max_files);

  size_t payload = 0;
  for (const auto &f : files) payload += fs::file_size(f);

  const size_t elem = opt.f64 ? 8 : 4;
  std::cout << (opt.no_compress
                    ? "field replay -> Clio core, NO compression (baseline), "
                      "runtime in-process\n"
                    : "field replay -> Clio compressor, runtime in-process\n")
            << "  " << files.size() << " file(s) under " << opt.dir
            << "  payload=" << payload / 1048576.0 << " MiB  ("
            << (opt.f64 ? "float64" : "float32") << ")\n"
            << "  chunk=" << (opt.chunk ? std::to_string(opt.chunk)
                                        : std::string("whole file"))
            << "  cte client pool=" << cte_client->pool_id_.major_ << "."
            << cte_client->pool_id_.minor_ << "  compressor pool=" << pool_major
            << "." << pool_minor
            << std::endl;

  // float32 = 1, float64 = 2 -- the same encoding the HDF5 VOL assigns from
  // an H5T_FLOAT's size. error_bound_ is 0 (lossless) unless
  // CLIO_NEUROPRESS_ERROR_BOUND asks otherwise; see ErrorBoundFromEnv.
  clio::cte::core::Context ctx;
  ctx.data_type_ = opt.f64 ? 2 : 1;
  ctx.error_bound_ = ErrorBoundFromEnv();
  ctx.target_psnr_ = TargetPsnrFromEnv();
  if (ctx.error_bound_ > 0.0)
    std::cout << "  error bound=" << ctx.error_bound_
              << "  LOSSY (quantize actions enabled)" << std::endl;
  if (ctx.error_bound_ > 0.0 && ctx.target_psnr_ > 0)
    std::cout << "  quality floor=" << ctx.target_psnr_
              << " dB (a chunk below it is stored losslessly)" << std::endl;

  std::vector<BlobRecord> records;
  std::vector<Pending> pending;
  double read_s = 0.0, stage_s = 0.0;
  // --no-compress only: host->device staging, a replay artifact the
  // harness subtracts the same way it subtracts the compressor's h2d_ms.
  double baseline_h2d_s = 0.0;

  auto drain = [&]() {
    for (auto &p : pending) {
      auto &r = records[p.record];
      if (opt.no_compress) {
        p.put.Wait();
        r.ok = p.put->GetReturnCode() == 0;
        r.lib = 0;  // stored raw, by construction
        r.ratio = 1.0;
        r.stored = r.bytes;
        if (!p.dev_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, p.dev_alloc);
        }
        CLIO_IPC->FreeBuffer(p.buf);
        continue;
      }
      p.fut.Wait();
      const auto &c = p.fut->context_;
      r.ok = p.fut->GetReturnCode() == 0;
      r.lib = c.compress_lib_;
      r.ratio = c.actual_compression_ratio_;
      r.ms = c.actual_compress_time_ms_;
      r.preproc_ms = c.actual_preproc_time_ms_;
      r.h2d_ms = c.actual_h2d_time_ms_;
      r.dt_ms = c.actual_decompress_time_ms_;
      // lib == 0 marks "stored raw": the codec did not shrink the chunk and
      // the caller's bytes went to the tier untouched, so the stored size is
      // the original -- actual_compressed_size_ still reports the codec's
      // output, which is NOT what was stored.
      r.stored = (r.ok && r.lib != 0) ? c.actual_compressed_size_ : r.bytes;
      CLIO_IPC->FreeBuffer(p.buf);
    }
    pending.clear();
  };

  // The workload starts here: read, stage+compress, then the final flush.
  const auto t_work = std::chrono::steady_clock::now();
  const double setup_s =
      std::chrono::duration<double>(t_work - t_proc).count();

  std::vector<char> filebuf;
  for (const auto &f : files) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t sz = fs::file_size(f);
    if (sz == 0) continue;
    if (sz % elem != 0) {
      std::cerr << "warning: " << f.filename().string() << " is " << sz
                << " bytes, not a whole number of "
                << (opt.f64 ? "float64" : "float32") << " elements\n";
    }
    filebuf.resize(sz);
    std::ifstream in(f, std::ios::binary);
    if (!in.read(filebuf.data(), static_cast<std::streamsize>(sz))) {
      std::cerr << "short read on " << f << "\n";
      return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    read_s += std::chrono::duration<double>(t1 - t0).count();

    const size_t chunk = (opt.chunk == 0 || opt.chunk > sz) ? sz : opt.chunk;
    const size_t nchunks = (sz + chunk - 1) / chunk;
    const std::string stem = f.stem().string();
    // Name blobs by the timestep directory plus the file stem, so a blob
    // name identifies the frame it came from even though every timestep
    // reuses the same component filenames.
    const std::string frame = f.parent_path().filename().string();

    for (size_t ci = 0; ci < nchunks; ++ci) {
      const size_t off = ci * chunk;
      const size_t n = std::min(chunk, sz - off);
      const char *src = filebuf.data() + off;

      auto buf = CLIO_IPC->AllocateBuffer(n);
      if (buf.IsNull()) { std::cerr << "AllocateBuffer failed\n"; return 1; }
      std::memcpy(buf.ptr_, src, n);

      BlobRecord rec;
      rec.name = frame + "/" + stem + "/chunk_" + std::to_string(ci);
      rec.bytes = n;
      rec.src_file = &f;  // digest is taken after the timer stops
      rec.src_off = off;
      records.push_back(rec);

      Pending p;
      if (opt.no_compress) {
        // BASELINE STARTS ON THE GPU, like every arm it is compared against.
        //
        // fs_bdev_transport.cc:328-334 starts the I/O timer BEFORE the
        // device-to-host copy a device-resident blob implies -- its own
        // comment calls that "the single D2H that terminates the device
        // path". A compressed arm hands the bdev a DEVICE pointer and so pays
        // that copy inside io_s; a Baseline that handed over a HOST buffer
        // skipped it entirely and was credited a free payload-sized D2H.
        // On a 40 GiB arm that is not a rounding error.
        //
        // So stage the chunk up and put the device pointer. The H2D itself is
        // a REPLAY artifact -- an in-situ producer already has the data on the
        // device -- so it is timed separately and reported for subtraction,
        // exactly as the compressor's own h2d_ms is.
        char *dev = nullptr;
        p.dev_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
            /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, n,
            &dev);
        if (p.dev_alloc.IsNull() || dev == nullptr) {
          std::cerr << "baseline: device staging failed for " << rec.name
                    << " (" << n << " B)\n";
          return 1;
        }
        const auto h2d_t0 = std::chrono::steady_clock::now();
        ctp::DeviceAwareMemcpy(dev, buf.ptr_, n);
        baseline_h2d_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - h2d_t0).count();
        // AllocateAndRegisterGpuBackend's convention: off_ carries the raw
        // device address, which IpcManager::ToFullPtr resolves for the process
        // that minted the id (ipc_manager.h, "Case 4").
        ctp::ipc::ShmPtr<void> dev_shm(p.dev_alloc,
                                       reinterpret_cast<size_t>(dev));
        p.put = cte_client->AsyncPutBlob(
            tag_id, rec.name, 0, n, dev_shm, -1.0f,
            clio::cte::core::Context(), 0, clio::run::PoolQuery::Local());
      } else {
        p.fut = compressor.AsyncDynamicSchedule(
            clio::run::PoolQuery::Local(), tag_id, rec.name, 0, n,
            buf.shm_.template Cast<void>(), -1.0f, ctx, 0, cte_client->pool_id_);
      }
      p.buf = buf;
      p.record = records.size() - 1;
      pending.push_back(std::move(p));
    }
    // Drain per file: the staging buffers of a 4 MiB-chunked 8 MiB file are
    // cheap, but a whole run's worth is not, and the exploration modes make
    // each chunk expensive enough that unbounded queueing buys nothing.
    drain();
    stage_s += std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t1).count();
  }
  drain();

  // CLIO_REPLAY_FINAL_FLUSH=1: move volatile blobs to a durable tier, inside
  // `total`. The fdatasync below then makes the bytes durable, also inside
  // `total`, so every arm's bar ends at the same guarantee.
  double flush_s = 0.0;
  unsigned long long flushed_bytes = 0, flushed_blobs = 0;
  int flush_rc = 0;
  bool flush_ran = false;
  if (const char *e = std::getenv("CLIO_REPLAY_FINAL_FLUSH");
      e && *e && *e != '0') {
    const auto tf = std::chrono::steady_clock::now();
    auto fl = cte_client->AsyncFlushData(clio::run::PoolQuery::Local(),
                                         /*target_persistence_level=*/1,
                                         /*period_us=*/0);
    fl.Wait();
    flush_s = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - tf).count();
    flushed_bytes = fl->bytes_flushed_;
    flushed_blobs = fl->blobs_flushed_;
    flush_rc = fl->GetReturnCode();
    flush_ran = true;
  }

  // DURABILITY BARRIER, inside `total`: see FsyncTiers.
  size_t synced_files = 0;
  std::string fsync_error;
  const char *fsync_spec = std::getenv("CLIO_REPLAY_FSYNC");
  const double fsync_s = FsyncTiers(fsync_spec != nullptr ? fsync_spec : "",
                                    &synced_files, &fsync_error);

  // ---- Report. ----
  size_t stored_total = 0, in_total = 0, kept = 0, raw = 0, failed = 0;
  std::map<std::string, std::pair<size_t, size_t>> per_field;
  std::map<int, size_t> per_lib;
  for (const auto &r : records) {
    if (!r.ok) { ++failed; continue; }
    in_total += r.bytes;
    stored_total += r.stored;
    if (r.lib != 0) { ++kept; ++per_lib[r.lib]; } else ++raw;
    const size_t s1 = r.name.find('/');
    const size_t s2 = r.name.find('/', s1 + 1);
    auto &pf = per_field[FieldOf(r.name.substr(s1 + 1, s2 - s1 - 1))];
    pf.first += r.bytes;
    pf.second += r.stored;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "\nstored " << records.size() << " blob(s), " << in_total
            << " B in -> " << stored_total << " B on the tier  (stored ratio "
            << (stored_total ? double(in_total) / double(stored_total) : 0.0)
            << ")\n  compressed: " << kept << "   stored raw: " << raw
            << "   failed: " << failed << "\n";
  for (const auto &[name, io] : per_field)
    std::cout << "  " << std::setw(14) << name << "  " << io.first << " -> "
              << io.second << "  ("
              << (io.second ? double(io.first) / io.second : 0.0) << "x)\n";
  for (const auto &[lib, n] : per_lib)
    std::cout << "  codec " << std::setw(16)
              << (lib == 0 ? std::string("raw(not-beneficial)")
                           : ctp::CompressionFactory::NameForWireId(lib))
              << " : " << n
              << " chunk(s)\n";
  // THE MEASURED WINDOW, in the same steady-clock nanoseconds the bdev's I/O
  // log stamps its writes with (io_log.h's Now()). Without it the harness can
  // only union whatever intervals the log holds, including any that ran before
  // the timer opened or after it closed; with it every interval is clipped to
  // [start, end] first, so the pale segment can never exceed the bar or count
  // work outside it.
  std::cout << "  window: start_ns "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   t_work.time_since_epoch()).count()
            << "   end_ns "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count()
            << std::endl;
  if (baseline_h2d_s > 0.0) {
    // Named h2d so the harness can subtract it exactly as it subtracts the
    // compressor's h2d_ms union: an in-situ producer would not pay it.
    std::cout << "  time: h2d " << baseline_h2d_s << " s   (baseline staging, "
                 "excluded from the bar)" << std::endl;
  }
  std::cout << "  time: read " << read_s << " s   stage+compress " << stage_s
            << " s   total "
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_work).count()
            << " s   (setup " << setup_s << " s, excluded)" << std::endl;
  if (synced_files > 0 || fsync_s > 0.0) {
    std::cout << "  fsync: " << synced_files << " file(s) in " << fsync_s
              << " s" << std::endl;
  }
  if (flush_ran) {
    std::cout << "  flush: " << flushed_blobs << " blob(s), " << flushed_bytes
              << " B moved to durable storage in " << flush_s << " s  (rc="
              << flush_rc << ")" << std::endl;
  }

  // Outside every timer above: the digest is verification bookkeeping.
  if (const size_t missing = FillDigestsFromSource(&records)) {
    std::cerr << "digest: could not re-read the source of " << missing
              << " blob(s)\n";
  }

  if (!opt.report.empty()) {
    std::ofstream csv(opt.report);
    // TWO ratios, because they answer different questions and disagreed
    // silently before this column existed:
    //   ratio        = bytes / CODEC PAYLOAD. Upstream's definition
    //                  (gpucompress_compress.cpp excludes its own header),
    //                  the NN's training label, and the number explore.csv
    //                  and selection.csv report. Do not change it: it feeds
    //                  the cost model and the SGD targets.
    //   stored_ratio = bytes / STORED. What the tier actually holds,
    //                  header included, and the basis of the aggregate
    //                  "(stored ratio N)" line printed above.
    // They differ by the 24-byte CTEC header (56 when quantized), so
    // sum(bytes)/sum(stored) never equals the mean of the `ratio` column.
    // Appended, never inserted: the first three columns are the cold
    // reader's contract.
    // wire 0 is NOT brotli here. It is how the compressor marks "stored raw",
    // and brotli merely happens to occupy wire id 0 in the factory registry --
    // so NameForWireId(0) answers "brotli" and a chunk that was never
    // compressed reads as a codec that never ran. Name the outcome instead.
    csv << "blob,bytes,fnv1a64,lib,codec,ratio,stored,compress_ms,preproc_ms,"
           "h2d_ms,decompress_ms,rc,stored_ratio\n";
    for (const auto &r : records)
      csv << r.name << ',' << r.bytes << ',' << std::hex << r.digest << std::dec
          << ',' << r.lib << ','
          << (r.lib == 0 ? std::string("raw(not-beneficial)")
                         : ctp::CompressionFactory::NameForWireId(r.lib))
          << ',' << r.ratio << ',' << r.stored << ',' << r.ms << ','
          << r.preproc_ms << ',' << r.h2d_ms << ',' << r.dt_ms << ',' << (r.ok ? 0 : 1) << ','
          << (r.stored ? double(r.bytes) / double(r.stored) : 0.0) << '\n';
  }

  int rc = failed ? 1 : 0;
  // THE BARRIER IS PART OF THE MEASUREMENT. A run whose bytes were never made
  // durable has not measured what this figure claims, so it fails outright --
  // the harness must not be able to record a time for it. One line in the same
  // grep-able shape as VERIFIED:/BOUND OK: above.
  if (!fsync_error.empty()) {
    std::cout << "FSYNC FAILED: " << fsync_error << std::endl;
    rc = 1;
  }
  // --verify after a write: --read-repeat timed read-backs (a memory-backed
  // tier must be read in this process), then one more, untimed and
  // sequential, that checks the data -- element-wise against the bound with
  // --check-bound, bit-exact against the digest for a lossless run -- so the
  // timed reads (return codes only) carry none of it.
  auto verify_repeated = [&](const std::vector<BlobRecord> &recs) -> bool {
    const bool check = opt.check_bound || verify_eb <= 0.0;
    const size_t window = opt.read_inflight;
    opt.check_bound = false;
    bool ok = true;
    for (size_t i = 1; i <= opt.read_repeat; ++i) {
      std::cout << "READ " << i << "/" << opt.read_repeat << std::endl;
      ok = verify_records(recs) && ok;
    }
    if (check) {
      opt.check_bound = verify_eb > 0.0;
      opt.read_inflight = 1;
      std::cout << "READ check (untimed; sequential, "
                << (verify_eb > 0.0 ? "element-wise bound" : "bit-exact digest")
                << ")" << std::endl;
      ok = verify_records(recs) && ok;
      opt.read_inflight = window;
    }
    return ok;
  };
  std::vector<BlobRecord> read_set;
  if (!opt.read_match.empty()) {
    const std::regex re(opt.read_match);
    for (const auto &r : records) {
      if (std::regex_search(r.name, re)) read_set.push_back(r);
    }
    std::cout << "read-back set: " << read_set.size() << " of " << records.size()
              << " blob(s) match '" << opt.read_match << "'" << std::endl;
  }
  if (opt.verify &&
      !verify_repeated(opt.read_match.empty() ? records : read_set)) {
    rc = 1;
  }
  // A bound violation is a FAILED run. Evaluated after verify_records, which
  // is what populates the counters, and independently of it: under --check-bound
  // verify_records only fails on a decompress error, so without this a run
  // whose data came back outside the bound would exit 0.
  if (opt.verify && !report_bound()) rc = 1;

  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
