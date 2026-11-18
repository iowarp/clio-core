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

/**
 * Mixed metadata + I/O reliability benchmark for the clio-fs filesystem chimod.
 *
 * Drives eight operations against a composed clio_cte_filesystem pool:
 *
 *   write_4k, read_4k, write_1m, read_1m   -- the I/O mix
 *   stat_size, readdir_small (10 entries),
 *   readdir_large (100 entries), rename    -- the metadata mix
 *
 * Each operation carries a percentage of the workload (--mix), so the same
 * binary produces both the COMBINED profile (an even split across all eight)
 * and the ISOLATED profile (100% of one operation). Comparing the two is the
 * point: metadata operations that are fast alone can collapse when they queue
 * behind 1 MiB transfers, and that interference is invisible in a
 * single-operation benchmark.
 *
 * The workload is TIME-based (--duration), not operation-count based, so every
 * profile gets the same wall-clock budget regardless of how expensive its
 * operations are. Data volume is capped by --max-data: the setup phase lays
 * down exactly that many bytes (split evenly across per-thread files) and every
 * write during the run targets an offset inside that region, so a long run
 * never grows the dataset.
 *
 * Latency is recorded per operation and reported as average and p99; I/O
 * operations additionally report IOPS and bandwidth. Results go to stdout and,
 * with --csv, to a machine-readable file for the comparison driver.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_ctp/util/logging.h>

#include "clio_cte/core/core_client.h"
#include "clio_cte/filesystem/filesystem_client.h"
#include <clio_cte/filesystem/filesystem_client.h>
#include "clio_runtime/clio_runtime.h"

namespace {

//===========================================================================
// Operation taxonomy
//===========================================================================

enum OpKind {
  kWrite4K = 0,
  kRead4K,
  kWrite1M,
  kRead1M,
  kStatSize,
  kReaddirSmall,
  kReaddirLarge,
  kRename,
  kNumOps
};

/** CLI/CSV name of each operation. */
const char *OpName(int op) {
  switch (op) {
    case kWrite4K: return "write_4k";
    case kRead4K: return "read_4k";
    case kWrite1M: return "write_1m";
    case kRead1M: return "read_1m";
    case kStatSize: return "stat_size";
    case kReaddirSmall: return "readdir_small";
    case kReaddirLarge: return "readdir_large";
    case kRename: return "rename";
    default: return "unknown";
  }
}

/** True for the operations that move bytes (bandwidth/IOPS are meaningful). */
bool IsDataOp(int op) {
  return op == kWrite4K || op == kRead4K || op == kWrite1M || op == kRead1M;
}

/** Transfer size of a data operation, 0 for metadata operations.
 *  `small` is the configured size of the small-I/O operations. */
clio::run::u64 OpIoSize(int op, clio::run::u64 small = 4096) {
  switch (op) {
    case kWrite4K:
    case kRead4K: return small;
    case kWrite1M:
    case kRead1M: return 1024 * 1024;
    default: return 0;
  }
}

//===========================================================================
// Configuration
//===========================================================================

constexpr size_t kSmallDirEntries = 10;
constexpr size_t kLargeDirEntries = 100;
constexpr clio::run::u64 kMaxIoSize = 1024 * 1024;

struct BenchConfig {
  double duration_s = 30.0;      // measured window
  double warmup_s = 2.0;         // discarded before measuring
  size_t threads = 8;
  clio::run::u64 max_data = 2ULL << 30;  // total bytes the dataset may occupy
  std::string root = "/clio_reliability_bench";
  std::string label = "combined";
  std::string csv_path;
  // Extra files created in a filler directory that no operation ever touches.
  // They exist only to grow the tag index, which is what isolates "cost of
  // listing THIS directory" from "cost of scanning the whole filesystem".
  size_t filler_files = 0;
  // Transfer size used by the "4k" operations. Exposed so the small-I/O size
  // can be moved across DefaultScheduler::kLargeIOThreshold (4096): at or
  // above it a request is routed as large I/O, below it it is not.
  clio::run::u64 small_io = 4096;
  // NOTE ON WHAT THIS MEASURES. Every operation goes through the descriptor
  // layer a real application reaches by interception -- open/read/write/
  // stat/readdir/rename -- and nothing here submits tasks directly. That is
  // the point: this is a
  // FILESYSTEM benchmark, so it has to pay what a filesystem caller pays,
  // including the adapter's descriptor lookup and the deferred-write path
  // (write(2) hands the bytes off and returns; fsync/close drains). Driving
  // the chimod client directly and Wait()ing on each task measures a
  // synchronous round trip that no application performs, and it is not
  // comparable to a buffered POSIX filesystem, which defers just the same.
  bool verbose = false;
  double mix[kNumOps] = {12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5};
};

/** Parse "write_4k=50,read_4k=50" into the mix table. Unlisted ops get 0. */
bool ParseMix(const std::string &spec, double (&mix)[kNumOps]) {
  for (int i = 0; i < kNumOps; ++i) {
    mix[i] = 0.0;
  }
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    size_t eq = item.find('=');
    if (eq == std::string::npos) {
      HLOG(kError, "ERROR: --mix entry '{}' is not <op>=<pct>", item);
      return false;
    }
    const std::string name = item.substr(0, eq);
    double pct = 0.0;
    try {
      pct = std::stod(item.substr(eq + 1));
    } catch (const std::exception &) {
      HLOG(kError, "ERROR: --mix entry '{}' has a non-numeric percentage",
           item);
      return false;
    }
    int found = -1;
    for (int i = 0; i < kNumOps; ++i) {
      if (name == OpName(i)) {
        found = i;
        break;
      }
    }
    if (found < 0) {
      HLOG(kError, "ERROR: --mix names unknown operation '{}'", name);
      return false;
    }
    mix[found] = pct;
  }
  double total = 0.0;
  for (int i = 0; i < kNumOps; ++i) {
    total += mix[i];
  }
  if (total <= 0.0) {
    HLOG(kError, "ERROR: --mix percentages sum to zero");
    return false;
  }
  return true;
}

void PrintUsage(const char *argv0) {
  HIPRINT("Usage: {} [options]", argv0);
  HIPRINT("Options:");
  HIPRINT("  --duration <sec>     Measured window (default: 30)");
  HIPRINT("  --warmup <sec>       Discarded warmup before measuring "
          "(default: 2)");
  HIPRINT("  --threads <N>        Client threads (default: 8)");
  HIPRINT("  --max-data <size>    Cap on dataset bytes, k/m/g suffix "
          "(default: 2g)");
  HIPRINT("  --mix <spec>         <op>=<pct>[,<op>=<pct>...] "
          "(default: even split)");
  HIPRINT("  --root <path>        Filesystem root for the dataset");
  HIPRINT("  --label <name>       Label recorded in the CSV "
          "(default: combined)");
  HIPRINT("  --csv <path>         Append per-operation results to this CSV");
  HIPRINT("  --filler <N>         Extra untouched files, to grow the tag "
          "index");
  HIPRINT("  --small-io <size>    Size of the small-I/O ops (default: 4k)");
  HIPRINT("  --verbose, -v        Per-thread detail");
  HIPRINT("  --help, -h           Show this help");
  HIPRINT("");
  HIPRINT("Operations: write_4k read_4k write_1m read_1m stat_size");
  HIPRINT("            readdir_small readdir_large rename");
}

bool ParseArgs(int argc, char **argv, BenchConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--duration" && i + 1 < argc) {
      cfg.duration_s = std::stod(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup_s = std::stod(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      cfg.threads = std::stoull(argv[++i]);
    } else if (arg == "--max-data" && i + 1 < argc) {
      cfg.max_data = ctp::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--mix" && i + 1 < argc) {
      if (!ParseMix(argv[++i], cfg.mix)) {
        return false;
      }
    } else if (arg == "--root" && i + 1 < argc) {
      cfg.root = argv[++i];
    } else if (arg == "--label" && i + 1 < argc) {
      cfg.label = argv[++i];
    } else if (arg == "--csv" && i + 1 < argc) {
      cfg.csv_path = argv[++i];
    } else if (arg == "--filler" && i + 1 < argc) {
      cfg.filler_files = std::stoull(argv[++i]);
    } else if (arg == "--small-io" && i + 1 < argc) {
      cfg.small_io = ctp::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      cfg.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else {
      HLOG(kError, "Unknown argument: {}", arg);
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (cfg.threads == 0) {
    HLOG(kError, "ERROR: --threads must be >= 1");
    return false;
  }
  // Every thread needs room for at least one 1 MiB transfer inside its slice
  // of the dataset, otherwise the large-I/O offsets have nowhere to land.
  if (cfg.max_data / cfg.threads < 2 * kMaxIoSize) {
    HLOG(kError,
         "ERROR: --max-data {} is too small for {} threads (need >= {} bytes)",
         cfg.max_data, cfg.threads, cfg.threads * 2 * kMaxIoSize);
    return false;
  }
  return true;
}

//===========================================================================
// Per-operation statistics
//===========================================================================

struct OpStats {
  std::vector<double> lat_us;   // one sample per completed operation
  clio::run::u64 bytes = 0;
  clio::run::u64 errors = 0;

  void Merge(const OpStats &other) {
    lat_us.insert(lat_us.end(), other.lat_us.begin(), other.lat_us.end());
    bytes += other.bytes;
    errors += other.errors;
  }
};

/** Left-pad `s` to `w` columns. ctp::Formatter supports plain {} only, so the
 *  report builds its own fixed-width columns. */
std::string Pad(const std::string &s, size_t w) {
  return s.size() >= w ? s : std::string(w - s.size(), ' ') + s;
}

/** Right-pad `s` to `w` columns (for the leading label column). */
std::string PadRight(const std::string &s, size_t w) {
  return s.size() >= w ? s : s + std::string(w - s.size(), ' ');
}

/** Format a double with `prec` decimals, right-aligned in `w` columns. */
std::string Num(double v, size_t w, int prec = 2) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return Pad(os.str(), w);
}

/** Percentile of an already-sorted sample vector. */
double Pctl(const std::vector<double> &sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const size_t idx = static_cast<size_t>(q * (sorted.size() - 1) + 0.5);
  return sorted[std::min(idx, sorted.size() - 1)];
}

//===========================================================================
// Filesystem helpers
//===========================================================================

std::string DataFilePath(const BenchConfig &cfg, size_t tid) {
  return cfg.root + "/data/f" + std::to_string(tid);
}

std::string RenamePath(const BenchConfig &cfg, size_t tid, int slot) {
  return cfg.root + "/renames/t" + std::to_string(tid) + "_" +
         std::to_string(slot);
}

/** mkdir that tolerates an existing directory. */
bool EnsureDir(clio::cte::filesystem::Client *fs, const std::string &path) {
  auto task = fs->AsyncMkdir(path);
  task.Wait();
  // A pre-existing directory is fine; only a hard failure with no directory
  // afterwards is fatal, which the subsequent Open/Readdir would surface.
  return true;
}

/**
 * Create `path` and fill it with `size` bytes, in `kMaxIoSize` chunks.
 * Pre-filling is what makes the read operations legitimate: every offset a
 * read can pick has real data behind it.
 */
bool CreateAndFill(clio::cte::filesystem::Client *fs, const std::string &path,
                   clio::run::u64 size, ctp::ipc::FullPtr<char> &buf) {
  auto open = fs->AsyncOpen(path, O_CREAT | O_RDWR, 0644);
  open.Wait();
  if (open->GetReturnCode() != 0) {
    HLOG(kError, "setup: open '{}' failed (rc={})", path,
         open->GetReturnCode());
    return false;
  }
  const clio::run::u64 handle = open->handle_;
  clio::run::u64 off = 0;
  while (off < size) {
    const clio::run::u64 n = std::min<clio::run::u64>(kMaxIoSize, size - off);
    auto w = fs->AsyncWrite(handle, off, n, buf.shm_.template Cast<void>());
    w.Wait();
    if (w->GetReturnCode() != 0 || w->bytes_written_ != n) {
      HLOG(kError, "setup: write '{}' at {} failed (rc={}, wrote={})", path,
           off, w->GetReturnCode(), w->bytes_written_);
      return false;
    }
    off += n;
  }
  auto close = fs->AsyncClose(handle);
  close.Wait();
  return true;
}

/** Populate a directory with `count` small files so Readdir has entries. */
bool PopulateDir(clio::cte::filesystem::Client *fs, const std::string &dir,
                 size_t count, ctp::ipc::FullPtr<char> &buf) {
  if (!EnsureDir(fs, dir)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    const std::string p = dir + "/e" + std::to_string(i);
    auto open = fs->AsyncOpen(p, O_CREAT | O_RDWR, 0644);
    open.Wait();
    if (open->GetReturnCode() != 0) {
      HLOG(kError, "setup: creating dir entry '{}' failed", p);
      return false;
    }
    // One 4 KiB block so the entry is a real file, not just an inode.
    auto w = fs->AsyncWrite(open->handle_, 0, 4096,
                            buf.shm_.template Cast<void>());
    w.Wait();
    auto close = fs->AsyncClose(open->handle_);
    close.Wait();
  }
  return true;
}

//===========================================================================
// Worker
//===========================================================================

/** The adapter intercepts a path only if it carries the clio:: marker. */
inline std::string Clio(const std::string &p) {
  return std::string(clio::cte::filesystem::kClioPrefix) + p;
}

/** Sub-intervals the measurement window is split into for stability
 *  reporting. 20 over a 15-20 s run is ~1 s each: long enough that a bucket
 *  is not dominated by scheduling jitter, short enough to expose a runtime
 *  that shifts gear partway through. */
static constexpr size_t kNumBuckets = 20;

struct ThreadContext {
  size_t tid = 0;
  int fd = -1;                      // adapter descriptor for this thread's file
  clio::run::u64 file_size = 0;     // bytes this thread owns in the dataset
  // Monotonic rename counter: each rename moves the file to a NEVER-BEFORE
  // -USED name. Ping-ponging between two fixed names would make every rename
  // reuse the same couple of index patterns, which flatters any pattern cache
  // and is not what a real rename workload looks like.
  clio::run::u64 rename_seq = 0;
  // Ordinary heap buffer: an application hands write(2) its OWN memory and
  // the adapter stages it. Handing the benchmark's shared-memory buffer
  // straight to the runtime would skip the copy every real caller pays.
  std::vector<char> buf;
  std::vector<std::string> dirents;  // readdir scratch, reused
  // Completions per sub-interval of the measurement window. Repeated runs of
  // this benchmark disagreed by up to 2.3x, and a point estimate cannot say
  // whether that is the workload oscillating DURING a run or whole runs
  // landing in different steady states -- which is the difference between "the
  // number is noisy" and "the number is meaningless". Bucketing separates them.
  std::vector<clio::run::u64> buckets;
  OpStats stats[kNumOps];
};

/**
 * Run the weighted operation mix until `deadline`. Samples taken before
 * `measure_start` are discarded so the steady state is what gets reported.
 */
void Worker(const BenchConfig &cfg, ThreadContext &ctx,
            const std::vector<double> &cdf,
            std::chrono::steady_clock::time_point measure_start,
            std::chrono::steady_clock::time_point deadline,
            std::atomic<bool> &abort_flag) {
  auto *cfs = CLIO_CFS_CLIENT;
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (ctx.tid * 1000003ULL));
  std::uniform_real_distribution<double> pick(0.0, 1.0);

  const std::string small_dir = Clio(cfg.root + "/small_dir");
  const std::string large_dir = Clio(cfg.root + "/large_dir");
  const std::string stat_path = Clio(DataFilePath(cfg, ctx.tid));

  while (!abort_flag.load(std::memory_order_relaxed)) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }

    // Weighted pick over the configured mix.
    const double r = pick(rng);
    int op = kNumOps - 1;
    for (int i = 0; i < kNumOps; ++i) {
      if (r <= cdf[i]) {
        op = i;
        break;
      }
    }

    const clio::run::u64 io_size = OpIoSize(op, cfg.small_io);
    clio::run::u64 offset = 0;
    if (io_size > 0) {
      // Random block-aligned offset inside this thread's slice. Staying inside
      // the slice is what enforces --max-data over an unbounded run.
      const clio::run::u64 blocks = ctx.file_size / io_size;
      offset = (blocks > 0 ? (rng() % blocks) : 0) * io_size;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    clio::run::u64 moved = 0;

    switch (op) {
      case kWrite4K:
      case kWrite1M: {
        ssize_t n = cfs->PwriteFd(ctx.fd, ctx.buf.data(),
                                static_cast<size_t>(io_size),
                                static_cast<off_t>(offset));
        ok = (n == static_cast<ssize_t>(io_size));
        moved = ok ? io_size : 0;
        break;
      }
      case kRead4K:
      case kRead1M: {
        ssize_t n = cfs->PreadFd(ctx.fd, ctx.buf.data(),
                               static_cast<size_t>(io_size),
                               static_cast<off_t>(offset));
        ok = (n >= 0);
        moved = ok ? static_cast<clio::run::u64>(n) : 0;
        break;
      }
      case kStatSize: {
        struct stat st;
        ok = (cfs->StatPath(stat_path, &st) == 0);
        break;
      }
      case kReaddirSmall: {
        ok = (cfs->ReaddirPath(small_dir, &ctx.dirents) == 0);
        break;
      }
      case kReaddirLarge: {
        ok = (cfs->ReaddirPath(large_dir, &ctx.dirents) == 0);
        break;
      }
      case kRename: {
        // Walk the file to a fresh name each time, within this thread's own
        // namespace so no thread ever renames a path another thread owns.
        const std::string src =
            Clio(RenamePath(cfg, ctx.tid, static_cast<int>(ctx.rename_seq)));
        const std::string dst = Clio(
            RenamePath(cfg, ctx.tid, static_cast<int>(ctx.rename_seq + 1)));
        ok = (cfs->RenamePath(src, dst) == 0);
        if (ok) {
          ++ctx.rename_seq;
        }
        break;
      }
      default:
        ok = false;
        break;
    }

    const auto t1 = std::chrono::steady_clock::now();
    if (t1 < measure_start) {
      continue;  // warmup
    }
    if (!ctx.buckets.empty()) {
      const double frac =
          std::chrono::duration<double>(t1 - measure_start).count() /
          cfg.duration_s;
      long idx = static_cast<long>(frac * static_cast<double>(kNumBuckets));
      if (idx < 0) idx = 0;
      if (idx >= static_cast<long>(kNumBuckets)) idx = kNumBuckets - 1;
      ctx.buckets[static_cast<size_t>(idx)]++;
    }
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (ok) {
      ctx.stats[op].lat_us.push_back(us);
      ctx.stats[op].bytes += moved;
    } else {
      ctx.stats[op].errors++;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  if (!ParseArgs(argc, argv, cfg)) {
    return 1;
  }

  HIPRINT("=== Clio-FS Reliability Benchmark ===");
  HIPRINT("Label: {}", cfg.label);
  HIPRINT("Threads: {}  Duration: {} s  Warmup: {} s", cfg.threads,
          cfg.duration_s, cfg.warmup_s);
  HIPRINT("Max dataset: {} bytes ({} MB)", cfg.max_data,
          cfg.max_data / (1024 * 1024));
  {
    std::ostringstream mixdesc;
    for (int i = 0; i < kNumOps; ++i) {
      if (cfg.mix[i] > 0.0) {
        mixdesc << OpName(i) << "=" << cfg.mix[i] << " ";
      }
    }
    HIPRINT("Mix: {}", mixdesc.str());
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "ERROR: Failed to initialize Clio client");
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "ERROR: Failed to initialize CTE client");
    return 1;
  }
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    HLOG(kError, "ERROR: Failed to bind the clio-fs pool (is it composed?)");
    return 1;
  }
  auto *fs = CLIO_CFS_CLIENT;

  //=========================================================================
  // Setup: lay down exactly --max-data bytes plus the metadata fixtures.
  //=========================================================================
  const clio::run::u64 per_thread = cfg.max_data / cfg.threads;
  HIPRINT("\nSetup: {} data files x {} MB ...", cfg.threads,
          per_thread / (1024 * 1024));

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> setup_buf = ipc->AllocateBuffer(kMaxIoSize);
  if (setup_buf.IsNull()) {
    HLOG(kError, "ERROR: could not allocate the {} byte setup buffer",
         kMaxIoSize);
    return 1;
  }
  std::memset(setup_buf.ptr_, 0xA5, kMaxIoSize);

  if (!EnsureDir(fs, cfg.root) || !EnsureDir(fs, cfg.root + "/data") ||
      !EnsureDir(fs, cfg.root + "/renames")) {
    HLOG(kError, "ERROR: could not create the benchmark directories");
    return 1;
  }
  for (size_t t = 0; t < cfg.threads; ++t) {
    if (!CreateAndFill(fs, DataFilePath(cfg, t), per_thread, setup_buf)) {
      return 1;
    }
    // Slot 0 exists, slot 1 does not: the rename op flips between them. A
    // previous run may have left the pair in either state, so clear both
    // before recreating slot 0 — otherwise the first rename of this run would
    // target an existing destination.
    // A previous run may have left the walk at an arbitrary sequence number;
    // clear the first few names so this run starts from a known slot 0.
    for (int slot = 0; slot < 4; ++slot) {
      auto unlink = fs->AsyncUnlink(RenamePath(cfg, t, slot));
      unlink.Wait();
    }
    auto open = fs->AsyncOpen(RenamePath(cfg, t, 0), O_CREAT | O_RDWR, 0644);
    open.Wait();
    if (open->GetReturnCode() != 0) {
      HLOG(kError, "ERROR: could not create the rename fixture for thread {}",
           t);
      return 1;
    }
    auto close = fs->AsyncClose(open->handle_);
    close.Wait();
  }
  if (!PopulateDir(fs, cfg.root + "/small_dir", kSmallDirEntries, setup_buf) ||
      !PopulateDir(fs, cfg.root + "/large_dir", kLargeDirEntries, setup_buf)) {
    return 1;
  }
  if (cfg.filler_files > 0) {
    const std::string filler_dir = cfg.root + "/filler";
    if (!EnsureDir(fs, filler_dir)) {
      return 1;
    }
    HIPRINT("Creating {} filler files (index pressure only) ...",
            cfg.filler_files);
    for (size_t i = 0; i < cfg.filler_files; ++i) {
      const std::string p = filler_dir + "/f" + std::to_string(i);
      auto open = fs->AsyncOpen(p, O_CREAT | O_RDWR, 0644);
      open.Wait();
      if (open->GetReturnCode() != 0) {
        HLOG(kError, "ERROR: could not create filler file '{}'", p);
        return 1;
      }
      auto close = fs->AsyncClose(open->handle_);
      close.Wait();
    }
  }
  ipc->FreeBuffer(setup_buf);
  HIPRINT("Setup complete ({} bytes materialized)", cfg.max_data);

  //=========================================================================
  // Run
  //=========================================================================
  double total_pct = 0.0;
  for (int i = 0; i < kNumOps; ++i) {
    total_pct += cfg.mix[i];
  }
  std::vector<double> cdf(kNumOps, 0.0);
  double acc = 0.0;
  for (int i = 0; i < kNumOps; ++i) {
    acc += cfg.mix[i] / total_pct;
    cdf[i] = acc;
  }

  std::vector<ThreadContext> ctxs(cfg.threads);
  for (size_t t = 0; t < cfg.threads; ++t) {
    ctxs[t].tid = t;
    ctxs[t].file_size = per_thread;
    ctxs[t].buf.assign(kMaxIoSize, 0x5A);
    ctxs[t].buckets.assign(kNumBuckets, 0);
    ctxs[t].fd = CLIO_CFS_CLIENT->OpenFd(Clio(DataFilePath(cfg, t)), O_RDWR, 0644);
    if (ctxs[t].fd < 0) {
      HLOG(kError, "ERROR: thread {} could not open its data file", t);
      return 1;
    }
  }

  std::atomic<bool> abort_flag{false};
  const auto start = std::chrono::steady_clock::now();
  const auto measure_start =
      start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(cfg.warmup_s));
  const auto deadline =
      measure_start +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(cfg.duration_s));

  HIPRINT("\nRunning ...");
  std::vector<std::thread> workers;
  workers.reserve(cfg.threads);
  for (size_t t = 0; t < cfg.threads; ++t) {
    workers.emplace_back(Worker, std::cref(cfg), std::ref(ctxs[t]),
                         std::cref(cdf), measure_start, deadline,
                         std::ref(abort_flag));
  }
  for (auto &w : workers) {
    w.join();
  }
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    measure_start)
          .count();

  for (size_t t = 0; t < cfg.threads; ++t) {
    // close(2) drains this file's deferred writes — the cost of the write
    // tail belongs to the run, not to whatever runs next.
    CLIO_CFS_CLIENT->CloseFd(ctxs[t].fd);
  }

  //=========================================================================
  // Report
  //=========================================================================
  OpStats agg[kNumOps];
  for (size_t t = 0; t < cfg.threads; ++t) {
    for (int i = 0; i < kNumOps; ++i) {
      agg[i].Merge(ctxs[t].stats[i]);
    }
  }

  // Whether reads were served without IPC is the single biggest determinant
  // of read latency here, and it is invisible in the timings alone -- a
  // disabled fast path just looks like a slow runtime.
  {
    const auto hits =
        clio::cte::filesystem::Client::ShmReadHits().load();
    const auto misses =
        clio::cte::filesystem::Client::ShmReadMisses().load();
    const auto total = hits + misses;
    HIPRINT("\nZero-IPC read fast path: {} / {} reads ({}%)", hits, total,
            total ? (100.0 * static_cast<double>(hits) /
                     static_cast<double>(total))
                  : 0.0);
    const auto shits = clio::cte::filesystem::Client::ShmStatHits().load();
    const auto smiss = clio::cte::filesystem::Client::ShmStatMisses().load();
    const auto stotal = shits + smiss;
    HIPRINT("Zero-IPC stat fast path: {} / {} stats ({}%)", shits, stotal,
            stotal ? (100.0 * static_cast<double>(shits) /
                      static_cast<double>(stotal))
                   : 0.0);
  }

  {
    std::vector<double> per_bucket(kNumBuckets, 0.0);
    for (size_t t = 0; t < cfg.threads; ++t) {
      for (size_t b = 0; b < kNumBuckets && b < ctxs[t].buckets.size(); ++b) {
        per_bucket[b] += static_cast<double>(ctxs[t].buckets[b]);
      }
    }
    const double bucket_s = cfg.duration_s / static_cast<double>(kNumBuckets);
    for (double &v : per_bucket) v /= bucket_s;
    std::vector<double> sorted = per_bucket;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double v : per_bucket) sum += v;
    const double mean = sum / static_cast<double>(kNumBuckets);
    double var = 0.0;
    for (double v : per_bucket) var += (v - mean) * (v - mean);
    var /= static_cast<double>(kNumBuckets);
    const double sd = std::sqrt(var);
    HIPRINT("\nIn-run stability ({} x {} s buckets, ops/s):", kNumBuckets,
            bucket_s);
    HIPRINT("  min={}  median={}  max={}  mean={}  CV={}%", sorted.front(),
            sorted[kNumBuckets / 2], sorted.back(), mean,
            mean > 0 ? 100.0 * sd / mean : 0.0);
  }

  HIPRINT("\n=== Results ({}) — {} s measured ===", cfg.label, elapsed_s);
  HIPRINT("{}{}{}{}{}{}", PadRight("operation", 16), Pad("count", 10),
          Pad("ops/sec", 12), Pad("avg_us", 11), Pad("p99_us", 11),
          Pad("MB/s", 11));

  std::ofstream csv;
  if (!cfg.csv_path.empty()) {
    const bool exists = std::ifstream(cfg.csv_path).good();
    csv.open(cfg.csv_path, std::ios::app);
    if (csv.is_open() && !exists) {
      csv << "label,operation,count,errors,ops_per_sec,avg_us,p50_us,p99_us,"
             "mb_per_sec,bytes\n";
    }
  }

  for (int i = 0; i < kNumOps; ++i) {
    if (agg[i].lat_us.empty() && agg[i].errors == 0) {
      continue;
    }
    std::sort(agg[i].lat_us.begin(), agg[i].lat_us.end());
    const double count = static_cast<double>(agg[i].lat_us.size());
    double sum = 0.0;
    for (double v : agg[i].lat_us) {
      sum += v;
    }
    const double avg = count > 0 ? sum / count : 0.0;
    const double p50 = Pctl(agg[i].lat_us, 0.50);
    const double p99 = Pctl(agg[i].lat_us, 0.99);
    const double ops_s = elapsed_s > 0 ? count / elapsed_s : 0.0;
    const double mb_s =
        elapsed_s > 0
            ? (static_cast<double>(agg[i].bytes) / elapsed_s) / (1024 * 1024)
            : 0.0;

    HIPRINT("{}{}{}{}{}{}", PadRight(OpName(i), 16),
            Pad(std::to_string(static_cast<clio::run::u64>(count)), 10),
            Num(ops_s, 12), Num(avg, 11), Num(p99, 11),
            IsDataOp(i) ? Num(mb_s, 11) : Pad("-", 11));
    if (agg[i].errors > 0) {
      HLOG(kWarning, "  {} had {} failed operations", OpName(i),
           agg[i].errors);
    }
    if (csv.is_open()) {
      csv << cfg.label << "," << OpName(i) << ","
          << static_cast<clio::run::u64>(count) << "," << agg[i].errors << ","
          << ops_s << "," << avg << "," << p50 << "," << p99 << "," << mb_s
          << "," << agg[i].bytes << "\n";
    }
  }
  if (csv.is_open()) {
    csv.close();
    HIPRINT("\nCSV appended: {}", cfg.csv_path);
  }

  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}
