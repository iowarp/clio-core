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
 * vector.Copy: synchronous vs lazy (copy-on-write), on a two-tier CTE of
 * host DRAM over a file.
 *
 * Each rep, per mode:
 *   1. seed a fresh source vector with pattern A              (untimed)
 *   2. Copy(name, sync)                                       -> copy_ms
 *   3. overwrite the source with pattern B, as an application
 *      that keeps running after its checkpoint would           (untimed)
 *   4. read the whole copy back and compare against A          -> read_ms
 *
 * Lazy defers the bytes to the first touch of each page, so its Copy is
 * cheap and its read pays the materialisation -- from the source AS IT IS
 * THEN, i.e. pattern B. Sync pays everything inside Copy and its read sees
 * A. The `bad` column is that difference; --no-overwrite removes it and
 * leaves a pure cost comparison.
 *
 * The vector is CTE-only (no device views), so no kernel is involved: this
 * isolates Copy itself from any workload.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Vector is host-only; the device compilation pass sees nothing here.
#if !CTP_IS_DEVICE_PASS

namespace {

namespace gv = clio::cte::gpu_vector;
using u32 = clio::run::u32;
using u64 = clio::run::u64;

constexpr u32 kSeedA = 0x1234u;  /**< pattern the copy must hold */
constexpr u32 kSeedB = 0xBEEFu;  /**< pattern written after the copy */

/** Command-line options. */
struct Opts {
  u64 data_mb = 256;       /**< source vector size */
  u64 page_kb = 1024;      /**< vector page size */
  u64 ram_mb = 384;        /**< DRAM tier capacity */
  u64 file_mb = 4096;      /**< file tier capacity */
  std::string file_path = "/tmp/gv_copy_sync_tier.dat";
  int reps = 3;            /**< measured reps per mode */
  bool overwrite = true;   /**< step 3 on/off */
};

/** Per-rep measurements. */
struct Sample {
  double copy_ms = 0, read_ms = 0;
  u64 bad = 0;                     /**< elements of the copy != pattern A */
  u64 ram_mb = 0, file_mb = 0;     /**< tier bytes written by the Copy call */
};

/** Milliseconds since an arbitrary epoch. */
double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** Deterministic element pattern for index `i` under `seed`. */
u32 Pat(u64 i, u32 seed) {
  return static_cast<u32>(i * 2654435761ull) ^ seed;
}

/**
 * Parse argv into Opts.
 * @return false on --help (usage printed)
 */
bool ParseArgs(int argc, char **argv, Opts *o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--data-mb") o->data_mb = next();
    else if (a == "--page-kb") o->page_kb = next();
    else if (a == "--ram-mb") o->ram_mb = next();
    else if (a == "--file-mb") o->file_mb = next();
    else if (a == "--file-path" && i + 1 < argc) o->file_path = argv[++i];
    else if (a == "--reps") o->reps = static_cast<int>(next());
    else if (a == "--no-overwrite") o->overwrite = false;
    else if (a == "--help") {
      std::printf("usage: %s [--data-mb N] [--page-kb N] [--ram-mb N] "
                  "[--file-mb N] [--file-path P] [--reps N] "
                  "[--no-overwrite]\n", argv[0]);
      return false;
    }
  }
  return true;
}

/**
 * Write the two-tier runtime config and point CLIO_SERVER_CONF at it.
 * DRAM scores above the file tier, both at or below the vector's blob score,
 * so DRAM fills first and the file tier takes the spill.
 */
void WriteConfig(const Opts &o) {
  if (std::getenv("CLIO_SERVER_CONF") != nullptr) return;
  std::ofstream cfg("gv_copy_sync_bench.yaml");
  cfg << "networking:\n  port: 9451\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
      << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
      << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
      << "      - path: \"ram::gv_copy_sync_dram\"\n"
      << "        bdev_type: \"ram\"\n"
      << "        capacity_limit: \"" << o.ram_mb << "MB\"\n"
      << "        score: 0.5\n"
      << "      - path: \"" << o.file_path << "\"\n"
      << "        bdev_type: \"file\"\n"
      << "        persistence_level: \"temporary\"\n"
      << "        capacity_limit: \"" << o.file_mb << "MB\"\n"
      << "        score: 0.0\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  cfg.close();
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_copy_sync_bench.yaml", 1);
}

/** Bytes remaining on tier bdev pool (512 + idx, 1); tiers follow config order. */
u64 TierRemaining(u32 idx) {
  clio::run::bdev::Client t(clio::run::PoolId(512 + idx, 1));
  auto f = t.AsyncGetStats();
  f.Wait();
  return f->remaining_size_;
}

/** Fill `v` with Pat(i, seed). */
void Fill(gv::Vector<u32> &v, u64 n, u32 seed) {
  std::vector<u32> host(n);
  for (u64 i = 0; i < n; ++i) host[i] = Pat(i, seed);
  v.Preload(host.data(), n);
}

/** Delete a tag and its blobs so the next rep starts from the same tiers. */
void DropTag(const std::string &name) {
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  auto f = cte.AsyncDelTag(name);
  f.Wait();
}

/**
 * One rep of one mode (see the file comment for the steps).
 * @param o    options
 * @param sync the Copy flag under test
 * @param tag  unique suffix for this rep's tags
 */
Sample RunOne(const Opts &o, bool sync, const std::string &tag) {
  const u64 page_bytes = o.page_kb * 1024;
  const u64 n = o.data_mb * 1024 * 1024 / sizeof(u32);
  const std::string src_name = "gv_cs_src_" + tag;
  const std::string dst_name = "gv_cs_dst_" + tag;
  Sample s;
  {
    gv::Vector<u32> src(src_name, {}, page_bytes, /*nblocks=*/1,
                        /*set_size=*/1, n);
    Fill(src, n, kSeedA);

    const u64 ram0 = TierRemaining(0), file0 = TierRemaining(1);
    const double t0 = NowMs();
    auto copy = src.Copy(dst_name, sync);
    s.copy_ms = NowMs() - t0;
    s.ram_mb = (ram0 - std::min(ram0, TierRemaining(0))) >> 20;
    s.file_mb = (file0 - std::min(file0, TierRemaining(1))) >> 20;

    if (o.overwrite) Fill(src, n, kSeedB);

    std::vector<u32> back(n, 0);
    const double t1 = NowMs();
    copy->Download(back.data(), n);
    s.read_ms = NowMs() - t1;
    for (u64 i = 0; i < n; ++i) s.bad += (back[i] != Pat(i, kSeedA));
  }
  DropTag(dst_name);
  DropTag(src_name);
  return s;
}

/** Print one result row; timings in ms. */
void Report(const char *mode, int rep, const Opts &o, const Sample &s) {
  const double gb = static_cast<double>(o.data_mb) / 1024.0;
  std::printf("%-5s rep=%d copy_ms=%9.2f read_ms=%9.2f total_ms=%9.2f "
              "copy_GBps=%6.2f copy_wrote_dram=%lluMiB copy_wrote_file=%lluMiB "
              "bad=%llu\n",
              mode, rep, s.copy_ms, s.read_ms, s.copy_ms + s.read_ms,
              s.copy_ms > 0 ? gb / (s.copy_ms / 1000.0) : 0.0,
              (unsigned long long)s.ram_mb, (unsigned long long)s.file_mb,
              (unsigned long long)s.bad);
}

}  // namespace

int main(int argc, char **argv) {
  Opts o;
  if (!ParseArgs(argc, argv, &o)) return 0;
  std::printf("copy_sync_bench: data=%lluMiB page=%lluKiB dram=%lluMiB "
              "file=%lluMiB (%s) reps=%d overwrite=%d\n",
              (unsigned long long)o.data_mb, (unsigned long long)o.page_kb,
              (unsigned long long)o.ram_mb, (unsigned long long)o.file_mb,
              o.file_path.c_str(), o.reps, o.overwrite ? 1 : 0);
  WriteConfig(o);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "COPY_SYNC ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "COPY_SYNC ERROR: cte client init failed\n");
    return 1;
  }
  // One untimed warm-up per mode: pool creation (the checkpoint chimod) and
  // first-touch allocator costs are not what this measures.
  RunOne(o, false, "warm_lazy");
  RunOne(o, true, "warm_sync");

  bool ok = true;
  for (int r = 0; r < o.reps; ++r) {
    for (bool sync : {false, true}) {
      const std::string tag =
          std::string(sync ? "sync_" : "lazy_") + std::to_string(r);
      const Sample s = RunOne(o, sync, tag);
      Report(sync ? "sync" : "lazy", r, o, s);
      // The sync copy must be the snapshot at Copy() no matter what.
      if (sync && s.bad != 0) ok = false;
    }
  }
  std::printf("COPY_SYNC GATE: %s (sync copy holds the pre-overwrite bytes)\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS
