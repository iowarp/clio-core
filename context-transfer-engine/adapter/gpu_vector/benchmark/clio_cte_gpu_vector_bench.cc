/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * gpu_vector performance benchmark.
 *
 * Drives a Vector<clio::run::u32> through a write/flush/read cycle while
 * exposing every interesting parameter on the command line:
 *
 *   --blocks N             Number of cache blocks (default 32)
 *   --pages-per-block P    Cache pages per block (default 2)
 *   --page-size B          Page size in bytes (default 1048576 = 1 MiB)
 *   --ratio R              Total data = R × cache size; controls how
 *                          much oversubscription each block sees
 *                          (default 1.0 → no eviction; > 1 forces
 *                          eviction; defaults to 1.0).
 *   --total-bytes B        Override total data size in bytes; if set
 *                          this takes precedence over --ratio.
 *   --cache-period-us N    Period of the periodic CacheMgmtKernel
 *                          (0 = disabled; default 0).
 *   --iters N              Repeat the write/flush/read cycle N times
 *                          (default 1).
 *   --no-read              Skip the read phase (only measure write +
 *                          flush). Useful when you want a clean write-
 *                          path number without faulting overhead.
 *   --bdev-capacity-mib N  kRam bdev target capacity in MiB
 *                          (default = 4× total_bytes, min 64).
 *   --gpu-id N             GPU index (default 0).
 *
 * Output: per-iteration timings (write/flush/read in ms) and bandwidth
 * (MiB/s), plus a final summary with min/median/max across iters. Per-
 * task latency is computed as wall_time / N_tasks where N_tasks is the
 * expected count given the geometry.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

struct BenchOpts {
  clio::run::u32 nblocks = 32;
  clio::run::u32 pages_per_block = 2;       // HBM tier slots per block
  clio::run::u32 host_pages_per_block = 0;  // DRAM tier slots; 0 = legacy mode
  clio::run::u32 manager_threads_per_block = 32;
  clio::run::u64 page_size = 1ULL << 20;  // 1 MiB
  double ratio = 1.0;
  clio::run::u64 total_bytes = 0;          // 0 = derive from ratio
  clio::run::u64 per_block_bytes = 0;      // 0 = derive from ratio / total_bytes
  clio::run::u32 cache_period_us = 0;
  clio::run::u32 iters = 1;
  bool do_read = true;
  clio::run::u32 gpu_id = 0;
  clio::run::u64 bdev_capacity_mib = 0;    // 0 = auto
  bool allow_cold_miss_fault = false;  // default: async-only
};

void PrintUsage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "  --blocks N             Cache blocks (default 32)\n"
               "  --pages-per-block P    HBM cache slots per block (default 2)\n"
               "  --host-pages-per-block P  DRAM cache slots per block.\n"
               "                            0 = legacy single-tier mode (default).\n"
               "                            >0 promotes to kAsync mode (CacheManagerKernel).\n"
               "  --manager-threads N    Threads/block for CacheManagerKernel (default 32, multiple of 32)\n"
               "  --page-size BYTES      Page size in bytes (default 1048576)\n"
               "  --per-block-bytes B    Bytes each block writes/reads (overrides --ratio\n"
               "                         and --total-bytes; total_bytes = blocks * B).\n"
               "                         Use this to amortize kernel-launch overhead\n"
               "                         (e.g. --per-block-bytes 16777216 = 16 MiB/block).\n"
               "  --ratio R              total_bytes = R * cache_bytes (default 1.0)\n"
               "  --total-bytes BYTES    Override total bytes (takes precedence over --ratio)\n"
               "  --cache-period-us N    Cache-thread tick period in microseconds.\n"
               "                            0=off (default). Doubles as the initial\n"
               "                            __nanosleep duration for the persistent\n"
               "                            CacheManagerKernel; blocks back off\n"
               "                            exponentially up to 16x when idle, and\n"
               "                            self-terminate after 500ms idle.\n"
               "  --iters N              Repeat write/flush/read cycle N times (default 1)\n"
               "  --no-read              Skip read+verify phase\n"
               "  --bdev-capacity-mib N  kRam bdev capacity (default = 4x total_bytes, min 64)\n"
               "  --gpu-id N             GPU index (default 0)\n"
               "  --help                 Show this message\n",
               prog);
}

bool ParseOpts(int argc, char *argv[], BenchOpts &opts) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); std::exit(0); }
    else if (a == "--blocks") opts.nblocks = std::atoi(next("--blocks"));
    else if (a == "--pages-per-block")
      opts.pages_per_block = std::atoi(next("--pages-per-block"));
    else if (a == "--host-pages-per-block")
      opts.host_pages_per_block =
          std::atoi(next("--host-pages-per-block"));
    else if (a == "--manager-threads")
      opts.manager_threads_per_block =
          std::atoi(next("--manager-threads"));
    else if (a == "--page-size")
      opts.page_size = std::strtoull(next("--page-size"), nullptr, 10);
    else if (a == "--ratio")
      opts.ratio = std::atof(next("--ratio"));
    else if (a == "--total-bytes")
      opts.total_bytes = std::strtoull(next("--total-bytes"), nullptr, 10);
    else if (a == "--per-block-bytes")
      opts.per_block_bytes =
          std::strtoull(next("--per-block-bytes"), nullptr, 10);
    else if (a == "--cache-period-us")
      opts.cache_period_us = std::atoi(next("--cache-period-us"));
    else if (a == "--iters")
      opts.iters = std::atoi(next("--iters"));
    else if (a == "--no-read") opts.do_read = false;
    else if (a == "--bdev-capacity-mib")
      opts.bdev_capacity_mib =
          std::strtoull(next("--bdev-capacity-mib"), nullptr, 10);
    else if (a == "--gpu-id") opts.gpu_id = std::atoi(next("--gpu-id"));
    else if (a == "--allow-cold-miss-fault") opts.allow_cold_miss_fault = true;
    else {
      std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

/** One-shot CLIO Runtime + CTE pool + kRam bdev target setup. Host-only:
 *  AsyncCreate / AsyncRegisterTarget aren't visible in the GPU device
 *  pass (nvcc parses both passes for __global__-bearing TUs). */
#if !CTP_IS_DEVICE_PASS
void EnsureInit(const BenchOpts &opts, clio::run::u64 bdev_capacity_bytes) {
  std::fprintf(stderr, "[INIT] Starting Clio server\n");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[INIT] CLIO_INIT failed\n");
    std::exit(2);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[INIT] CLIO_CTE_CLIENT_INIT failed\n");
    std::exit(2);
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] CTE pool create failed rc=%u\n",
                 create_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);

  clio::run::PoolId bdev_pool_id(951, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("gpu_vector_bench_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, bdev_capacity_bytes);
  bdev_create.Wait();
  if (bdev_create->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] bdev create failed rc=%u\n",
                 bdev_create->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gpu_vector_bench_ram", clio::run::bdev::BdevType::kRam,
      bdev_capacity_bytes, clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] RegisterTarget failed rc=%u\n",
                 reg_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
  (void)opts;
}
#endif  // !CTP_IS_DEVICE_PASS

/** Compute the number of expected PutBlobs for the given geometry. */
clio::run::u64 ExpectedPuts(clio::run::u32 nblocks, clio::run::u32 pages_per_block,
                       clio::run::u64 per_block_pages) {
  if (per_block_pages <= pages_per_block) {
    // No eviction; only the final flush of dirty pages contributes.
    return static_cast<clio::run::u64>(nblocks) * per_block_pages;
  }
  // 1 eviction per page beyond the cache; evictions push 1 put per
  // dirty page, plus pages_per_block final-flush puts at FlushAllSync.
  clio::run::u64 evict_per_block = per_block_pages - pages_per_block;
  return static_cast<clio::run::u64>(nblocks) *
         (evict_per_block + pages_per_block);
}

/** Compute the number of expected GetBlobs (sequential read pattern):
 *  one fault per page when the stripe exceeds the cache. */
clio::run::u64 ExpectedGets(clio::run::u32 nblocks, clio::run::u32 pages_per_block,
                       clio::run::u64 per_block_pages) {
  if (per_block_pages <= pages_per_block) return 0;  // hits cache
  return static_cast<clio::run::u64>(nblocks) * per_block_pages;
}

}  // namespace

/** Sequential write of v[i] = i across each block's stripe.
 *  Uses dev::vector::write_range so the page lookup + dirty-range
 *  bookkeeping happens once per page; the inner loop is a tight
 *  stride-1 store. */
__global__ void BenchWriteKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceView<clio::run::u32> view,
                                  clio::run::u64 per_block) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<clio::run::u32> v(view, g_ipc_manager_ptr);
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * per_block;
  clio::run::u64 hi = lo + per_block;
  v.write_range(lo, hi, [] (clio::run::u64 i) {
    return static_cast<clio::run::u32>(i);
  });
  (void)g_ipc_manager;
}

/** Sequential read into a result buffer (host-pinned).
 *  Uses dev::vector::read_range so we resolve the page once per page.
 *  Accumulates into a per-thread register and atomicAdd_system's a single
 *  reduction per block back to result[0]. This matches the "read-and-
 *  consume" usage pattern far more honestly than writing every element
 *  back to host — that pattern is bottlenecked by PCIe transaction
 *  count (32-lane × 4B coalesced = 128B per burst), not bandwidth. */
__global__ void BenchReadKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<clio::run::u32> view,
                                 clio::run::u32 *result, clio::run::u64 per_block) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<clio::run::u32> v(view, g_ipc_manager_ptr);
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * per_block;
  clio::run::u64 hi = lo + per_block;
  clio::run::u32 lane_sum = 0;
  v.read_range(lo, hi, [&lane_sum] (clio::run::u64 /*i*/, clio::run::u32 val) {
    lane_sum ^= val;
  });
  // Warp-reduce + one atomicAdd_system per block.
  for (int o = 16; o > 0; o >>= 1) {
    lane_sum ^= __shfl_xor_sync(0xffffffff, lane_sum, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd_system(reinterpret_cast<unsigned int *>(result), lane_sum);
  }
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

/** Format a byte count as a human-friendly MiB / GiB string. */
std::string FmtBytes(clio::run::u64 b) {
  char buf[64];
  if (b >= (1ULL << 30)) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", b / static_cast<double>(1ULL << 30));
  } else if (b >= (1ULL << 20)) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB", b / static_cast<double>(1ULL << 20));
  } else if (b >= (1ULL << 10)) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB", b / static_cast<double>(1ULL << 10));
  } else {
    std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
  }
  return std::string(buf);
}

int main(int argc, char *argv[]) {
  BenchOpts opts;
  if (!ParseOpts(argc, argv, opts)) return 2;
  if (opts.nblocks == 0 || opts.pages_per_block == 0 ||
      opts.page_size == 0) {
    std::fprintf(stderr, "blocks/pages/page_size must be > 0\n");
    return 2;
  }
  if (opts.page_size % sizeof(clio::run::u32) != 0) {
    std::fprintf(stderr, "page_size must be a multiple of sizeof(u32)\n");
    return 2;
  }

  clio::run::u64 cache_bytes = static_cast<clio::run::u64>(opts.nblocks) *
                         opts.pages_per_block * opts.page_size;
  // Resolve in priority: --per-block-bytes > --total-bytes > --ratio.
  clio::run::u64 per_block_bytes;
  if (opts.per_block_bytes > 0) {
    per_block_bytes = opts.per_block_bytes;
  } else if (opts.total_bytes > 0) {
    per_block_bytes = opts.total_bytes / opts.nblocks;
  } else {
    clio::run::u64 t = static_cast<clio::run::u64>(opts.ratio * cache_bytes);
    if (t < opts.page_size) t = opts.page_size;
    per_block_bytes = t / opts.nblocks;
  }
  // Round per_block_bytes up to a whole number of pages so the kernel
  // doesn't have to handle a partial-page tail.
  if (per_block_bytes < opts.page_size) per_block_bytes = opts.page_size;
  if (per_block_bytes % opts.page_size != 0) {
    per_block_bytes = ((per_block_bytes + opts.page_size - 1) /
                       opts.page_size) * opts.page_size;
  }
  opts.total_bytes = per_block_bytes * opts.nblocks;
  clio::run::u64 per_block_pages = per_block_bytes / opts.page_size;
  clio::run::u64 total_elems = opts.total_bytes / sizeof(clio::run::u32);
  clio::run::u64 per_block_elems = per_block_bytes / sizeof(clio::run::u32);

  clio::run::u64 bdev_capacity_bytes;
  if (opts.bdev_capacity_mib > 0) {
    bdev_capacity_bytes = opts.bdev_capacity_mib * (1ULL << 20);
  } else {
    bdev_capacity_bytes =
        std::max<clio::run::u64>(64ULL << 20, opts.total_bytes * 4);
  }

  clio::run::u64 expected_puts =
      ExpectedPuts(opts.nblocks, opts.pages_per_block, per_block_pages);
  clio::run::u64 expected_gets =
      ExpectedGets(opts.nblocks, opts.pages_per_block, per_block_pages);

  std::fprintf(stderr,
               "[BENCH] blocks=%u pages_per_block=%u host_pages=%u "
               "mode=%s page_size=%s\n"
               "[BENCH] cache=%s total=%s ratio=%.2fx\n"
               "[BENCH] per_block=%llu pages (%s)\n"
               "[BENCH] expected_puts=%llu expected_gets=%llu\n"
               "[BENCH] cache_period_us=%u manager_threads=%u iters=%u "
               "read=%s\n",
               opts.nblocks, opts.pages_per_block,
               opts.host_pages_per_block,
               opts.host_pages_per_block > 0 ? "kAsync" : "kLegacy",
               FmtBytes(opts.page_size).c_str(),
               FmtBytes(cache_bytes).c_str(),
               FmtBytes(opts.total_bytes).c_str(),
               opts.total_bytes / static_cast<double>(cache_bytes),
               (unsigned long long)per_block_pages,
               FmtBytes(per_block_bytes).c_str(),
               (unsigned long long)expected_puts,
               (unsigned long long)expected_gets,
               opts.cache_period_us, opts.manager_threads_per_block,
               opts.iters,
               opts.do_read ? "yes" : "no");

  EnsureInit(opts, bdev_capacity_bytes);
  auto *ipc = CLIO_CPU_IPC;

  using clock = std::chrono::steady_clock;
  auto us_since = [](clock::time_point t0) -> long long {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clock::now() - t0)
        .count();
  };

  std::vector<long long> write_us;
  std::vector<long long> flush_us;
  std::vector<long long> read_us;
  write_us.reserve(opts.iters);
  flush_us.reserve(opts.iters);
  read_us.reserve(opts.iters);

  for (clio::run::u32 it = 0; it < opts.iters; ++it) {
    // Fresh Vector per iteration to avoid cumulative cache state.
    std::string tag_name = "bench_iter_" + std::to_string(it);
    gv::Vector<clio::run::u32> vec(tag_name, opts.nblocks, opts.gpu_id,
                              opts.pages_per_block,
                              opts.host_pages_per_block,
                              opts.page_size,
                              opts.cache_period_us,
                              opts.host_pages_per_block > 0
                                  ? gv::CacheMode::kAsync
                                  : gv::CacheMode::kLegacy,
                              opts.manager_threads_per_block,
                              opts.allow_cold_miss_fault);
    auto view = vec.Device();
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(opts.gpu_id);

    // Dedicated non-blocking stream for the user kernels. We
    // cudaStreamSynchronize on THIS stream only — never
    // cudaDeviceSynchronize. The persistent manager runs on its own
    // non-blocking stream inside Vector and is never waited on by the
    // bench. Cache management is fully automatic.
    cudaStream_t user_stream = nullptr;
    cudaStreamCreateWithFlags(&user_stream, cudaStreamNonBlocking);

    // ----- Write -----
    vec.StatsReset();
    std::fprintf(stderr, "[BENCH-DBG] iter %u: launching write kernel\n", it);
    auto t0 = clock::now();
    BenchWriteKernel<<<opts.nblocks, 32, 0, user_stream>>>(
        gpu_info, view, per_block_elems);
    cudaStreamSynchronize(user_stream);
    long long w = us_since(t0);
    auto stats_w = vec.StatsSnapshot();
    std::fprintf(stderr, "[BENCH-DBG] iter %u: write done in %lld us\n",
                 it, w);

    // No flush step — the persistent manager kernel flushes dirty
    // pages continuously in the background. The Vector is
    // automatically coherent.
    long long f = 0;
    gv::VectorStats stats_f{};

    long long r = 0;
    gv::VectorStats stats_r{};
    if (opts.do_read) {
      auto *result = ctp::GpuApi::MallocHost<clio::run::u32>(
          total_elems * sizeof(clio::run::u32));
      if (!result) {
        std::fprintf(stderr, "MallocHost(%llu B) returned nullptr\n",
                     (unsigned long long)(total_elems * sizeof(clio::run::u32)));
        std::exit(2);
      }
      std::memset(result, 0, total_elems * sizeof(clio::run::u32));
      vec.StatsReset();
      std::fprintf(stderr, "[BENCH-DBG] iter %u: launching read kernel\n", it);
      auto t2 = clock::now();
      BenchReadKernel<<<opts.nblocks, 32, 0, user_stream>>>(
          gpu_info, view, result, per_block_elems);
      cudaStreamSynchronize(user_stream);
      r = us_since(t2);
      stats_r = vec.StatsSnapshot();
      std::fprintf(stderr, "[BENCH-DBG] iter %u: read done in %lld us\n",
                   it, r);
      ctp::GpuApi::FreeHost(result);
    }
    cudaStreamDestroy(user_stream);

    auto pct = [](unsigned long long num, unsigned long long denom) {
      if (denom == 0) return 0.0;
      return 100.0 * static_cast<double>(num) /
             static_cast<double>(denom);
    };
    auto dump_stats = [&](const char *phase, const gv::VectorStats &s,
                          long long elapsed_us) {
      double sec = elapsed_us / 1e6;
      auto rate = [&](unsigned long long n) {
        return (sec > 0.0) ? n / sec : 0.0;
      };
      std::fprintf(stderr,
          "[STATS %s %.2fms] resolve total=%llu hits=%llu(%.1f%%) "
          "cold_miss=%llu(%.1f%%) fault_get=%llu spin_iters=%llu\n",
          phase, elapsed_us / 1e3,
          (unsigned long long)s.resolve_total,
          (unsigned long long)s.resolve_hits, pct(s.resolve_hits, s.resolve_total),
          (unsigned long long)s.resolve_cold_miss, pct(s.resolve_cold_miss, s.resolve_total),
          (unsigned long long)s.resolve_fault_get,
          (unsigned long long)s.resolve_spin_iters);
      std::fprintf(stderr,
          "[STATS %s rate/s] manager=%.0f/s prefetch=%.0f/s "
          "swap=%.0f/s flush=%.0f/s rescore=%.0f/s\n",
          phase,
          rate(s.manager_iters),
          rate(s.phase4_prefetches),
          rate(s.phase2_swaps),
          rate(s.phase3_flushes),
          rate(s.phase1c_drained));
      std::fprintf(stderr,
          "[STATS %s] manager iters=%llu work=%llu(%.1f%%) "
          "p1c_drained=%llu p2_swaps=%llu p3_flushes=%llu "
          "p4_pops=%llu p4_prefetch=%llu p4_skip_cached=%llu p4_skip_nofree=%llu\n",
          phase,
          (unsigned long long)s.manager_iters,
          (unsigned long long)s.manager_work_iters,
          pct(s.manager_work_iters, s.manager_iters),
          (unsigned long long)s.phase1c_drained,
          (unsigned long long)s.phase2_swaps,
          (unsigned long long)s.phase3_flushes,
          (unsigned long long)s.phase4_pops,
          (unsigned long long)s.phase4_prefetches,
          (unsigned long long)s.phase4_skip_cached,
          (unsigned long long)s.phase4_skip_nofree);
    };
    dump_stats("write", stats_w, w);
    dump_stats("flush", stats_f, f);
    if (opts.do_read) dump_stats("read ", stats_r, r);

    write_us.push_back(w);
    flush_us.push_back(f);
    read_us.push_back(r);

    auto bw = [&](long long us) {
      if (us <= 0) return 0.0;
      return (opts.total_bytes / static_cast<double>(1ULL << 20)) /
             (us / 1e6);
    };
    std::fprintf(stderr,
                 "[ITER %u/%u] write=%.3f ms (%.1f MiB/s) "
                 "flush=%.3f ms read=%.3f ms (%.1f MiB/s) "
                 "write+flush=%.1f MiB/s\n",
                 it + 1, opts.iters,
                 w / 1e3, bw(w),
                 f / 1e3, r / 1e3,
                 opts.do_read ? bw(r) : 0.0,
                 bw(w + f));
  }

  // ----- Summary -----
  auto stat = [&](std::vector<long long> &v) {
    if (v.empty()) return std::tuple<long long, long long, long long>(0, 0, 0);
    std::sort(v.begin(), v.end());
    return std::tuple<long long, long long, long long>(
        v.front(), v[v.size() / 2], v.back());
  };
  auto [wmin, wmed, wmax] = stat(write_us);
  auto [fmin, fmed, fmax] = stat(flush_us);
  auto [rmin, rmed, rmax] = stat(read_us);
  auto bw_med = [&](long long us) {
    if (us <= 0) return 0.0;
    return (opts.total_bytes / static_cast<double>(1ULL << 20)) /
           (us / 1e6);
  };

  std::fprintf(stderr,
               "\n[SUMMARY] write : min=%.3f med=%.3f max=%.3f ms "
               "(median %.1f MiB/s, %.3f ms/put)\n",
               wmin / 1e3, wmed / 1e3, wmax / 1e3, bw_med(wmed),
               expected_puts
                   ? (wmed / 1e3) / static_cast<double>(expected_puts)
                   : 0.0);
  std::fprintf(stderr,
               "[SUMMARY] flush : min=%.3f med=%.3f max=%.3f ms "
               "(median %.1f MiB/s, %.3f ms/put)\n",
               fmin / 1e3, fmed / 1e3, fmax / 1e3, bw_med(fmed),
               expected_puts
                   ? (fmed / 1e3) / static_cast<double>(expected_puts)
                   : 0.0);
  if (opts.do_read) {
    std::fprintf(stderr,
                 "[SUMMARY] read  : min=%.3f med=%.3f max=%.3f ms "
                 "(median %.1f MiB/s, %.3f ms/get)\n",
                 rmin / 1e3, rmed / 1e3, rmax / 1e3, bw_med(rmed),
                 expected_gets
                     ? (rmed / 1e3) / static_cast<double>(expected_gets)
                     : 0.0);
  }
  std::fprintf(stderr,
               "[SUMMARY] write+flush+read : %.1f MiB/s end-to-end "
               "(median)\n",
               bw_med(wmed + fmed + rmed));
  return 0;
}

#else

int main() { return 0; }

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
