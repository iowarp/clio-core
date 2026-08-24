/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Can a paged GPU vector hide its I/O behind compute?
 *
 * The workload is deliberately the simplest thing that has both halves: sum a
 * page, then do a fixed amount of real arithmetic on it. The amount of
 * arithmetic is CALIBRATED so that compute-only time equals I/O-only time, so
 * the 50/50 mix is established by measurement rather than assumed.
 *
 * The working set is a multiple of what the device cache holds (--oversub,
 * default 2x), so it CANNOT all be resident: every page is faulted, used, and
 * evicted. That is the regime where prefetching either works or does not.
 *
 * Two modes over identical work:
 *
 *   serial    fault page i, compute on it, move on. I/O and compute strictly
 *             alternate, so the runtime is faults + compute.
 *   prefetch  ask for page i+1 (BeginFetch) BEFORE computing on page i, and
 *             use RescorePage to pin the page in use and release the previous
 *             one so eviction takes the right victim. The transfer for i+1
 *             runs while the SM computes on i.
 *
 * With a 50/50 mix, perfect overlap makes prefetch take max(io, compute)
 * instead of io + compute -- half the runtime. The benchmark reports the
 * breakdown, the speedup, and what fraction of the available I/O was actually
 * hidden, so a partial win is visible as a number rather than a vibe.
 *
 * Correctness is checked every run: the sum is compared against the closed
 * form. An overlap that returns wrong data is a failure, not a speedup.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/gpu/yield_stack.h>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {

constexpr u64 kPageBytes = 65536;
constexpr u64 kPageElems = kPageBytes / sizeof(u32);

/** Element value. Small ints so the exact sum is computable on the host. */
CTP_INLINE_CROSS_FUN u32 Elem(u64 i) { return static_cast<u32>(i % 251); }

}  // namespace

/** Fill the vector and flush it, so every page exists as a blob.
 *
 * A COROUTINE: the blocking HoldPage spun in-kernel on every fault, which is
 * the in-kernel-wait pattern that wedges the moment anything on the service
 * path needs a kernel launch, and the reason this benchmark timed out. Each
 * fault suspends instead; the block's flush is one batched FlushAsync at the
 * end rather than a put-and-wait per page. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<u32> v, u64 per,
                                  u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 i = 0; i < per;) {
    u64 run = 0;
    {
      co_await v.Fetch(v.PageLo(base + i), v.PageSpan(base + i, 1));
      auto h = co_await v.HoldPage(base + i, per - i, /*write=*/true);
      run = h.run();
      for (u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[base + i + k] = Elem(base + i + k);
      }
      __syncthreads();
    }
    // FLUSH AS WE GO, and this is the overlap the benchmark measures: a
    // ranged BeginFlush returns as soon as the put is submitted, so this
    // page's writeback rides alongside the next page's fetch and compute.
    // Deferring every flush to the end would dirty the whole table and then
    // fail -- a dirty frame is not evictable, so the region cannot exceed
    // the cache without flushing first.
    co_await v.BeginFlush(base + i, run);
    i += run;
  }
  co_await v.EndFlush();
}

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 per,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedCoro(v, per, yv.Block()));
}

/**
 * The compute half: a fixed amount of REAL arithmetic, done by every lane.
 *
 * Deliberately not a clock64() spin. A spin loop with __nanosleep measures
 * SM cycles, and its wall cost moves with the clock: in the mixed run the SM
 * idles during faults and clocks up, so the identical "compute" got cheaper
 * and the mixed run measured FASTER than compute alone -- a nonsense result
 * that made the whole comparison invalid. A dependent FMA chain keeps the SM
 * genuinely busy and costs the same instructions in every mode.
 */
CTP_GPU_FUN float DoWork(u32 iters, float seed) {
  float x = seed;
  // Dependent chain so the compiler cannot vectorise or elide it.
  for (u32 i = 0; i < iters; ++i) {
    x = fmaf(x, 1.0000001f, 1.0e-7f);
  }
  return x;
}

/**
 * Sum + compute, with prefetching of the next page when enabled.
 *
 * Lane 0 drives paging; the whole block participates in the summation, so the
 * page is read the way a real kernel would read it.
 */
__device__ gy::YCoroMain SumComputeCoro(gv::DeviceVector<u32> v, u64 per,
                                        u32 compute_iters, int prefetch,
                                        u32 depth, unsigned long long *total,
                                        u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  const u64 first_page = base / v.ElemsPerPage();
  const u64 pages = per / v.ElemsPerPage();
  unsigned long long acc = 0;

  const u64 pe = v.ElemsPerPage();
  (void) first_page;
  // PRIME: page 0 must be resident before the loop, because the loop body
  // issues the fetch for page p+1 and then computes on p.
  co_await v.Fetch(v.PageLo(base), pe);

  for (u64 p = 0; p < pages; ++p) {
    const u64 off = base + p * pe;
    if (prefetch && p + 1 < pages) {
      // THE OVERLAP: submit the next window and do NOT await it. The page
      // being computed on is already resident, so this transfer is in flight
      // for the whole compute below. `depth` pages per window because one
      // page of lookahead cannot cover a fault round trip unless a page's
      // compute exceeds it.
      u64 win = (depth > 0) ? depth : 1;
      if (p + 1 + win > pages) win = pages - (p + 1);
      co_await v.BeginFetch(v.PageLo(base + (p + 1) * pe), pe * win);
    } else if (!prefetch) {
      // No lookahead: pay the fault right here, which is the baseline this
      // case is measured against.
      co_await v.Fetch(v.PageLo(off), pe);
    }
    auto h = co_await v.HoldPage(off, pe);
    unsigned long long local = 0;
    for (u64 i = threadIdx.x; i < h.run(); i += blockDim.x) {
      local += h[off + i];
    }
    acc += local;
    __syncthreads();

    // The compute half of the mix. Every lane works, like real compute.
    // Seed from the PAGE and the data just read, so the chain is not
    // loop-invariant. Seeding it from threadIdx alone lets the compiler hoist
    // the whole chain out of the page loop and compute it once -- the
    // "compute" would then cost 1/pages of what it claims to.
    acc += static_cast<unsigned long long>(
        DoWork(compute_iters, 1.0f + static_cast<float>(threadIdx.x) +
                                  static_cast<float>(p) +
                                  static_cast<float>(local & 0xff)) *
        1.0e-9f);
    __syncthreads();
  }
  atomicAdd(total, acc);
}

__global__ void SumComputeKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 per,
                                 u32 compute_iters, int prefetch, u32 depth,
                                 unsigned long long *total,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(
      SumComputeCoro(v, per, compute_iters, prefetch, depth, total,
                     yv.Block()));
}

/* Dropping resident pages is a HOST operation now (Vector::ClearCache).
 * Cache management moved off the device with the rest of the slot policy, so
 * there is no DropAllKernel any more -- callers reset between modes from the
 * host, between launches, where it belongs. */

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr unsigned kYieldLaneBytes = 8192;

/** Drive a coroutine kernel to completion: launch, service, relaunch. */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, kYieldLaneBytes);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000,
      gv::ResumeWhenComplete);
}

struct Args {
  u32 blocks = 8;
  u64 pages_per_block = 32;   // working-set pages each block walks
  u32 oversub = 2;            // working set / cache size
  u32 compute_iters = 0;      // 0 = auto (calibrate to match measured I/O)
  u32 depth = 4;              // pages kept in flight when prefetching
  u32 repeat = 3;
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

unsigned long long ExpectedSum(u64 n) {
  unsigned long long acc = 0;
  for (u64 i = 0; i < n; ++i) acc += Elem(i);
  return acc;
}

/** One timed run. Returns wall ms; fills stats and the checksum verdict. */
double RunOnce(gv::Vector<u32> &host_vec, const gv::DeviceVector<u32> &dev,
               const clio::run::IpcManagerGpuInfo &gpu, u32 blocks, u64 per,
               u32 compute_iters, int prefetch, u32 depth, bool cold,
               unsigned long long *d_sum, unsigned long long *out_sum) {
  ctp::GpuApi::Memset(d_sum, 0, sizeof(unsigned long long));
  // Cold start, OUTSIDE the timed region: otherwise whichever mode runs
  // second inherits the other's warm cache. The first version of this
  // benchmark did exactly that and gave prefetch 152 faults against
  // serial's 256 -- a "speedup" that was really just a warmer cache.
  // NOT for the resident compute-only baseline: dropping its cache would make
  // it fault too, and then "compute only" measures compute PLUS I/O. Doing
  // that silently calibrated the compute down to near zero, so there was
  // nothing left for a prefetch to hide and every mode measured the same.
  if (cold) {
    // Host-side now: cache management left the device with the slot policy.
    host_vec.ClearCache();
    ctp::GpuApi::Synchronize();
  }
  const double t0 = NowMs();
  RunYieldable(blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                           gy::YieldStackView sv) {
    SumComputeKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu, dev, per, compute_iters, prefetch, depth, d_sum, vw, sv);
  });
  ctp::GpuApi::Synchronize();
  const double ms = NowMs() - t0;
  ctp::GpuApi::Memcpy(out_sum, d_sum, sizeof(*out_sum));
  return ms;
}

}  // namespace

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : "0"; };
    if (f == "--blocks") a.blocks = std::atoi(next());
    else if (f == "--pages") a.pages_per_block = std::atoll(next());
    else if (f == "--oversub") a.oversub = std::atoi(next());
    else if (f == "--compute-iters") a.compute_iters = std::atoi(next());
    else if (f == "--depth") a.depth = std::atoi(next());
    else if (f == "--repeat") a.repeat = std::atoi(next());
    else if (f == "--help") {
      std::printf(
          "usage: %s [--blocks N] [--pages N] [--oversub N] "
          "[--compute-iters N] [--depth N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }
  // The cache is the working set divided by the oversubscription factor: at
  // the default 2x, the data is twice what the device can hold.
  const u32 slots =
      static_cast<u32>(a.pages_per_block / (a.oversub ? a.oversub : 1));
  if (slots < 3) {
    std::fprintf(stderr,
                 "need >=3 cache slots (one in use, one prefetching, one "
                 "being evicted); got %u\n", slots);
    return 1;
  }

  {
    std::ofstream cfg("gpu_vector_overlap.yaml");
    cfg << "networking:\n  port: 9436\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"4GB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_overlap_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"2GB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_overlap.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);


  const u64 per = a.pages_per_block * kPageElems;
  const u64 n = per * a.blocks;
  const double data_mb =
      static_cast<double>(n * sizeof(u32)) / (1024.0 * 1024.0);
  const double cache_mb = static_cast<double>(
      static_cast<u64>(a.blocks) * slots * kPageBytes) / (1024.0 * 1024.0);

  std::printf(
      "gpu_vector overlap benchmark\n"
      "  blocks=%u pages/block=%llu slots/block=%u oversub=%ux depth=%u\n"
      "  page=%lluKB working set=%.1fMB device cache=%.1fMB\n",
      a.blocks, (unsigned long long) a.pages_per_block, slots, a.oversub, a.depth,
      (unsigned long long) (kPageBytes / 1024), data_mb, cache_mb);

  gv::Vector<u32> vec("gv_overlap", {0}, kPageBytes, a.blocks, slots, n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  RunYieldable(a.blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
    SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, per, vw, sv);
  });
  ctp::GpuApi::Synchronize();

  unsigned long long *d_sum =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));
  const unsigned long long want = ExpectedSum(n);

  // ---- Step 0: the SCAN BASELINE, present in every mode -----------------
  //
  // Reading a page costs real time even when it is already resident: the
  // summation touches every element through at(), which resolves a page per
  // access. That cost is in EVERY mode, so it must be subtracted before
  // anything is called "the I/O half" or "the compute half".
  //
  // Getting this wrong invalidated an entire earlier round of results. The
  // scan is ~11 ms here while faulting is ~0.7 ms, so attributing the whole
  // cold-run time to I/O inflated per-page fault cost by ~16x and made a
  // strictly sequential "serial" run look like it cost max(io, compute)
  // rather than io+compute -- i.e. like the vector was overlapping when it
  // does no such thing. It faults on demand, exactly as it looks like it does.
  unsigned long long got = 0;
  double scan_ms = 1e30;
  {
    gv::Vector<u32> warm("gv_overlap_scan", {0}, kPageBytes, a.blocks,
                         static_cast<u32>(a.pages_per_block), n);
    auto wdev = warm.GetDevice(0);
    RunYieldable(a.blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                               gy::YieldStackView sv) {
      SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, wdev, per, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    for (u32 r = 0; r < a.repeat; ++r) {
      const double ms =
          RunOnce(warm, wdev, gpu, a.blocks, per, 0, 0, 0, false, d_sum, &got);
      if (ms < scan_ms) scan_ms = ms;
    }
  }
  std::printf("\n[scan]      %8.1f ms   (resident, no faults, no compute)\n",
              scan_ms);

  // ---- Step 1: measure the I/O half alone (no compute) ------------------
  double io_ms = 1e30;
  for (u32 r = 0; r < a.repeat; ++r) {
    vec.ResetStats();
    const double ms =
        RunOnce(vec, dev, gpu, a.blocks, per, 0, 0, 0, true, d_sum, &got);
    if (ms < io_ms) io_ms = ms;
  }
  const bool io_ok = (got == want);
  const auto io_stats = vec.ReadStats(0);
  const u64 total_pages = a.pages_per_block * a.blocks;
  // The I/O contribution is the MARGINAL cost over the scan, not the whole run.
  const double io_only_ms = (io_ms > scan_ms) ? (io_ms - scan_ms) : 0.0;
  const double us_per_page =
      io_only_ms * 1000.0 / static_cast<double>(total_pages);

  std::printf(
      "[io-only]   %8.1f ms   faults=%llu evicts=%llu   io=%.1f ms over scan "
      "(%.1f us/page)  %s\n",
      io_ms, (unsigned long long) io_stats.faults,
      (unsigned long long) io_stats.evicts, io_only_ms, us_per_page,
      io_ok ? "sum=OK" : "sum=MISMATCH");
  if (!io_ok) {
    std::fprintf(stderr, "checksum mismatch in io-only run; aborting\n");
    return 1;
  }

  // ---- Step 2: CALIBRATE compute to match the I/O half ------------------
  //
  // The burn is a clock64() spin, and clock64() counts SM cycles whose wall
  // rate moves with clock throttling -- a fixed microsecond target costs
  // measurably different wall time under different load. Setting the burn
  // from prop.clockRate produced a "50/50" mix that was really about 1:2, and
  // a compute-only run SLOWER than the mixed run it was supposed to bound.
  //
  // So the mix is calibrated instead of computed: pick the constant that makes
  // the MARGINAL compute cost equal the MARGINAL I/O cost, both measured over
  // the scan baseline. Calibrating against the totals instead is degenerate --
  // the shared scan term dominates both sides, so it converges with the actual
  // compute near zero and there is nothing to overlap.
  double compute_ms = 1e30;
  u32 compute_iters = a.compute_iters ? a.compute_iters : 20000u;
  {
    gv::Vector<u32> hot("gv_overlap_hot", {0}, kPageBytes, a.blocks,
                        static_cast<u32>(a.pages_per_block), n);
    auto hotdev = hot.GetDevice(0);
    RunYieldable(a.blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                               gy::YieldStackView sv) {
      SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, hotdev, per, vw, sv);
    });
    ctp::GpuApi::Synchronize();

    auto measure = [&](u32 cu) {
      double best = 1e30;
      for (u32 r = 0; r < a.repeat; ++r) {
        const double ms =
            RunOnce(hot, hotdev, gpu, a.blocks, per, cu, 0, 0, false, d_sum, &got);
        if (ms < best) best = ms;
      }
      return best;
    };

    compute_ms = measure(compute_iters);
    if (!a.compute_iters) {
      // Match the marginal costs: (compute_ms - scan) == (io_ms - scan).
      for (int iter = 0; iter < 8; ++iter) {
        const double marginal = compute_ms - scan_ms;
        if (marginal <= 0.0) {   // too small to measure yet; grow it
          compute_iters *= 4;
          compute_ms = measure(compute_iters);
          continue;
        }
        const double err = marginal / io_only_ms;
        if (err > 0.9 && err < 1.1) break;
        double scaled = static_cast<double>(compute_iters) / err;
        if (scaled < 1.0) scaled = 1.0;
        if (scaled > 1.0e8) scaled = 1.0e8;
        compute_iters = static_cast<u32>(scaled + 0.5);
        compute_ms = measure(compute_iters);
      }
      std::printf(
          "  -> compute calibrated to %u FMA iters/page: marginal compute "
          "%.1f ms vs marginal io %.1f ms\n",
          compute_iters, compute_ms - scan_ms, io_only_ms);
    } else {
      std::printf("  -> compute forced to %u iters/page\n", compute_iters);
    }
  }
  std::printf("[compute]   %8.1f ms   (resident, no faults)\n", compute_ms);

  // ---- Step 3: the two modes over the same oversubscribed working set ---
  double serial_ms = 1e30, pre_ms = 1e30;
  gv::Vector<u32>::Stats serial_stats{}, pre_stats{};
  bool serial_ok = false, pre_ok = false;

  for (u32 r = 0; r < a.repeat; ++r) {
    vec.ResetStats();
    const double ms = RunOnce(vec, dev, gpu, a.blocks, per, compute_iters, 0, 0, true, d_sum, &got);
    if (ms < serial_ms) {
      serial_ms = ms;
      serial_stats = vec.ReadStats(0);
      serial_ok = (got == want);
    }
  }
  for (u32 r = 0; r < a.repeat; ++r) {
    vec.ResetStats();
    const double ms = RunOnce(vec, dev, gpu, a.blocks, per, compute_iters, 1, a.depth, true, d_sum,
                              &got);
    if (ms < pre_ms) {
      pre_ms = ms;
      pre_stats = vec.ReadStats(0);
      pre_ok = (got == want);
    }
  }

  std::printf(
      "\n[serial]    %8.1f ms   faults=%llu evicts=%llu   %s\n"
      "[prefetch]  %8.1f ms   faults=%llu evicts=%llu   %s\n",
      serial_ms, (unsigned long long) serial_stats.faults,
      (unsigned long long) serial_stats.evicts,
      serial_ok ? "sum=OK" : "sum=MISMATCH",
      pre_ms, (unsigned long long) pre_stats.faults,
      (unsigned long long) pre_stats.evicts,
      pre_ok ? "sum=OK" : "sum=MISMATCH");

  // ---- Breakdown --------------------------------------------------------
  // Perfect overlap costs max(io, compute); no overlap costs io + compute.
  // Where the prefetch run lands between those two IS the overlap achieved.
  // Everything is expressed over the scan floor, which every mode pays and
  // which no amount of overlapping can remove.
  const double compute_only_ms =
      (compute_ms > scan_ms) ? (compute_ms - scan_ms) : 0.0;
  const double no_overlap = scan_ms + io_only_ms + compute_only_ms;
  const double perfect =
      scan_ms + ((io_only_ms > compute_only_ms) ? io_only_ms : compute_only_ms);
  const double hidden = (no_overlap > perfect)
                            ? (no_overlap - pre_ms) / (no_overlap - perfect)
                            : 0.0;
  std::printf(
      "\nbreakdown (every mode pays the scan; io and compute are MARGINAL)\n"
      "  scan floor           %8.1f ms   (resident reads, unavoidable)\n"
      "  + io                 %8.1f ms   (%.1f us/page of faulting)\n"
      "  + compute            %8.1f ms\n"
      "  = no overlap         %8.1f ms   (scan + io + compute)\n"
      "  = perfect overlap    %8.1f ms   (scan + max(io, compute))\n"
      "  serial measured      %8.1f ms\n"
      "  prefetch measured    %8.1f ms\n"
      "  speedup              %8.2fx\n"
      "  I/O hidden           %8.1f %%\n",
      scan_ms, io_only_ms, us_per_page, compute_only_ms, no_overlap, perfect,
      serial_ms, pre_ms, serial_ms / pre_ms, hidden * 100.0);

  // Honesty check on the baseline. With many blocks resident the GPU already
  // overlaps one block's fault against another block's compute, so "serial"
  // is not serial at the device level and there is little left for an
  // explicit prefetch to win. When that is happening, say so instead of
  // reporting a small speedup as if prefetching had underperformed.
  if (serial_ms < no_overlap * 0.75 && a.blocks > 1) {
    std::printf(
        "\n  NOTE: serial (%.1f ms) is well under io+compute (%.1f ms), so the\n"
        "  %u concurrent blocks are ALREADY overlapping each other's I/O and\n"
        "  compute. The headroom an explicit prefetch can win is only\n"
        "  %.1f ms (serial - max), not %.1f ms. Re-run with --blocks 1 to see\n"
        "  the per-block effect without inter-block overlap.\n",
        serial_ms, no_overlap, a.blocks, serial_ms - perfect,
        no_overlap - perfect);
  }

  if (!serial_ok || !pre_ok) {
    std::fprintf(stderr, "\nCHECKSUM MISMATCH -- results above are void\n");
    return 1;
  }
  ctp::GpuApi::Free(d_sum);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
