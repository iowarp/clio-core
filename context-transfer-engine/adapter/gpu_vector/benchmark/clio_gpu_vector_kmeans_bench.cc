/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * K-means over a GPU vector whose point set does not fit on the device.
 *
 * The access pattern is the one the GPU vector exists for and is NOT the same
 * as the weights or flush benchmarks:
 *
 *   - it is a STREAMING READ. Every Lloyd iteration walks the entire point set
 *     start to finish, so a page is touched once per iteration and never
 *     revisited within one. A cache smaller than the whole dataset therefore
 *     cannot produce a hit within an iteration -- only ACROSS iterations, and
 *     only for the tail that is still resident when the next pass reaches it.
 *     That is the property this benchmark is here to measure.
 *   - the working state (centroids) is tiny and stays in device memory, so
 *     unlike the flush benchmark there is no writeback on the hot path: the
 *     cost is faults, not puts.
 *
 * Correctness is checked by a centroid checksum after a fixed number of
 * iterations. Page size, cache size, block count and tier capacity must not
 * change WHICH points are summed, so every configuration of the same problem
 * converges to the same centroids and a gross difference means paging
 * corrupted the data.
 *
 * COMPARE IT WITH A TOLERANCE, NOT FOR EQUALITY. The sums are accumulated with
 * atomicAdd, so the ORDER of the float additions depends on how points are
 * distributed across pages and blocks, and float addition is not associative.
 * Measured across 16KB/64KB/1MB pages on the same problem: 30719.999694,
 * 30719.999674, 30720.000029 -- identical to ~1e-8 relative, but not bit-equal.
 * An exact-equality check would report every page-size sweep as corrupt.
 *
 * The point set is generated deterministically from the index, so no input
 * file is needed and every configuration sees identical data.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "bench_memcpy_probe.h"

/** The CTE core pool id, matching the pool_id written into the server
 *  config below. Needed by the baseline path, which talks to the core
 *  directly instead of through the vector. */
const clio::run::PoolId kCorePool(512, 0);

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

/** Per-lane yield frame; coroutine frames are compiler-laid-out and far larger
 *  than the hand-packed macro ones. */
#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_KM_CORO 1
#endif

/** Deterministic synthetic coordinate: cluster-structured, so the assignment
 *  step does real work instead of every point landing on one centroid. */
CTP_INLINE_CROSS_FUN float PointVal(u64 idx, u32 dims, u32 k) {
  const u64 point = idx / dims;
  const u64 dim = idx % dims;
  const u64 cluster = point % k;            // ground-truth cluster
  const float centre = static_cast<float>(cluster) * 8.0f;
  // Deterministic jitter in [-1, 1]; cheap hash so it is reproducible on host
  // and device without a RNG state.
  const u64 h = (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<u32>(h >> 40)) * (2.0f / 16777216.0f) - 1.0f;
  return centre + jitter;
}

/** Squared distance from a point to a centroid, over `dims`. */
CTP_GPU_FUN float DistSq(const float *pt, const float *cent, u32 dims) {
  float d = 0.0f;
  for (u32 i = 0; i < dims; ++i) {
    const float x = pt[i] - cent[i];
    d += x * x;
  }
  return d;
}

// ---------------------------------------------------------------------------
// Seed: write the point set into the vector, one page at a time.
// ---------------------------------------------------------------------------
#if defined(GV_KM_CORO)
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<float> v, u64 per,
                                  u64 page_elems, u32 dims, u32 k, u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
co_await v.BeginFetch(v.PageLo(base + off), v.PageSpan(base + off, n));
    co_await v.AwaitFetch();
        auto h = co_await v.HoldPage(base + off, n, /*write=*/true);
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = PointVal(base + off + i, dims, k);
    }
    // Collective: name the page just written.
    co_await v.BeginFlush(base + off, n);
  }
  // Collect every flush started above: only explicit flushes write data back
  // now (drops refuse dirty pages), so a seeded page left in flight or left
  // to eviction would simply be lost.
  co_await v.EndFlush();
}

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> v, u64 per, u64 page_elems,
                           u32 dims, u32 k, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedCoro(v, per, page_elems, dims, k, yv.Block()));
}

/**
 * One Lloyd assignment pass over this block's slice.
 *
 * Centroid sums are accumulated with global atomics rather than a shared-memory
 * reduction: k*dims is small, the loop is dominated by paging, and a shared
 * tile would have to be sized at compile time for the largest k this accepts.
 */
__device__ gy::YCoroMain AssignCoro(gv::DeviceVector<float> v, u64 per,
                                    u64 page_elems, u32 dims, u32 k,
                                    const float *cent, float *sums,
                                    unsigned *counts, u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
co_await v.BeginFetch(v.PageLo(base + off), v.PageSpan(base + off, n));
    co_await v.AwaitFetch();
        // Read-only pass: no write intent, so every page stays clean and the
    // oversubscribed streaming read sheds pages without writeback.
    auto h = co_await v.HoldPage(base + off, n);
    // Whole pages hold whole points (enforced on the host), so a page is
    // exactly n/dims points and no point straddles a page boundary.
    const u64 npts = n / dims;
    for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
      const u64 pbase = base + off + p * dims;
      float best = 3.4e38f;
      u32 bestk = 0;
      for (u32 c = 0; c < k; ++c) {
        float d = 0.0f;
        for (u32 i = 0; i < dims; ++i) {
          const float x = h[pbase + i] - cent[c * dims + i];
          d += x * x;
        }
        if (d < best) { best = d; bestk = c; }
      }
      for (u32 i = 0; i < dims; ++i) {
        atomicAdd(&sums[bestk * dims + i], h[pbase + i]);
      }
      atomicAdd(&counts[bestk], 1u);
    }
    __syncthreads();
    // NO RELEASE HINT HERE. Telling the cache "this page is dead after use"
    // was measured to be actively WRONG: the page IS re-read on the next
    // Lloyd pass, and the frequency policy was retaining ~10% of them across
    // passes. Releasing guaranteed a 0% hit rate and cost ~6% in rescore
    // traffic on top.
  }
}

__global__ void AssignKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> v, u64 per, u64 page_elems,
                             u32 dims, u32 k, const float *cent, float *sums,
                             unsigned *counts, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(AssignCoro(v, per, page_elems, dims, k, cent, sums, counts,
                            yv.Block()));
}
#else
// The macro/Duff's-device mechanism is not implemented for this benchmark.
// It exists only so the file still compiles under nvcc; the run refuses to
// proceed (see main), rather than silently measuring a different code path.
__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> v, u64 per, u64 page_elems,
                           u32 dims, u32 k, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  (void)info; (void)v; (void)per; (void)page_elems; (void)dims; (void)k;
  (void)yv; (void)ys;
}
__global__ void AssignKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> v, u64 per, u64 page_elems,
                             u32 dims, u32 k, const float *cent, float *sums,
                             unsigned *counts, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  (void)info; (void)v; (void)per; (void)page_elems; (void)dims; (void)k;
  (void)cent; (void)sums; (void)counts; (void)yv; (void)ys;
}
#endif  // GV_KM_CORO

/**
 * BASELINE KERNEL -- the out-of-core model WITHOUT in-kernel faulting.
 *
 * The control for the GPU vector's claim: storage I/O synchronous, HBM<->DRAM
 * copy synchronous, and the kernel torn down for every transfer. One launch
 * per tile, nothing overlapped.
 *
 * Reads its points from a STAGED TILE rather than through the vector, but
 * performs the identical assignment and the identical atomicAdds, so the
 * centroid checksum is directly comparable with the paged path -- with the
 * usual relative tolerance, since atomicAdd makes float summation order
 * follow the launch layout.
 */
__global__ void KmeansBaselineKernel(const float *tile, u64 n, u32 dims, u32 k,
                                     const float *cent, float *sums,
                                     unsigned *counts) {
  const u64 npts = n / dims;
  for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
    const float *pt = tile + p * dims;
    float best = 3.4e38f;
    u32 bestk = 0;
    for (u32 c = 0; c < k; ++c) {
      float d = 0.0f;
      for (u32 i = 0; i < dims; ++i) {
        const float x = pt[i] - cent[c * dims + i];
        d += x * x;
      }
      if (d < best) { best = d; bestk = c; }
    }
    for (u32 i = 0; i < dims; ++i) {
      atomicAdd(&sums[bestk * dims + i], pt[i]);
    }
    atomicAdd(&counts[bestk], 1u);
  }
}

/** centroid = sum / count, leaving an empty cluster where it was. */
__global__ void UpdateKernel(float *cent, const float *sums,
                             const unsigned *counts, u32 dims, u32 k) {
  const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  const unsigned n = counts[c];
  if (n == 0) return;
  for (u32 i = 0; i < dims; ++i) {
    cent[c * dims + i] = sums[c * dims + i] / static_cast<float>(n);
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

/** Runs a yieldable kernel to completion, relaunching as blocks suspend.
 *  Both Reset() calls are required: RunToCompletion does not reset, so a
 *  reused runner whose driver still reads "done" skips the launch entirely
 *  and reports an instant, empty success. */
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000,
      gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, dims = 32, k = 16, slots = 8, iters = 4;
  u64 page_kb = 64, data_mb = 2048, hbm_mb = 512;
  int repeat = 3;
  // Out-of-core WITHOUT in-kernel faulting: sync storage I/O, sync
  // HBM<->DRAM copy, kernel torn down for every transfer.
  bool baseline = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--dims") dims = static_cast<u32>(next());
    else if (a == "--clusters") k = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--iters") iters = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--hbm-mb") hbm_mb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--baseline") baseline = true;
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--dims N] "
                  "[--clusters N] [--slots N] [--iters N] [--page-kb N] "
                  "[--data-mb N] [--hbm-mb N] [--repeat N] [--hbm-only]\n",
                  argv[0]);
      return 0;
    }
  }

#if !defined(GV_KM_CORO)
  std::fprintf(stderr,
               "KMEANS ERROR: built without C++20 device coroutines. This "
               "benchmark only implements the coroutine paging path; running "
               "it under nvcc would measure nothing. Rebuild with "
               "-DCLIO_GPU_YIELD_CORO=ON and CMAKE_CUDA_COMPILER=clang++.\n");
  return 2;
#else
  const u64 page_bytes = page_kb * 1024;
  // REFERENCE CEILING for the paging path: a bare H2D cudaMemcpy at exactly
  // this page size, on an idle device before the runtime starts. The gap
  // between this and the achieved rate is the vector's overhead.
  const MemcpyProbe mcp = ProbeMemcpyBandwidth(static_cast<size_t>(page_bytes));
  const u64 page_elems = page_bytes / sizeof(float);

  // A page must hold a whole number of points, or a point straddles a page
  // boundary and the assignment loop would read across a fault it does not
  // hold. Rejected up front rather than silently rounded, because a rounded
  // page size makes the page-size axis of a sweep a lie.
  if (page_elems % dims != 0) {
    std::fprintf(stderr,
                 "KMEANS ERROR: page of %lluKB holds %llu floats, which is not "
                 "a multiple of dims=%u. Choose a page size and dims that "
                 "divide evenly.\n",
                 (unsigned long long)page_kb, (unsigned long long)page_elems,
                 dims);
    return 2;
  }

  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 per = (total_elems / blocks / page_elems) * page_elems;  // page-aligned
  if (per == 0) {
    std::fprintf(stderr, "KMEANS ERROR: %lluMB over %u blocks leaves less than "
                 "one %lluKB page per block.\n",
                 (unsigned long long)data_mb, blocks,
                 (unsigned long long)page_kb);
    return 2;
  }
  const u64 n = per * blocks;
  const u64 npoints = n / dims;
  const double logical_mb =
      static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);

  {
    std::ofstream cfg("gv_kmeans_bench.yaml");
    cfg << "networking:\n  port: 9439\n\n"
        // Workers that sleep add their sleep to every fault, and a fault is a
        // synchronous round trip -- that buries the paging differences this
        // benchmark measures. Keep them spinning.
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        // MaxBwDpe splits on target_score_ <= blob_score and sorts the
        // preferred group DESCENDING; the vector puts pages at blob score 1.0,
        // so the HIGHER score is the preferred tier. HBM must therefore be
        // above the host tier, not below it.
        << "      - path: \"hbm::gv_km_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: 1.0\n";
    if (!hbm_only) {
      cfg << "      - path: \"ram::gv_km_ram\"\n        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << (data_mb + 512) << "MB\"\n"
          << "        score: 0.2\n";
    }
      // OPTIONAL STORAGE TIER. Without it the whole dataset lives in host
      // DRAM and NOTHING EVER TOUCHES STORAGE -- what such a run measures is
      // DRAM over PCIe, not I/O. On this machine that spill is nearly free,
      // which is exactly why a cache-size sweep over a DRAM-only hierarchy
      // comes back flat: there is no penalty for the cache to save.
      //
      // score BELOW the host tier. MaxBwDpe splits on target_score <=
      // blob_score and sorts the preferred group DESCENDING, and the vector
      // puts pages at blob score 1.0, so HIGHER score = preferred. This is the
      // REVERSE of the GNN trainer's hierarchy, whose put path uses blob score
      // 0.5 -- copying its numbers here would silently make storage the
      // FIRST-choice tier.
      if (nvme_mb > 0) {
        cfg << "      - path: \"" << nvme_path << "\"\n"
            << "        bdev_type: \"file\"\n"
            << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
            << "        score: 0.0\n";
      }

    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_kmeans_bench.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "KMEANS ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "KMEANS ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  std::printf("kmeans over a GPU vector\n"
              "  blocks=%u threads=%u dims=%u k=%u iters=%u\n"
              "  page=%lluKB (%llu floats = %llu points/page)  cache=%u "
              "pages/block\n"
              "  data=%.0fMB (%llu points)  kHBM tier=%lluMB%s\n",
              blocks, threads, dims, k, iters,
              (unsigned long long)page_kb, (unsigned long long)page_elems,
              (unsigned long long)(page_elems / dims), slots, logical_mb,
              (unsigned long long)npoints, (unsigned long long)hbm_mb,
              hbm_only ? " (HBM ONLY)" : "");

  gv::Vector<float> vec("gv_kmeans", {0}, page_bytes, blocks, slots, n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(blocks, threads);

  // ---- seed the point set -------------------------------------------------
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, per, page_elems,
                                                dims, k, vw, sv);
  });
  ctp::GpuApi::Synchronize();

  // ---- device state -------------------------------------------------------
  float *d_cent = nullptr, *d_sums = nullptr;
  unsigned *d_counts = nullptr;
  d_cent = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_cent)>>(k * dims * sizeof(float));
  d_sums = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sums)>>(k * dims * sizeof(float));
  d_counts = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_counts)>>(k * sizeof(unsigned));

  // Initial centroids: the first k points, taken on the host from the same
  // generator, so every configuration starts identically.
  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u32 c = 0; c < k; ++c) {
    for (u32 i = 0; i < dims; ++i) {
      h_cent[c * dims + i] = PointVal(static_cast<u64>(c) * dims + i, dims, k);
    }
  }

  // ---- BASELINE DRIVER ------------------------------------------------
  // One tile at a time: block on the CTE read, block on the H2D copy, launch,
  // tear the grid down, repeat. Nothing overlaps.
  ctp::ipc::FullPtr<char> bl_host;
  float *bl_dev = nullptr;
  clio::cte::core::Client *bl_core = nullptr;
  const u64 total_pages = (per * static_cast<u64>(blocks)) / page_elems;
  if (baseline) {
    bl_host = CLIO_IPC->AllocateBuffer(static_cast<size_t>(page_bytes));
    if (bl_host.IsNull()) {
      std::fprintf(stderr, "KMEANS ERROR: baseline staging alloc failed\n");
      return 1;
    }
    // GpuApi::Malloc fails fatally, which is the same abort with less code.
    bl_dev = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(bl_dev)>>(
        static_cast<size_t>(page_bytes));
    bl_core = new clio::cte::core::Client(kCorePool);
  }
  auto run_baseline_pass = [&]() {
    for (u64 p = 0; p < total_pages; ++p) {
      const std::string nm = std::to_string(p);
      auto gf = bl_core->AsyncGetBlob(vec.TagId(), nm, 0, page_bytes, 0,
                                      bl_host.shm_.template Cast<void>(),
                                      clio::run::PoolQuery::Local());
      gf.Wait();
      if (gf->GetReturnCode() != 0) continue;
      ctp::GpuApi::Memcpy(reinterpret_cast<char *>(bl_dev), bl_host.ptr_,
                          static_cast<size_t>(page_bytes));
      KmeansBaselineKernel<<<1, threads>>>(bl_dev, page_elems, dims, k, d_cent,
                                           d_sums, d_counts);
      ctp::GpuApi::Synchronize();
    }
    return true;
  };

  double best_ms = 1e30;
  std::vector<float> h_final(static_cast<size_t>(k) * dims);
  for (int r = 0; r < repeat; ++r) {
    ctp::GpuApi::Memcpy(d_cent, h_cent.data(), h_cent.size() * sizeof(float));
    vec.ResetStats();
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();
    for (u32 it = 0; it < iters; ++it) {
      ctp::GpuApi::Memset(d_sums, 0, k * dims * sizeof(float));
      ctp::GpuApi::Memset(d_counts, 0, k * sizeof(unsigned));
      if (baseline) {
        if (!run_baseline_pass()) {
          std::fprintf(stderr, "KMEANS ERROR: baseline tile loop failed\n");
          return 1;
        }
      } else {
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          AssignKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu, dev, per, page_elems, dims, k, d_cent, d_sums, d_counts, vw,
              sv);
        });
      }
      UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums, d_counts, dims, k);
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    ctp::GpuApi::Memcpy(h_final.data(), d_cent, h_final.size() * sizeof(float));
  }

  // Checksum of the final centroids. Compare across configurations with a
  // RELATIVE TOLERANCE (~1e-4 is generous): atomicAdd makes the summation
  // order depend on the page and block layout, and float addition is not
  // associative, so bit-equality is not expected even when the run is
  // perfectly correct.
  double csum = 0.0;
  for (float f : h_final) csum += static_cast<double>(f);

  const auto st = vec.ReadStats(0);
  // Bytes read per iteration is the whole point set; iters passes per timed
  // run. Reported as effective bandwidth over the measured time.
  const double gbps = (best_ms > 0.0)
      ? (logical_mb * iters / 1024.0) / (best_ms / 1000.0) : 0.0;

  // ---- TIER PLACEMENT CHECK -------------------------------------------
  // Data must land in the FASTEST tier that has capacity, spilling only once
  // that tier is full. MEASURED, not assumed: MaxBwDpe splits tiers on
  // target_score <= blob_score and ranks within the group, so a tier scored on
  // the wrong side of the blob's own score silently drops out of the preferred
  // set. That has produced "kHBM 0MiB" with a healthy, correctly sized HBM
  // tier sitting empty -- no error, just a run that never touched the GPU tier
  // and lost the device-to-device fault path with it.
  //
  // Tier bdevs are numbered from major 512 with minor 1, in CONFIG ORDER,
  // independent of cte_core's own major: (512,1) is the first storage
  // entry (kHBM) and (513,1) the second (host). Deriving them from
  // cte_core's major instead gave remaining > capacity -- an impossible
  // reading that would have been reported as a placement violation.
  {
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    auto ha = t_host.AsyncGetStats(); ha.Wait();
    const clio::run::u64 fast_cap = (clio::run::u64)hbm_mb * 1024ull * 1024ull;
    const clio::run::u64 host_cap = (clio::run::u64)(data_mb + 512) * 1024ull * 1024ull;
    const clio::run::u64 fast_used =
        fast_cap > fa->remaining_size_ ? fast_cap - fa->remaining_size_ : 0;
    const clio::run::u64 host_used =
        host_cap > ha->remaining_size_ ? host_cap - ha->remaining_size_ : 0;
    // RAW remaining is printed alongside the derived used, because the
    // derived number alone is not interpretable: if a queried pool does not
    // exist or the stat fails, remaining reads 0 and "used" then equals the
    // full capacity -- which looks like a completely full tier rather than a
    // failed query. Both were indistinguishable in the first version of this
    // report and it nearly produced a false VIOLATION.
    std::fprintf(stderr,
                 "TIER SPLIT: kHBM used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "host used=%lluMiB cap=%lluMiB remain=%lluMiB%s\n",
                 (unsigned long long)(fast_used >> 20),
                 (unsigned long long)(fast_cap >> 20),
                 (unsigned long long)(fa->remaining_size_ >> 20),
                 (unsigned long long)(host_used >> 20),
                 (unsigned long long)(host_cap >> 20),
                 (unsigned long long)(ha->remaining_size_ >> 20),
                 (fast_used == 0 && fa->remaining_size_ == fast_cap)
                     ? "   <-- nothing landed in the fastest tier"
                     : "");
  }

  std::fprintf(stderr,
               "KMEANS mode=%s blocks=%u thr=%u dims=%u k=%u iters=%u page_kb=%llu "
               "slots=%u data_mb=%.0f hbm_mb=%llu points=%llu ms=%.1f "
               "GB/s=%.2f centroid_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
               baseline ? "baseline" : "paged",
               blocks, threads, dims, k, iters, (unsigned long long)page_kb,
               slots, logical_mb, (unsigned long long)hbm_mb,
               (unsigned long long)npoints, best_ms, gbps, csum,
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.puts, (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps);

  ctp::GpuApi::Free(d_cent); ctp::GpuApi::Free(d_sums); ctp::GpuApi::Free(d_counts);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
#endif  // GV_KM_CORO
}

#endif  // !CTP_IS_DEVICE_PASS
