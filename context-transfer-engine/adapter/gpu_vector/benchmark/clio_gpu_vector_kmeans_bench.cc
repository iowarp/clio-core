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
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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
  u64 run = 0;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    co_await v.HoldPageCoro(base + off, n, &run);
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      v[base + off + i] = PointVal(base + off + i, dims, k);
    }
    __syncthreads();
    if (threadIdx.x == 0) v.BeginFlush(base + off, n);
    __syncthreads();
  }
}

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> v, u64 per, u64 page_elems,
                           u32 dims, u32 k, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
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
  u64 run = 0;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    co_await v.HoldPageCoro(base + off, n, &run);
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
          const float x = v.at(pbase + i) - cent[c * dims + i];
          d += x * x;
        }
        if (d < best) { best = d; bestk = c; }
      }
      for (u32 i = 0; i < dims; ++i) {
        atomicAdd(&sums[bestk * dims + i], v.at(pbase + i));
      }
      atomicAdd(&counts[bestk], 1u);
    }
    __syncthreads();
  }
}

__global__ void AssignKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> v, u64 per, u64 page_elems,
                             u32 dims, u32 k, const float *cent, float *sums,
                             unsigned *counts, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
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
        [] {}, /*max_rounds=*/2000000);
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
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "KMEANS ERROR: seed failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  // ---- device state -------------------------------------------------------
  float *d_cent = nullptr, *d_sums = nullptr;
  unsigned *d_counts = nullptr;
  cudaMalloc(&d_cent, k * dims * sizeof(float));
  cudaMalloc(&d_sums, k * dims * sizeof(float));
  cudaMalloc(&d_counts, k * sizeof(unsigned));

  // Initial centroids: the first k points, taken on the host from the same
  // generator, so every configuration starts identically.
  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u32 c = 0; c < k; ++c) {
    for (u32 i = 0; i < dims; ++i) {
      h_cent[c * dims + i] = PointVal(static_cast<u64>(c) * dims + i, dims, k);
    }
  }

  double best_ms = 1e30;
  std::vector<float> h_final(static_cast<size_t>(k) * dims);
  for (int r = 0; r < repeat; ++r) {
    cudaMemcpy(d_cent, h_cent.data(), h_cent.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    vec.ResetStats();
    cudaDeviceSynchronize();
    const double t0 = NowMs();
    for (u32 it = 0; it < iters; ++it) {
      cudaMemset(d_sums, 0, k * dims * sizeof(float));
      cudaMemset(d_counts, 0, k * sizeof(unsigned));
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        AssignKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dev, per, page_elems, dims, k, d_cent, d_sums, d_counts, vw,
            sv);
      });
      UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums, d_counts, dims, k);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "KMEANS ERROR: assign failed: %s\n",
                   cudaGetErrorString(cudaGetLastError()));
      return 1;
    }
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    cudaMemcpy(h_final.data(), d_cent, h_final.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
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

  std::fprintf(stderr,
               "KMEANS blocks=%u thr=%u dims=%u k=%u iters=%u page_kb=%llu "
               "slots=%u data_mb=%.0f hbm_mb=%llu points=%llu ms=%.1f "
               "GB/s=%.2f centroid_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu\n",
               blocks, threads, dims, k, iters, (unsigned long long)page_kb,
               slots, logical_mb, (unsigned long long)hbm_mb,
               (unsigned long long)npoints, best_ms, gbps, csum,
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.puts, (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors);

  cudaFree(d_cent); cudaFree(d_sums); cudaFree(d_counts);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
#endif  // GV_KM_CORO
}

#endif  // !CTP_IS_DEVICE_PASS
