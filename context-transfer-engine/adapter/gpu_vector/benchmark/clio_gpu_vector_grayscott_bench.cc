/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott reaction-diffusion over a GPU vector whose grid does not fit on
 * the device.
 *
 * This is the STENCIL access pattern, and it is the one the other three
 * benchmarks do not have:
 *
 *   - flush   writes a region and flushes it       (write, no reuse)
 *   - weights re-reads a matrix every pass         (read, full reuse)
 *   - kmeans  streams the point set once per pass  (read, no reuse)
 *   - THIS    reads a 3-plane window that SLIDES   (read+write, partial reuse)
 *
 * The sliding window is the point. Computing plane z needs planes z-1, z and
 * z+1, and computing z+1 then needs z, z+1, z+2 -- so two of the three planes
 * are immediately reused. A cache of >= 4 pages therefore turns 3 reads per
 * plane into 1, and a smaller cache cannot. That is a real reuse distance,
 * unlike kmeans where nothing is revisited within a pass.
 *
 * LAYOUT. One page is exactly one XY plane, so the page size chosen on the
 * command line sets the plane dimensions (16 KB -> 64x64, 8 MB -> 2048x1024).
 * That makes --page-kb a first-class axis instead of an arbitrary chunking of
 * a fixed grid. The vector holds four regions -- u, v, u_next, v_next -- and
 * the step writes into the next pair and then swaps, because a stencil cannot
 * be computed in place.
 *
 * CACHE REQUIREMENT. The kernel holds three input planes at once (z-1, z, z+1)
 * plus writes an output plane, so slots must be >= 4. A smaller cache is
 * REJECTED rather than run: with slots < 4 a plane the kernel is still reading
 * can be evicted under it, which would not crash but would silently read
 * whatever replaced it.
 *
 * Correctness is a checksum of the final field, to be compared across
 * configurations with a RELATIVE TOLERANCE -- the reduction is order-dependent
 * in the same way k-means' centroid sums are.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cmath>
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

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_GS_CORO 1
#endif

/** Initial condition: v seeded in a centred cube, u elsewhere. Deterministic,
 *  so every configuration starts from the identical field. */
CTP_INLINE_CROSS_FUN float InitU(u64 x, u64 y, u64 z, u64 nx, u64 ny, u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.5f : 1.0f;
}
CTP_INLINE_CROSS_FUN float InitV(u64 x, u64 y, u64 z, u64 nx, u64 ny, u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.25f : 0.0f;
}

#if defined(GV_GS_CORO)
/** Seed u and v for this block's z-range, one plane (= one page) at a time. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 ubase, u64 vbase) {
  u64 run = 0;
  for (u64 z = z0; z < z1; ++z) {
    co_await vec.HoldPageCoro(ubase + z * plane, plane, &run);
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      vec[ubase + z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) vec.BeginFlush(ubase + z * plane, plane);
    __syncthreads();

    co_await vec.HoldPageCoro(vbase + z * plane, plane, &run);
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      vec[vbase + z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) vec.BeginFlush(vbase + z * plane, plane);
    __syncthreads();
  }
}

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> vec, u64 plane, u64 nx,
                           u64 ny, u64 nz, u64 zper, u64 ubase, u64 vbase,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
  CLIO_YCORO_RUN(SeedCoro(vec, plane, nx, ny, nz, z0, z1, ubase, vbase));
}

/**
 * One Gray-Scott step over this block's z-range.
 *
 * Holds z-1, z and z+1 for BOTH fields, then the output plane. The holds are
 * issued back to back so all of them are resident together -- which is why
 * slots >= 4 is enforced on the host. Boundary planes (z=0, z=nz-1) are copied
 * through rather than computed, the usual fixed-boundary treatment.
 */
__device__ gy::YCoroMain StepCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 ubase, u64 vbase, u64 unext, u64 vnext,
                                  float Du, float Dv, float F, float K,
                                  float dt) {
  u64 run = 0;
  for (u64 z = z0; z < z1; ++z) {
    const bool interior = (z > 0 && z + 1 < nz);
    const u64 zm = interior ? (z - 1) : z;
    const u64 zp = interior ? (z + 1) : z;

    // Three input planes of u, then three of v. Held together: the sliding
    // window means z and z+1 are needed again on the next iteration, which is
    // the reuse this benchmark measures.
    co_await vec.HoldPageCoro(ubase + zm * plane, plane, &run);
    co_await vec.HoldPageCoro(ubase + z * plane, plane, &run);
    co_await vec.HoldPageCoro(ubase + zp * plane, plane, &run);
    co_await vec.HoldPageCoro(vbase + zm * plane, plane, &run);
    co_await vec.HoldPageCoro(vbase + z * plane, plane, &run);
    co_await vec.HoldPageCoro(vbase + zp * plane, plane, &run);
    co_await vec.HoldPageCoro(unext + z * plane, plane, &run);
    co_await vec.HoldPageCoro(vnext + z * plane, plane, &run);

    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const u64 x = i % nx, y = i / nx;
      const float u = vec.at(ubase + z * plane + i);
      const float v = vec.at(vbase + z * plane + i);
      float lu, lv;
      if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
        lu = 0.0f; lv = 0.0f;      // fixed boundary
      } else {
        lu = vec.at(ubase + z * plane + i - 1) +
             vec.at(ubase + z * plane + i + 1) +
             vec.at(ubase + z * plane + i - nx) +
             vec.at(ubase + z * plane + i + nx) +
             vec.at(ubase + zm * plane + i) +
             vec.at(ubase + zp * plane + i) - 6.0f * u;
        lv = vec.at(vbase + z * plane + i - 1) +
             vec.at(vbase + z * plane + i + 1) +
             vec.at(vbase + z * plane + i - nx) +
             vec.at(vbase + z * plane + i + nx) +
             vec.at(vbase + zm * plane + i) +
             vec.at(vbase + zp * plane + i) - 6.0f * v;
      }
      const float uvv = u * v * v;
      vec[unext + z * plane + i] = u + dt * (Du * lu - uvv + F * (1.0f - u));
      vec[vnext + z * plane + i] = v + dt * (Dv * lv + uvv - (F + K) * v);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      vec.BeginFlush(unext + z * plane, plane);
      vec.BeginFlush(vnext + z * plane, plane);
    }
    __syncthreads();
  }
}

__global__ void StepKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> vec, u64 plane, u64 nx,
                           u64 ny, u64 nz, u64 zper, u64 ubase, u64 vbase,
                           u64 unext, u64 vnext, float Du, float Dv, float F,
                           float K, float dt, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
  CLIO_YCORO_RUN(StepCoro(vec, plane, nx, ny, nz, z0, z1, ubase, vbase, unext,
                          vnext, Du, Dv, F, K, dt));
}

/** Sum of v over this block's range, for the correctness checksum. */
__device__ gy::YCoroMain SumCoro(gv::DeviceVector<float> vec, u64 plane,
                                 u64 z0, u64 z1, u64 vbase, double *out) {
  u64 run = 0;
  for (u64 z = z0; z < z1; ++z) {
    co_await vec.HoldPageCoro(vbase + z * plane, plane, &run);
    double acc = 0.0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      acc += static_cast<double>(vec.at(vbase + z * plane + i));
    }
    atomicAdd(out, acc);
    __syncthreads();
  }
}

__global__ void SumKernel(clio::run::IpcManagerGpuInfo info,
                          gv::DeviceVector<float> vec, u64 plane, u64 nz,
                          u64 zper, u64 vbase, double *out,
                          gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
  CLIO_YCORO_RUN(SumCoro(vec, plane, z0, z1, vbase, out));
}
#else
__global__ void SeedKernel(clio::run::IpcManagerGpuInfo, gv::DeviceVector<float>,
                           u64, u64, u64, u64, u64, u64, u64,
                           gy::YieldableView<>, gy::YieldStackView) {}
__global__ void StepKernel(clio::run::IpcManagerGpuInfo, gv::DeviceVector<float>,
                           u64, u64, u64, u64, u64, u64, u64, u64, u64, float,
                           float, float, float, float, gy::YieldableView<>,
                           gy::YieldStackView) {}
__global__ void SumKernel(clio::run::IpcManagerGpuInfo, gv::DeviceVector<float>,
                          u64, u64, u64, u64, double *, gy::YieldableView<>,
                          gy::YieldStackView) {}
#endif  // GV_GS_CORO

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    // Both resets are required: RunToCompletion does not reset, so a reused
    // runner whose driver still reads "done" skips the launch entirely.
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
  u32 blocks = 64, threads = 256, slots = 8, steps = 4;
  u64 page_kb = 1024, data_mb = 16384, hbm_mb = 4096;
  int repeat = 3;
  bool hbm_only = false;
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 1.0f;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    auto nextf = [&]() -> float {
      return (i + 1 < argc) ? std::strtof(argv[++i], nullptr) : 0.0f;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--steps") steps = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--hbm-mb") hbm_mb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--Du") Du = nextf();
    else if (a == "--Dv") Dv = nextf();
    else if (a == "--F") F = nextf();
    else if (a == "--K") K = nextf();
    else if (a == "--dt") dt = nextf();
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--slots N] "
                  "[--steps N] [--page-kb N] [--data-mb N] [--hbm-mb N] "
                  "[--repeat N] [--hbm-only] [--Du f] [--Dv f] [--F f] "
                  "[--K f] [--dt f]\n", argv[0]);
      return 0;
    }
  }

#if !defined(GV_GS_CORO)
  std::fprintf(stderr,
               "GRAYSCOTT ERROR: built without C++20 device coroutines. This "
               "benchmark only implements the coroutine paging path. Rebuild "
               "with -DCLIO_GPU_YIELD_CORO=ON and CMAKE_CUDA_COMPILER="
               "clang++.\n");
  return 2;
#else
  // The kernel holds 6 input planes + 2 output planes at once. A smaller cache
  // could evict a plane the kernel is still reading -- that would not crash,
  // it would silently read whatever replaced it, so it is refused.
  const u32 kNeededSlots = 8;
  if (slots < kNeededSlots) {
    std::fprintf(stderr,
                 "GRAYSCOTT ERROR: slots=%u but the stencil holds %u planes at "
                 "once (z-1,z,z+1 of u and v, plus both outputs). A smaller "
                 "cache would let a plane still being read be evicted.\n",
                 slots, kNeededSlots);
    return 2;
  }

  const u64 page_bytes = page_kb * 1024;
  const u64 plane = page_bytes / sizeof(float);   // one page == one XY plane
  // Square-ish plane: nx*ny == plane, both powers of two.
  u64 nx = 1, ny = plane;
  while (nx * 2 <= ny) { nx *= 2; ny /= 2; }
  // Four regions (u, v, u_next, v_next) share the dataset budget.
  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  u64 nz = total_elems / (4 * plane);
  if (nz < 3) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: %lluMB at %lluKB pages leaves "
                 "nz=%llu planes; need at least 3 for a stencil.\n",
                 (unsigned long long)data_mb, (unsigned long long)page_kb,
                 (unsigned long long)nz);
    return 2;
  }
  const u64 zper = (nz + blocks - 1) / blocks;
  const u64 region = plane * nz;
  const u64 n = 4 * region;
  const double logical_mb =
      static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);

  {
    std::ofstream cfg("gv_grayscott_bench.yaml");
    cfg << "networking:\n  port: 9441\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        // MaxBwDpe sorts the preferred group DESCENDING and the vector puts at
        // blob score 1.0, so the HIGHER score is the preferred tier: HBM must
        // sit above the host tier.
        << "      - path: \"hbm::gv_gs_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: 1.0\n";
    if (!hbm_only) {
      cfg << "      - path: \"ram::gv_gs_ram\"\n        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << (data_mb + 1024) << "MB\"\n"
          << "        score: 0.2\n";
    }
    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_grayscott_bench.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  std::printf("Gray-Scott over a GPU vector\n"
              "  grid=%llux%llux%llu (page = one %lluKB plane)\n"
              "  blocks=%u (%llu planes each) threads=%u cache=%u pages/block\n"
              "  fields=4 (u,v,u_next,v_next)  total=%.0fMB  kHBM=%lluMB%s\n"
              "  steps=%u Du=%.3f Dv=%.3f F=%.3f K=%.3f dt=%.2f\n",
              (unsigned long long)nx, (unsigned long long)ny,
              (unsigned long long)nz, (unsigned long long)page_kb, blocks,
              (unsigned long long)zper, threads, slots, logical_mb,
              (unsigned long long)hbm_mb, hbm_only ? " (HBM ONLY)" : "",
              steps, Du, Dv, F, K, dt);

  gv::Vector<float> vec("gv_grayscott", {0}, page_bytes, blocks, slots, n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(blocks, threads);

  const u64 ubase = 0, vbase = region, unext = 2 * region, vnext = 3 * region;

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, plane, nx, ny, nz,
                                                zper, ubase, vbase, vw, sv);
  });
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: seed failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  double *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(double));

  double best_ms = 1e30, checksum = 0.0;
  for (int r = 0; r < repeat; ++r) {
    vec.ResetStats();
    cudaDeviceSynchronize();
    const double t0 = NowMs();
    u64 cu = ubase, cv = vbase, nu = unext, nv = vnext;
    for (u32 s = 0; s < steps; ++s) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        StepKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dev, plane, nx, ny, nz, zper, cu, cv, nu, nv, Du, Dv, F, K,
            dt, vw, sv);
      });
      std::swap(cu, nu);
      std::swap(cv, nv);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: step failed: %s\n",
                   cudaGetErrorString(cudaGetLastError()));
      return 1;
    }
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;

    cudaMemset(d_sum, 0, sizeof(double));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      SumKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, plane, nz, zper, cv,
                                                 d_sum, vw, sv);
    });
    cudaDeviceSynchronize();
    cudaMemcpy(&checksum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  }

  const auto st = vec.ReadStats(0);
  // Bytes touched per step: 6 input planes + 2 output planes per z.
  const double moved_gb =
      static_cast<double>(nz) * plane * sizeof(float) * 8.0 * steps /
      (1024.0 * 1024.0 * 1024.0);
  const double gbps = (best_ms > 0.0) ? moved_gb / (best_ms / 1000.0) : 0.0;

  std::fprintf(stderr,
               "GRAYSCOTT blocks=%u thr=%u nx=%llu ny=%llu nz=%llu "
               "page_kb=%llu slots=%u steps=%u data_mb=%.0f hbm_mb=%llu "
               "ms=%.1f GB/s=%.2f v_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu\n",
               blocks, threads, (unsigned long long)nx, (unsigned long long)ny,
               (unsigned long long)nz, (unsigned long long)page_kb, slots,
               steps, logical_mb, (unsigned long long)hbm_mb, best_ms, gbps,
               checksum, (unsigned long long)st.faults,
               (unsigned long long)st.evicts, (unsigned long long)st.puts,
               (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors);

  cudaFree(d_sum);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
#endif  // GV_GS_CORO
}

#endif  // !CTP_IS_DEVICE_PASS
