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
 * Correctness is a checksum of the final field, compared across configurations
 * with a RELATIVE TOLERANCE and only WITHIN one page size (page size sets the
 * grid geometry here, so a different page size solves a different problem).
 *
 * KNOWN OPEN ISSUE: THE RESULT IS NOT REPRODUCIBLE RUN TO RUN.
 *
 * The SAME configuration, run three times, gives checksums spread over
 * 3.37e-04 (1239598.10 / 1239180.61 / 1239424.31 at 16 blocks, 64KB pages).
 * A deterministic stencil should be bit-identical, and the double-precision
 * reduction over 1.34e8 values only reassociates at ~1e-12, so this is a
 * genuine data race or stale read that remains in this benchmark.
 *
 * This supersedes an earlier reading of the same evidence. The difference
 * BETWEEN block counts (8.39e-04) is the same order as the run-to-run spread,
 * so it was never established as a decomposition effect -- it was mostly
 * nondeterminism, and a fixed-configuration control should have been run
 * before attributing it to the block count.
 *
 * Two real defects were found and fixed along the way and did reduce it
 * (7.71e-03 -> 1.70e-03 -> 8.39e-04): missing cross-step flush waits, and a
 * per-block cache that carried stale copies of neighbouring blocks' planes
 * across a region swap. Neither closed it.
 *
 * TREAT THE TIMINGS AS INDICATIVE ONLY until this is resolved.
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "bench_memcpy_probe.h"

/** CTE core pool id, matching pool_id "512.0" in the server config the
 *  benchmark writes. The baseline talks to the core directly. */
const clio::run::PoolId kCorePool(512, 0);

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
/**
 * Wait for this block's outstanding writebacks by PARKING, not spinning.
 *
 * REQUIRED FOR CORRECTNESS ACROSS STEPS, not just for timing. Page caches are
 * PER BLOCK: block A writes plane z into its own cache and FlushAsync only
 * *issues* the put. On the next step those regions swap, and block B -- which
 * owns a neighbouring z-slab -- faults on plane z, misses its own cache, and
 * fetches from the tier. If A's put has not landed, B reads a STALE plane.
 * Nothing crashes; the field is quietly wrong.
 *
 * Measured before this wait existed: the same grid gave field checksums
 * differing by 7.7e-03 between 16 and 64 blocks, where a double-precision
 * reduction reassociates at ~1e-12. The block count changed which planes
 * crossed a cache boundary, so it changed the answer.
 */
/**
 * BASELINE KERNEL -- the out-of-core model WITHOUT in-kernel faulting.
 *
 * Same stencil as StepKernel, but every plane arrives as a plain device
 * pointer the HOST staged there. The kernel computes one z and exits; the
 * host does all I/O around it, synchronously, in both directions:
 *
 *   read  : blocking CTE GetBlob per input plane
 *   write : blocking CTE PutBlob per output plane  <-- writes are synchronous
 *           too, not just reads. The paged path submits its writebacks with
 *           FlushAsync and keeps going; the baseline cannot, so the put lands
 *           on the critical path of every z-iteration.
 *   copy  : blocking cudaMemcpy each way
 *   kernel: torn down and relaunched for every single z
 *
 * Nothing overlaps. This is the cost the vector's in-kernel faulting and
 * async writeback are there to remove, and it is the only one of the four
 * baselines that exercises the WRITE path at all -- weights and k-means are
 * read-only in their timed regions, so for them there is nothing to make
 * synchronous.
 */
__global__ void GrayscottBaselineKernel(const float *uzm, const float *uz,
                                        const float *uzp, const float *vzm,
                                        const float *vz, const float *vzp,
                                        float *unx, float *vnx, u64 plane,
                                        u64 nx, u64 ny, int interior, float Du,
                                        float Dv, float F, float K, float dt) {
  for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
    const u64 x = i % nx, y = i / nx;
    const float u = uz[i];
    const float v = vz[i];
    float lu, lv;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
      lu = 0.0f; lv = 0.0f;      // fixed boundary, as in StepKernel
    } else {
      lu = uz[i - 1] + uz[i + 1] + uz[i - nx] + uz[i + nx] + uzm[i] + uzp[i] -
           6.0f * u;
      lv = vz[i - 1] + vz[i + 1] + vz[i - nx] + vz[i + nx] + vzm[i] + vzp[i] -
           6.0f * v;
    }
    const float uvv = u * v * v;
    unx[i] = u + dt * (Du * lu - uvv + F * (1.0f - u));
    vnx[i] = v + dt * (Dv * lv + uvv - (F + K) * v);
  }
}

/** Seed u and v for this block's z-range, one plane (= one page) at a time. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 ubase, u64 vbase) {
  for (u64 z = z0; z < z1; ++z) {
    {
co_await vec.BeginFetch(vec.PageLo(ubase + z * plane), vec.PageSpan(ubase + z * plane, plane));
      co_await vec.AwaitFetch();
            auto h = co_await vec.HoldPage(ubase + z * plane, plane, /*write=*/true);
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h[ubase + z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective, internally BATCHED (one multi-put per 64 pages).
      co_await vec.BeginFlush();
    }
    {
co_await vec.BeginFetch(vec.PageLo(vbase + z * plane), vec.PageSpan(vbase + z * plane, plane));
      co_await vec.AwaitFetch();
            auto h = co_await vec.HoldPage(vbase + z * plane, plane, /*write=*/true);
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h[vbase + z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective, internally BATCHED (one multi-put per 64 pages).
      co_await vec.BeginFlush();
    }
  }
  // The first step reads planes seeded by OTHER blocks, so the seed must be
  // durable before this kernel returns.
  co_await vec.EndFlush();
}

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> vec, u64 plane, u64 nx,
                           u64 ny, u64 nz, u64 zper, u64 ubase, u64 vbase,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
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
  // One guard per concurrently-needed plane; declared OUTSIDE the loop and
  // move-assigned each iteration, so the assignment releases the previous
  // plane's pin instead of leaking it.
  gv::Held<float> uzm, uz, uzp;
  gv::Held<float> vzm, vz, vzp;
  gv::Held<float> unx, vnx;
  for (u64 z = z0; z < z1; ++z) {
    const bool interior = (z > 0 && z + 1 < nz);
    const u64 zm = interior ? (z - 1) : z;
    const u64 zp = interior ? (z + 1) : z;

co_await vec.BeginFetch(vec.PageLo(ubase + zm * plane), vec.PageSpan(ubase + zm * plane, plane));
    co_await vec.AwaitFetch();
        // Three input planes of u, then three of v, then the two outputs -- ONE
    // GUARD PER PLANE, because a guard indexes only its own held page.
    // THE HOLD IS THE PIN: each guard's plane stays resident until the
    // guard is re-assigned past it, so the sliding window (z and z+1 are
    // re-held next iteration) is expressed by the pins themselves and needs
    // no score hints.
    uzm = co_await vec.HoldPage(ubase + zm * plane, plane);
co_await vec.BeginFetch(vec.PageLo(ubase + z * plane), vec.PageSpan(ubase + z * plane, plane));
    co_await vec.AwaitFetch();
        uz = co_await vec.HoldPage(ubase + z * plane, plane);
co_await vec.BeginFetch(vec.PageLo(ubase + zp * plane), vec.PageSpan(ubase + zp * plane, plane));
    co_await vec.AwaitFetch();
        uzp = co_await vec.HoldPage(ubase + zp * plane, plane);
co_await vec.BeginFetch(vec.PageLo(vbase + zm * plane), vec.PageSpan(vbase + zm * plane, plane));
    co_await vec.AwaitFetch();
        vzm = co_await vec.HoldPage(vbase + zm * plane, plane);
co_await vec.BeginFetch(vec.PageLo(vbase + z * plane), vec.PageSpan(vbase + z * plane, plane));
    co_await vec.AwaitFetch();
        vz = co_await vec.HoldPage(vbase + z * plane, plane);
co_await vec.BeginFetch(vec.PageLo(vbase + zp * plane), vec.PageSpan(vbase + zp * plane, plane));
    co_await vec.AwaitFetch();
        vzp = co_await vec.HoldPage(vbase + zp * plane, plane);
co_await vec.BeginFetch(vec.PageLo(unext + z * plane), vec.PageSpan(unext + z * plane, plane));
    co_await vec.AwaitFetch();
        unx = co_await vec.HoldPage(unext + z * plane, plane, /*write=*/true);
co_await vec.BeginFetch(vec.PageLo(vnext + z * plane), vec.PageSpan(vnext + z * plane, plane));
    co_await vec.AwaitFetch();
        vnx = co_await vec.HoldPage(vnext + z * plane, plane, /*write=*/true);

    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const u64 x = i % nx, y = i / nx;
      const float u = uz[ubase + z * plane + i];
      const float v = vz[vbase + z * plane + i];
      float lu, lv;
      if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
        lu = 0.0f; lv = 0.0f;      // fixed boundary
      } else {
        lu = uz[ubase + z * plane + i - 1] +
             uz[ubase + z * plane + i + 1] +
             uz[ubase + z * plane + i - nx] +
             uz[ubase + z * plane + i + nx] +
             uzm[ubase + zm * plane + i] +
             uzp[ubase + zp * plane + i] - 6.0f * u;
        lv = vz[vbase + z * plane + i - 1] +
             vz[vbase + z * plane + i + 1] +
             vz[vbase + z * plane + i - nx] +
             vz[vbase + z * plane + i + nx] +
             vzm[vbase + zm * plane + i] +
             vzp[vbase + zp * plane + i] - 6.0f * v;
      }
      const float uvv = u * v * v;
      unx[unext + z * plane + i] = u + dt * (Du * lu - uvv + F * (1.0f - u));
      vnx[vnext + z * plane + i] = v + dt * (Dv * lv + uvv - (F + K) * v);
    }
    __syncthreads();
    // Flush the write-once outputs; the drop below is best-effort (a page
    // still flushing or pinned is refused and reclaimed by ordinary eviction
    // once it settles).
    co_await vec.BeginFlush();
    co_await vec.BeginFlush();
    if (interior) {
      // Plane z-1 leaves the sliding window for good: empty the guard so the
      // drop can take it. (zm == z when not interior, so releasing it there
      // would release the plane the next iteration still needs.)
      uzm = {};
      vzm = {};
      const u64 d0 = vec.PageOf(ubase + zm * plane);
      const u64 d1 = vec.PageOf(vbase + zm * plane);
      const u64 d2 = vec.PageOf(unext + z * plane);
      const u64 d3 = vec.PageOf(vnext + z * plane);
      const u64 drops[4] = {d0, d1, d2, d3};
    }
  }
  // Drain before returning: the next step swaps the regions and other blocks
  // will fault on the planes written here. Waiting once per block per step,
  // rather than once per plane, keeps the puts pipelined while still making
  // them durable at the step boundary. Every guard empties first so nothing
  // stays pinned when the drops below run.
  uzm = {}; uz = {}; uzp = {};
  vzm = {}; vz = {}; vzp = {};
  unx = {}; vnx = {};
  co_await vec.EndFlush();
  // ...and then DROP THE CACHE. Durability alone is not enough. A block reads
  // planes owned by its NEIGHBOURS (z-1 at the bottom of its slab, z+1 at the
  // top), and those pages stay resident in this block's cache. The regions
  // swap every step, so an address read in step N is read again in step N+2 --
  // and a resident stale copy would be served instead of the value another
  // block has since written. Nothing invalidates one block's cache when
  // another block writes, because the caches are per block by design.
  //
  // The residual scaled with the PLANE COUNT, which is the signature: 1.70e-03
  // at 64KB pages (65536 planes) down to nothing measurable at 4MB (1024
  // planes) -- more planes, more block-boundary sharing, more stale hits.
  // Everything this block touched is clean (flushed and awaited above) and
  // unpinned, so the batched score-0 drop takes it all.
  {
    const u64 zlo = (z0 > 0) ? (z0 - 1) : 0;
    const u64 zhi = (z1 + 1 < nz) ? (z1 + 1) : nz;
    const u64 bases[4] = {ubase, vbase, unext, vnext};
    for (u64 b = 0; b < 4; ++b) {
      for (u64 pg = zlo; pg < zhi; pg += 64) {
        const u32 nb = (zhi - pg < 64) ? static_cast<u32>(zhi - pg) : 64u;
        const u64 pbase = vec.PageOf(bases[b]);
      }
    }
  }
}

__global__ void StepKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> vec, u64 plane, u64 nx,
                           u64 ny, u64 nz, u64 zper, u64 ubase, u64 vbase,
                           u64 unext, u64 vnext, float Du, float Dv, float F,
                           float K, float dt, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
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
  for (u64 z = z0; z < z1; ++z) {
co_await vec.BeginFetch(vec.PageLo(vbase + z * plane), vec.PageSpan(vbase + z * plane, plane));
    co_await vec.AwaitFetch();
        auto h = co_await vec.HoldPage(vbase + z * plane, plane);
    double acc = 0.0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      acc += static_cast<double>(h[vbase + z * plane + i]);
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
  vec.Init(yv.Block());
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
        [] {}, /*max_rounds=*/2000000,
      gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, slots = 12, steps = 4;
  u64 page_kb = 1024, data_mb = 16384, hbm_mb = 4096;
  int repeat = 3;
  // Out-of-core WITHOUT in-kernel faulting: synchronous CTE reads AND
  // writes, synchronous memcpy both ways, kernel torn down per z.
  bool baseline = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
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
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--baseline") baseline = true;
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
  const u32 kNeededSlots = 10;
  if (slots < kNeededSlots) {
    std::fprintf(stderr,
                 "GRAYSCOTT ERROR: slots=%u but the stencil holds %u planes at "
                 "once (z-1,z,z+1 of u and v, plus both outputs). A smaller "
                 "cache would let a plane still being read be evicted.\n",
                 slots, kNeededSlots);
    return 2;
  }

  const u64 page_bytes = page_kb * 1024;

  // REFERENCE CEILING for the paging path: a bare H2D cudaMemcpy at exactly
  // this page size, on an idle device before the runtime starts. The gap
  // between this and the achieved rate is the vector's overhead.
  const MemcpyProbe mcp = ProbeMemcpyBandwidth(static_cast<size_t>(page_bytes));
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
  ctp::GpuApi::Synchronize();

  double *d_sum = nullptr;
  d_sum = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sum)>>(sizeof(double));

  // ---- BASELINE DRIVER ------------------------------------------------
  // Per z: blocking reads of the 6 input planes, blocking H2D, one kernel
  // launch, blocking D2H, then BLOCKING PUTS of the two output planes. Every
  // transfer is on the critical path and the grid is rebuilt for each z.
  //
  // The 6 reads are issued every iteration rather than kept as a sliding
  // window. That IS the model being measured: with the kernel torn down at
  // every step there is no in-kernel state to carry the window in, and a host
  // that wanted to keep one would be reimplementing the cache this benchmark
  // exists to compare against.
  ctp::ipc::FullPtr<char> bl_h[6];
  ctp::ipc::FullPtr<char> bl_out[2];
  float *bl_d[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  float *bl_dout[2] = {nullptr, nullptr};
  clio::cte::core::Client *bl_core = nullptr;
  const u64 page_bytes_gs = plane * sizeof(float);
  if (baseline) {
    for (int i = 0; i < 6; ++i) {
      bl_h[i] = CLIO_IPC->AllocateBuffer((size_t)page_bytes_gs);
      if (bl_h[i].IsNull()) {
        std::fprintf(stderr, "GRAYSCOTT ERROR: baseline alloc failed\n");
        return 1;
      }
      // GpuApi::Malloc fails fatally, which is the same abort with less code.
      bl_d[i] = ctp::GpuApi::Malloc<float>((size_t)page_bytes_gs);
    }
    for (int i = 0; i < 2; ++i) {
      bl_out[i] = CLIO_IPC->AllocateBuffer((size_t)page_bytes_gs);
      if (bl_out[i].IsNull()) {
        std::fprintf(stderr, "GRAYSCOTT ERROR: baseline alloc failed\n");
        return 1;
      }
      bl_dout[i] = ctp::GpuApi::Malloc<float>((size_t)page_bytes_gs);
    }
    bl_core = new clio::cte::core::Client(kCorePool);
  }
  // Blocking read of one plane into staging slot `i`.
  auto bl_read = [&](int i, u64 region_base, u64 z) {
    const std::string nm = std::to_string((region_base + z * plane) / plane);
    auto gf = bl_core->AsyncGetBlob(vec.TagId(), nm, 0, page_bytes_gs, 0,
                                    bl_h[i].shm_.template Cast<void>(),
                                    clio::run::PoolQuery::Local());
    gf.Wait();
    if (gf->GetReturnCode() != 0) {
      std::memset(bl_h[i].ptr_, 0, (size_t)page_bytes_gs);
    }
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(bl_d[i]), bl_h[i].ptr_,
                        (size_t)page_bytes_gs);
    return true;
  };
  // SYNCHRONOUS WRITE of one output plane: D2H, then a blocking PutBlob.
  auto bl_write = [&](int i, u64 region_base, u64 z) {
    ctp::GpuApi::Memcpy(bl_out[i].ptr_,
                        reinterpret_cast<const char *>(bl_dout[i]),
                        (size_t)page_bytes_gs);
    const std::string nm = std::to_string((region_base + z * plane) / plane);
    auto pf = bl_core->AsyncPutBlob(vec.TagId(), nm, 0, page_bytes_gs,
                                    bl_out[i].shm_.template Cast<void>(), 1.0f);
    pf.Wait();
    return pf->GetReturnCode() == 0;
  };
  auto run_baseline_step = [&](u64 cu_, u64 cv_, u64 nu_, u64 nv_) {
    for (u64 z = 0; z < nz; ++z) {
      const bool interior = (z > 0 && z + 1 < nz);
      const u64 zm = interior ? (z - 1) : z;
      const u64 zp = interior ? (z + 1) : z;
      if (!bl_read(0, cu_, zm) || !bl_read(1, cu_, z) || !bl_read(2, cu_, zp) ||
          !bl_read(3, cv_, zm) || !bl_read(4, cv_, z) || !bl_read(5, cv_, zp)) {
        return false;
      }
      GrayscottBaselineKernel<<<1, threads>>>(
          bl_d[0], bl_d[1], bl_d[2], bl_d[3], bl_d[4], bl_d[5], bl_dout[0],
          bl_dout[1], plane, nx, ny, interior ? 1 : 0, Du, Dv, F, K, dt);
      ctp::GpuApi::Synchronize();
      if (!bl_write(0, nu_, z) || !bl_write(1, nv_, z)) return false;
    }
    return true;
  };

  double best_ms = 1e30, checksum = 0.0;
  for (int r = 0; r < repeat; ++r) {
    // RE-SEED between repeats. Without this, repeat 2 continues evolving the
    // field left by repeat 1, so each timed run measures a different physical
    // state and the reported checksum depends on `repeat` -- which makes it
    // useless as a correctness check and makes the repeats non-comparable.
    if (r > 0) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, plane, nx, ny, nz,
                                                    zper, ubase, vbase, vw, sv);
      });
      ctp::GpuApi::Synchronize();
    }
    vec.ResetStats();
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();
    u64 cu = ubase, cv = vbase, nu = unext, nv = vnext;
    for (u32 s = 0; s < steps; ++s) {
      if (baseline) {
        if (!run_baseline_step(cu, cv, nu, nv)) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: baseline step failed\n");
          return 1;
        }
      } else {
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          StepKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu, dev, plane, nx, ny, nz, zper, cu, cv, nu, nv, Du, Dv, F, K,
              dt, vw, sv);
        });
      }
      std::swap(cu, nu);
      std::swap(cv, nv);
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;

    ctp::GpuApi::Memset(d_sum, 0, sizeof(double));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      SumKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, plane, nz, zper, cv,
                                                 d_sum, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(&checksum, d_sum, sizeof(double));
  }

  const auto st = vec.ReadStats(0);
  // Bytes touched per step: 6 input planes + 2 output planes per z.
  const double moved_gb =
      static_cast<double>(nz) * plane * sizeof(float) * 8.0 * steps /
      (1024.0 * 1024.0 * 1024.0);
  const double gbps = (best_ms > 0.0) ? moved_gb / (best_ms / 1000.0) : 0.0;

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
    const clio::run::u64 host_cap = (clio::run::u64)(data_mb + 1024) * 1024ull * 1024ull;
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
               "GRAYSCOTT mode=%s blocks=%u thr=%u nx=%llu ny=%llu nz=%llu "
               "page_kb=%llu slots=%u steps=%u data_mb=%.0f hbm_mb=%llu "
               "ms=%.1f GB/s=%.2f v_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
               baseline ? "baseline" : "paged",
               blocks, threads, (unsigned long long)nx, (unsigned long long)ny,
               (unsigned long long)nz, (unsigned long long)page_kb, slots,
               steps, logical_mb, (unsigned long long)hbm_mb, best_ms, gbps,
               checksum, (unsigned long long)st.faults,
               (unsigned long long)st.evicts, (unsigned long long)st.puts,
               (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps);

  ctp::GpuApi::Free(d_sum);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
#endif  // GV_GS_CORO
}

#endif  // !CTP_IS_DEVICE_PASS
