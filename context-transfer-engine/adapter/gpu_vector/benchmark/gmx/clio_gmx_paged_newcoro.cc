#if CTP_ENABLE_SYCL
#define CLIO_SYCL_KERNEL_TU 1
#endif
/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * GROMACS science kernel -- PME charge spreading and force-stage gathering --
 * over a GPU vector whose GRID does not fit on the device.
 *
 * WHY THIS KERNEL. The md benchmark already covers the short-range pair
 * loop; what GROMACS adds that no other benchmark here has is PME: N atoms
 * SCATTER onto a K^3 mesh through an order-4 B-spline (each atom touches a
 * 4x4x4 neighbourhood), and the mesh is then read back at atom positions.
 * The mesh is the thing that outgrows VRAM, and spreading is the one PME
 * stage with a local, streamable access pattern -- the FFT that follows in
 * real PME needs the whole mesh at once, which is exactly the consumer that
 * cannot page (measured on GROMACS itself; see the eternia notes), so this
 * benchmark ends where the FFT would begin.
 *
 * DECOMPOSITION: ONE WRITER PER PAGE, BY CONSTRUCTION. A page is one XY
 * plane; a CUDA block owns a contiguous z-range of planes and PULLS into
 * each: the atoms whose spline base lands in bins z-3..z are exactly the
 * ones that touch plane z, so the block visits plane z once, accumulates
 * every contribution, publishes, and moves on. Two blocks never write one
 * page, so the write-site flush is sound under eviction -- the same rule the
 * md resort's gather learned (two blocks sharing a page silently clobber
 * each other under page-granular writeback).
 *
 * FIXED-POINT, FOR A DIGIT-EXACT GATE. Atoms within one plane still collide
 * on grid points, and float atomics make the sum order-dependent -- a
 * tolerance gate would then hide real staleness bugs inside "atomics
 * noise". Charge is therefore accumulated in Q40.24 fixed point on an
 * unsigned-64 mesh: integer addition commutes, so the paged run must match
 * the dense in-VRAM reference BIT FOR BIT, total charge must equal the sum
 * of input charges EXACTLY, and any lost or stale page shows as a hard
 * mismatch rather than a plausible wobble.
 *
 * GATES (all exact, no tolerances):
 *   CONSERVATION  sum over mesh == sum of input charges, in fixed point
 *   MESH          64-bit checksum of the paged mesh == dense reference mesh
 *   GATHER        fixed-point interpolation energy == dense reference
 *
 * OUT OF CORE: --cap M caps the mesh cache at M pages. The spread's window
 * is self-limiting (a block holds ONE plane at a time; the gather holds
 * four), so the floor is small and pressure means eviction of published
 * planes and refaults on the gather pass -- Fetch/Flush consistency, not
 * luck, is what the exact gates certify.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

static constexpr u32 kYieldLaneBytes = 1024;


/*
 * THE DEVICE CODE IS NOT HERE ANY MORE.
 *
 * The workload -- the atom generator, the spline weights, the four
 * coroutines and the dense reference bodies -- lives in gmx_kernels.h (over
 * gmx_math.h), in ONE copy compiled by both backends. The launches live in
 * cuda/ and sycl/ and differ only in how a grid is submitted. This file is
 * the host driver and is now ordinary C++.
 *
 * See gmx_launch.h for why the seam is at the launch.
 */
#include "../gv_launch_bounds.h"
#include "gmx_kernels.h"
#include "gmx_launch.h"

namespace gx = clio::gv_bench::gmx;
using gx::Frac01;
using gx::kFxScale;
using gx::Lcg;
using gx::Spline4;

#if CTP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

/* =====================================================
 * THE WORKLOAD, on the new coroutine API. Ordinary return
 * types, ordinary locals, one marker per suspending call.
 * The `, clio::co::Ctx &_cy` on each signature and at each
 * call site is appended by clio-coroc, not written here.
 * ===================================================== */
namespace clio::gv_bench::gmx {

/**
 * Spread this block's planes. For plane z the contributing atoms are those
 * whose spline base bin b satisfies b <= z <= b+3, i.e. bins z-3..z (mod K).
 * The plane is fetched, written by THIS BLOCK ONLY, published, released.
 * Partial sums are durable at every unpin: if the plane is evicted between
 * two owner visits (it is not -- one visit per plane -- but the rule is what
 * matters), the refault reads the published partials back and accumulation
 * continues, which is the write-site-publish contract the md workload
 * bled for.
 */
CTP_GPU_FUN CLIO_COROC_INLINE void SpreadCoro(gv::DeviceVector<unsigned long long> mesh,
                                    const float *ax, const float *ay,
                                    const float *az, const long long *aq,
                                    const u32 *bin_start, u64 K, u64 plane,
                                    u64 z0, u64 z1) {
  for (u64 z = z0; z < z1; ++z) {
    CO_AWAIT(mesh.CoFetch(0, z * plane, plane));
    auto h = CO_AWAIT(mesh.CoHoldPage(z * plane, plane, /*write=*/true));
    // Four source bins feed plane z; threads stride the atoms of each bin.
    for (int db = -3; db <= 0; ++db) {
      const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
      const u32 a0 = bin_start[b];
      const u32 a1 = bin_start[b + 1];
      for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const float x = ax[a];
        const float y = ay[a];
        const float zz = az[a];
        const int ix0 = static_cast<int>(floorf(x)) - 1;
        const int iy0 = static_cast<int>(floorf(y)) - 1;
        // Which of the atom's four z-nodes is THIS plane? The bin choice
        // already guarantees (z - b) mod K lands in 0..3.
        const int dzw = static_cast<int>((z + K - b) % K);
        float wx[4];
        float wy[4];
        float wz[4];
        Spline4(x - floorf(x), wx);
        Spline4(y - floorf(y), wy);
        Spline4(zz - floorf(zz), wz);
        // EXACTLY-CONSERVING SPLIT. Rounding each of the 64 pieces
        // independently leaves a per-atom residue (measured: 809 fixed-point
        // units over 200k atoms), so the LAST piece at each level absorbs
        // the remainder: the four z-pieces sum to q exactly, and the 16
        // xy-pieces of each z-piece sum to it exactly. Deterministic, so
        // the dense reference computes the identical values.
        long long qz4[4];
        {
          long long run = 0;
          for (int j = 0; j < 3; ++j) {
            qz4[j] = static_cast<long long>(
                FxRound(static_cast<double>(aq[a]) * wz[j]));
            run += qz4[j];
          }
          qz4[3] = aq[a] - run;
        }
        const long long qz = qz4[dzw];
        long long xy_run = 0;
        for (int jy = 0; jy < 4; ++jy) {
          const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                           static_cast<int>(K));
          for (int jx = 0; jx < 4; ++jx) {
            const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                            static_cast<int>(K));
            const long long v =
                (jy == 3 && jx == 3)
                    ? qz - xy_run
                    : static_cast<long long>(FxRound(
                          static_cast<double>(qz) * wy[jy] * wx[jx]));
            xy_run += v;
            atomicAdd(reinterpret_cast<unsigned long long *>(
                          &h[z * plane + gy_ * K + gx]),
                      static_cast<unsigned long long>(v));
          }
        }
      }
      __syncthreads();
    }
    // PUBLISH AT THE WRITE SITE, then release. One writer per page makes
    // this ordering sound; eviction after the unpin costs a refault, never
    // data.
    CO_AWAIT(mesh.CoBeginFlush(0, z * plane, plane));
    mesh.UnpinRange(z * plane, plane);
  }
  CO_AWAIT(mesh.CoEndFlush());
}

/** Mesh checksum + exact charge total, striding planes across blocks. */
CTP_GPU_FUN CLIO_COROC_INLINE void SumCoro(gv::DeviceVector<unsigned long long> mesh,
                                 u64 K, u64 plane, u64 z0, u64 z1,
                                 unsigned long long *out) {
  for (u64 z = z0; z < z1; ++z) {
    CO_AWAIT(mesh.CoFetch(0, z * plane, plane));
    auto h = CO_AWAIT(mesh.CoHoldPage(z * plane, plane, /*write=*/false));
    unsigned long long q = 0, ck = 0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const unsigned long long v = h[z * plane + i];
      q += v;
      // Position-dependent mixing so a value landing on the WRONG grid
      // point cannot cancel: checksum(v at i) != checksum(v at j).
      ck += v * (2ull * (z * plane + i) + 1ull);
    }
    atomicAdd(&out[0], q);
    atomicAdd(&out[1], ck);
    __syncthreads();
    mesh.UnpinRange(z * plane, plane);
  }
}

/**
 * The force-stage read pattern: interpolate the mesh back at every atom
 * position (the potential/force gather of PME, minus the convolution the
 * FFT would have applied). A block owns atom bins; an atom's stencil spans
 * planes iz0..iz0+3, held as a sliding window exactly like the md force
 * stencil. Accumulation is fixed point again, so the result is bit-equal to
 * the dense path.
 */
CTP_GPU_FUN CLIO_COROC_INLINE void GatherCoro(gv::DeviceVector<unsigned long long> mesh,
                                    const float *ax, const float *ay,
                                    const float *az, const long long *aq,
                                    const u32 *bin_start, u64 K, u64 plane,
                                    u64 b0, u64 b1, unsigned long long *out) {
  gv::PageRef<unsigned long long> hz[4];
  unsigned long long acc = 0;
  for (u64 b = b0; b < b1; ++b) {
    // Atoms in bin b have iz0 == b: hold planes b..b+3 (mod K).
    for (int j = 0; j < 4; ++j) {
      const u64 z = (b + static_cast<u64>(j)) % K;
      CO_AWAIT(mesh.CoFetch(0, z * plane, plane));
      hz[j] = CO_AWAIT(mesh.CoHoldPage(z * plane, plane, /*write=*/false));
    }
    const u32 a0 = bin_start[b];
    const u32 a1 = bin_start[b + 1];
    for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
      const float x = ax[a];
      const float y = ay[a];
      const float zz = az[a];
      const int ix0 = static_cast<int>(floorf(x)) - 1;
      const int iy0 = static_cast<int>(floorf(y)) - 1;
      float wx[4];
      float wy[4];
      float wzS[4];
      Spline4(x - floorf(x), wx);
      Spline4(y - floorf(y), wy);
      Spline4(zz - floorf(zz), wzS);
      double phi = 0.0;
      for (int jz = 0; jz < 4; ++jz) {
        const u64 z = (b + static_cast<u64>(jz)) % K;
        double pl = 0.0;
        for (int jy = 0; jy < 4; ++jy) {
          const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                           static_cast<int>(K));
          double row = 0.0;
          for (int jx = 0; jx < 4; ++jx) {
            const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                            static_cast<int>(K));
            const long long v = static_cast<long long>(
                hz[jz][z * plane + gy_ * K + gx]);
            row += static_cast<double>(v) * wx[jx];
          }
          pl += row * wy[jy];
        }
        phi += pl * wzS[jz];
      }
      // q_a * phi(x_a), requantized: exact and order-independent.
      acc += static_cast<unsigned long long>(static_cast<long long>(
          FxRound(phi * (static_cast<double>(aq[a]) / kFxScale))));
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      const u64 z = (b + static_cast<u64>(j)) % K;
      hz[j] = {};
      mesh.UnpinRange(z * plane, plane);
    }
  }
  atomicAdd(out, acc);
}

/** Zero this block's planes and publish, so a fault after eviction reads
 *  zeros rather than "blob not found". */
CTP_GPU_FUN CLIO_COROC_INLINE void ZeroCoro(gv::DeviceVector<unsigned long long> mesh,
                                  u64 plane, u64 z0, u64 z1) {
  for (u64 z = z0; z < z1; ++z) {
    CO_AWAIT(mesh.CoFetch(0, z * plane, plane));
    auto h = CO_AWAIT(mesh.CoHoldPage(z * plane, plane, /*write=*/true));
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) h[z * plane + i] = 0;
    __syncthreads();
    CO_AWAIT(mesh.CoBeginFlush(0, z * plane, plane));
    mesh.UnpinRange(z * plane, plane);
  }
  CO_AWAIT(mesh.CoEndFlush());
}

}  // namespace clio::gv_bench::gmx

/* TWO BACKENDS, ONE WORKLOAD. Everything above this line is compiled for both: the transpiled state machine contains no vendor token. What differs is only how a grid is submitted. */
#if CTP_ENABLE_SYCL

namespace clio::gv_bench::gmx {

namespace {

template <typename BodyT>
void Submit(dim3 grid, dim3 block, BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>) { body(); })
      .wait();
}

template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevMesh mesh, View vw,
                     StackView sv, MakeCoro make) {
  Submit(grid, block, [=]() {
    DevMesh dev = mesh;
    dev.Init(vw.Block());
    __syncthreads();
    CLIO_COROC_RUN(vw, sv, make(_cy, dev, vw.Block()));
  });
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
}

void LaunchZero(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                u64 K, u64 plane, u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, mesh, vw, sv, [=](clio::co::Ctx &_cy, DevMesh dev, u32 blk) {
    const u64 z0 = zbase + static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    ZeroCoro(_cy, dev, plane, z0, z1);
  });
}

void LaunchSpread(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](clio::co::Ctx &_cy, DevMesh dev, u32 blk) {
    const u64 z0 = zbase + static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    SpreadCoro(_cy, dev, ax, ay, az, aq, bin_start, K, plane, z0, z1);
  });
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh, u64 K,
               u64 plane, u64 zper, unsigned long long *out, u64 zbase, u64 zend, View vw,
               StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](clio::co::Ctx &_cy, DevMesh dev, u32 blk) {
    const u64 z0 = zbase + static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    SumCoro(_cy, dev, K, plane, z0, z1, out);
  });
}

void LaunchGather(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 bper, unsigned long long *out, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](clio::co::Ctx &_cy, DevMesh dev, u32 blk) {
    const u64 b0 = zbase + static_cast<u64>(blk) * bper;
    const u64 b1 = (b0 + bper < zend) ? (b0 + bper) : zend;
    GatherCoro(_cy, dev, ax, ay, az, aq, bin_start, K, plane, b0, b1, out);
  });
}

void LaunchDenseSpread(u32 blocks, u32 threads, unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 zper) {
  Submit(dim3(blocks), dim3(threads), [=]() {
    DenseSpreadBody(mesh, ax, ay, az, aq, bin_start, K, plane, zper);
  });
}

void LaunchDenseSum(const unsigned long long *mesh, u64 n,
                    unsigned long long *out) {
  Submit(dim3(64), dim3(256), [=]() { DenseSumBody(mesh, n, out); });
}

void LaunchDenseGather(u32 blocks, u32 threads, const unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 bper, unsigned long long *out) {
  Submit(dim3(blocks), dim3(threads), [=]() {
    DenseGatherBody(mesh, ax, ay, az, aq, bin_start, K, plane, bper, out);
  });
}

}  // namespace clio::gv_bench::gmx

namespace clio::run::gpu {

/** The out-of-line half of YieldStack::Reset; see yield_stack.h. */
void SyclYieldStackReset(const YieldStackView &view, clio::run::u32 nlanes,
                         char *smem_base) {
  auto &q = ctp::GpuApi::SyclQueue();
  YieldStackView v = view;
  q.parallel_for(sycl::range<1>(nlanes), [=](sycl::id<1> i) {
     auto *h = reinterpret_cast<YieldLaneHeader *>(
         v.base_ + static_cast<clio::run::u64>(i[0]) * v.bytes_per_lane_);
     h->sp_ = sizeof(YieldLaneHeader);   // the header is not frame space
     h->live_depth_ = 0;
     h->cur_depth_ = 0;
     h->error_ = kYieldErrNone;
     h->coro_resume_ = 0;
     h->coro_top_ = 0;
     h->coro_park_ = 0;
   }).wait();
  char *base = smem_base;
  q.copy(&base, g_yield_smem_dg, 1).wait();
}

}  // namespace clio::run::gpu

#else  /* CUDA */

namespace clio::gv_bench::gmx {

namespace {

__global__ GV_LAUNCH_BOUNDS void ZeroKernel(GpuInfo info, DevMesh mesh, u64 K, u64 plane,
                           u64 zper, u64 zbase, u64 zend, View yv,
                           StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_COROC_RUN(yv, ys, ZeroCoro(_cy, mesh, plane, z0, z1));
}

__global__ GV_LAUNCH_BOUNDS void SpreadKernel(GpuInfo info, DevMesh mesh, const float *ax,
                             const float *ay, const float *az,
                             const long long *aq, const u32 *bin_start, u64 K,
                             u64 plane, u64 zper, u64 zbase, u64 zend, View yv,
                           StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_COROC_RUN(yv, ys, SpreadCoro(_cy, mesh, ax, ay, az, aq, bin_start, K, plane, z0, z1));
}

__global__ GV_LAUNCH_BOUNDS void SumKernel(GpuInfo info, DevMesh mesh, u64 K, u64 plane,
                          u64 zper, unsigned long long *out, u64 zbase, u64 zend, View yv,
                          StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_COROC_RUN(yv, ys, SumCoro(_cy, mesh, K, plane, z0, z1, out));
}

__global__ GV_LAUNCH_BOUNDS void GatherKernel(GpuInfo info, DevMesh mesh, const float *ax,
                             const float *ay, const float *az,
                             const long long *aq, const u32 *bin_start, u64 K,
                             u64 plane, u64 bper, unsigned long long *out,
                             u64 zbase, u64 zend, View yv,
                           StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  __syncthreads();
  const u64 b0 = zbase + static_cast<u64>(yv.Block()) * bper;
  const u64 b1 = (b0 + bper < zend) ? (b0 + bper) : zend;
  CLIO_COROC_RUN(yv, ys, GatherCoro(_cy, mesh, ax, ay, az, aq, bin_start, K, plane, b0, b1,
                            out));
}

__global__ GV_LAUNCH_BOUNDS void DenseSpreadKernel(unsigned long long *mesh, const float *ax,
                                  const float *ay, const float *az,
                                  const long long *aq, const u32 *bin_start,
                                  u64 K, u64 plane, u64 zper) {
  DenseSpreadBody(mesh, ax, ay, az, aq, bin_start, K, plane, zper);
}

__global__ GV_LAUNCH_BOUNDS void DenseSumKernel(const unsigned long long *mesh, u64 n,
                               unsigned long long *out) {
  DenseSumBody(mesh, n, out);
}

__global__ GV_LAUNCH_BOUNDS void DenseGatherKernel(const unsigned long long *mesh,
                                  const float *ax, const float *ay,
                                  const float *az, const long long *aq,
                                  const u32 *bin_start, u64 K, u64 plane,
                                  u64 bper, unsigned long long *out) {
  DenseGatherBody(mesh, ax, ay, az, aq, bin_start, K, plane, bper, out);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // CUDA's per-block IpcManager is __shared__ storage, born fresh at every
  // launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchZero(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                u64 K, u64 plane, u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  ZeroKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, K, plane, zper, zbase, zend, vw, sv);
}

void LaunchSpread(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  SpreadKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, ax, ay, az, aq, bin_start, K, plane, zper, zbase, zend,
      vw, sv);
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh, u64 K,
               u64 plane, u64 zper, unsigned long long *out, u64 zbase, u64 zend, View vw,
               StackView sv) {
  SumKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, K, plane, zper, out, zbase, zend, vw, sv);
}

void LaunchGather(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 bper, unsigned long long *out, u64 zbase, u64 zend, View vw,
                  StackView sv) {
  GatherKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, ax, ay, az, aq, bin_start, K, plane, bper, out, zbase,
      zend, vw, sv);
}

void LaunchDenseSpread(u32 blocks, u32 threads, unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 zper) {
  DenseSpreadKernel<<<blocks, threads>>>(mesh, ax, ay, az, aq, bin_start, K,
                                         plane, zper);
}

void LaunchDenseSum(const unsigned long long *mesh, u64 n,
                    unsigned long long *out) {
  DenseSumKernel<<<64, 256>>>(mesh, n, out);
}

void LaunchDenseGather(u32 blocks, u32 threads, const unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 bper, unsigned long long *out) {
  DenseGatherKernel<<<blocks, threads>>>(mesh, ax, ay, az, aq, bin_start, K,
                                         plane, bper, out);
}

}  // namespace clio::gv_bench::gmx

#endif  /* CTP_ENABLE_SYCL */

#if !CTP_IS_DEVICE_PASS

// Cross-node reduction. Included INSIDE the device-pass guard: it uses the
// CTE client, whose members are compiled out of the CUDA device pass.
#include "../bench_dist.h"

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

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
        [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 16, threads = 256, cap = 0;
  // Z-PLANE SLAB DECOMPOSITION, exactly the MPI edition's: the mesh is the
  // big object and the atoms are REPLICATED on every node. Spread and
  // gather are both decomposed by owner, so no halo is ever exchanged --
  // only the three integer gate totals cross the wire, and they are exact,
  // so the gates stay bit-equal at any node count.
  u32 nodes = 1, node = 0;
  // Skip the dense in-VRAM reference: a 6 GB-class mesh cannot hold a
  // second full copy beside the paged one, and the reference exists only
  // for the two bit-equality gates. CONSERVATION stays enforced -- it is
  // self-contained (mesh total == input charge, exact).
  bool no_dense = false;
  // Optional file tier (full CTE stack: hbm-resident cache + RAM + file).
  u64 nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_gmx_tier.dat";
  u64 page_kb = 64, atoms = 200000;
  int repeat = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--cap") cap = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--atoms") atoms = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--no-dense") no_dense = true;
    else if (a == "--nodes") nodes = static_cast<u32>(next());
    else if (a == "--node") node = static_cast<u32>(next());
    else if (a == "--nvme-mb") nvme_mb = next();
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--cap PAGES] "
                  "[--page-kb N] [--atoms N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }

  // The coroutine-mode refusal moved with the kernels: a build that cannot
  // compile them does not produce this target at all now.
  // One page = one XY plane of u64 mesh points: K^2 * 8 bytes. ANY K whose
  // plane is exactly the page works, not just powers of two -- the MPI
  // edition takes arbitrary K, and matching its 6 GB-class meshes (e.g.
  // K=912 -> page-kb 6498) needs the same freedom here. The exactness
  // check stays: a page that is not exactly one plane breaks the
  // one-writer-per-plane decomposition the digit-exact gates rely on.
  const u64 page_bytes = page_kb * 1024;
  u64 K = static_cast<u64>(std::sqrt(
      static_cast<double>(page_bytes / sizeof(unsigned long long))));
  while (K * K * sizeof(unsigned long long) < page_bytes) ++K;
  if (K * K * sizeof(unsigned long long) != page_bytes) {
    std::fprintf(stderr, "GMX ERROR: --page-kb %llu is not a square u64 "
                 "plane (K^2*8 must equal the page exactly); e.g. 8, 32, "
                 "128, 512, 6498 (K=912)...\n",
                 (unsigned long long)page_kb);
    return 2;
  }
  const u64 plane = K * K;
  const u64 nmesh = plane * K;   // K^3
  if (node >= nodes) {
    std::fprintf(stderr, "GMX ERROR: --node %u out of range for --nodes "
                 "%u\n", node, nodes);
    return 2;
  }
  // This node's slab. K stays GLOBAL: the mesh geometry and the fixed-point
  // science must be identical however the mesh is cut, or the gates stop
  // comparing the same quantity.
  const u64 k_local = (K + nodes - 1) / nodes;
  const u64 zbase = static_cast<u64>(node) * k_local;
  const u64 zend = (zbase + k_local < K) ? (zbase + k_local) : K;
  if (nodes > 1 && zbase >= K) {
    std::fprintf(stderr, "GMX ERROR: --nodes %u leaves node %u with no "
                 "planes of K=%llu\n", nodes, node,
                 (unsigned long long)K);
    return 2;
  }
  const u64 zper = ((zend - zbase) + blocks - 1) / blocks;
  // The gather holds 4 planes at once per block; refuse a cache that could
  // evict a plane mid-read (same rule as grayscott's slots >= window).
  if (cap != 0 && cap < 4 * blocks + 2) {
    std::fprintf(stderr, "GMX ERROR: --cap %u < %u (4 planes held per block "
                 "x %u blocks + slack). The gather window would evict pages "
                 "it is reading.\n", cap, 4 * blocks + 2, blocks);
    return 2;
  }

  // THE BENCH OWNS ITS CONFIG ONLY WHEN NOBODY ELSE SUPPLIED ONE. Writing
  // one and Setenv-ing it with overwrite=1 unconditionally makes it
  // impossible to point this bench at a cluster: any CLIO_SERVER_CONF the
  // caller exported is clobbered a line later, so every node stands up its
  // own single-host runtime on the same port and they collide. A distributed
  // harness needs exactly that config -- one naming a hostfile and the other
  // nodes -- so an already-set CLIO_SERVER_CONF is left alone.
  if (getenv("CLIO_SERVER_CONF") != nullptr) {
    std::printf("  runtime: using CLIO_SERVER_CONF=%s (not writing one)\n",
                getenv("CLIO_SERVER_CONF"));
  } else {
    std::ofstream cfg("gv_gmx_bench.yaml");
    // The RAM tier must hold the WHOLE mesh: a fixed capacity under
    // K^3*8B makes writebacks fail (put_errors > 0) and the conservation
    // gate then reports the lost planes. Derived with a 1 GB margin.
    const u64 ram_mb =
        K * K * K * sizeof(unsigned long long) / (1024ull * 1024ull) +
        1024ull;
    cfg << "networking:\n  port: 9447\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        << "      - path: \"ram::gv_gmx_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"" << ram_mb << "MB\"\n"
        << "        score: 1.0\n";
    if (nvme_mb > 0) {
      cfg << "      - path: \"" << nvme_path << "\"\n"
          << "        bdev_type: \"file\"\n"
          << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
          << "        score: 0.0\n"
          << "        persistence_level: \"temporary\"\n";
    }
    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_gmx_bench.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GMX ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GMX ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  // WITHOUT THIS the first paged launch reads a null IpcManager and faults
  // inside BeginFetch -- memcheck: "Invalid __global__ read ... Access to 0x0".
  gx::InitBackend(blocks, gpu);

  std::printf("PME spread+gather over a GPU vector\n"
              "  mesh=%llu^3 (page = one %lluKB plane, %llu planes)\n"
              "  atoms=%llu  blocks=%u threads=%u  cache=%s\n",
              (unsigned long long)K, (unsigned long long)page_kb,
              (unsigned long long)K, (unsigned long long)atoms, blocks,
              threads,
              cap == 0 ? "resident" : (std::to_string(cap) + " pages").c_str());

  // ---- Atoms: deterministic cloud, z-binned CSR on the host. -------------
  std::vector<float> hx(atoms), hy(atoms), hz(atoms);
  std::vector<long long> hq(atoms);
  std::vector<u32> bin_count(K + 1, 0);
  {
    u64 s = 0x9E3779B97F4A7C15ull;
    for (u64 a = 0; a < atoms; ++a) {
      s = Lcg(s); hx[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hy[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hz[a] = Frac01(s) * static_cast<float>(K);
      // Alternating unit charges in Q40.24: the mesh total must come back
      // as EXACTLY zero for even atom counts.
      hq[a] = (a & 1) ? -(1ll << 24) : (1ll << 24);
    }
  }
  // Sort by base bin iz0 = floor(z) - 1 (mod K): CSR over bins.
  std::vector<u32> order(atoms);
  {
    std::vector<u32> bin(atoms);
    for (u64 a = 0; a < atoms; ++a) {
      const int iz0 = static_cast<int>(std::floor(hz[a])) - 1;
      bin[a] = static_cast<u32>((iz0 + static_cast<int>(K)) %
                                static_cast<int>(K));
      bin_count[bin[a] + 1]++;
    }
    for (u64 b = 0; b < K; ++b) bin_count[b + 1] += bin_count[b];
    std::vector<u32> cur(bin_count.begin(), bin_count.end() - 1);
    for (u64 a = 0; a < atoms; ++a) order[cur[bin[a]]++] = static_cast<u32>(a);
  }
  auto permute_f = [&](std::vector<float> &v) {
    std::vector<float> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = v[order[a]];
    v.swap(t);
  };
  permute_f(hx); permute_f(hy); permute_f(hz);
  {
    std::vector<long long> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hq[order[a]];
    hq.swap(t);
  }
  long long q_total = 0;
  for (u64 a = 0; a < atoms; ++a) q_total += hq[a];

  auto *d_ax = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_ay = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_az = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_aq = ctp::GpuApi::Malloc<long long>(atoms * sizeof(long long));
  auto *d_bs = ctp::GpuApi::Malloc<u32>((K + 1) * sizeof(u32));
  ctp::GpuApi::Memcpy(d_ax, hx.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_ay, hy.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_az, hz.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_aq, hq.data(), atoms * sizeof(long long));
  ctp::GpuApi::Memcpy(d_bs, bin_count.data(), (K + 1) * sizeof(u32));

  // ---- Dense reference: same kernels, plain memory. ----------------------
  double t_ref_spread = 0.0;
  unsigned long long ref[4] = {0, 0, 0, 0};
  auto *d_out = ctp::GpuApi::Malloc<unsigned long long>(
      4 * sizeof(unsigned long long));
  ctp::GpuApi::Memset(d_out, 0, 4 * sizeof(unsigned long long));
  // Bins share the 0..K-1 index space with planes, so the same slab split
  // gives every bin exactly one owner -- summing the per-node gather
  // energies reproduces the single-node total with nothing double counted.
  const u64 bper = ((zend - zbase) + blocks - 1) / blocks;
  // THE DENSE REFERENCE IS NOT SHARDED. Its mesh is private device memory,
  // and dense gather for bin b reads mesh planes b..b+3 -- planes a
  // slab-local dense mesh simply does not have, because only this node's
  // slab was spread into it. They read as ZERO, so a sharded reference is
  // not a reference at all: measured as MESH and GATHER failing while
  // CONSERVATION (which never leaves the slab) passed. The atoms are
  // replicated, so every node can and must compute the whole thing.
  const u64 zper_all = (K + blocks - 1) / blocks;
  const u64 bper_all = zper_all;
  if (!no_dense) {
    auto *d_mesh = ctp::GpuApi::Malloc<unsigned long long>(
        nmesh * sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_mesh, 0, nmesh * sizeof(unsigned long long));
    const double t_ref0 = NowMs();
    gx::LaunchDenseSpread(blocks, threads, d_mesh, d_ax, d_ay, d_az, d_aq,
                          d_bs, K, plane, zper_all);
    ctp::GpuApi::Synchronize();
    t_ref_spread = NowMs() - t_ref0;
    gx::LaunchDenseSum(d_mesh, nmesh, d_out);
    gx::LaunchDenseGather(blocks, threads, d_mesh, d_ax, d_ay, d_az, d_aq,
                          d_bs, K, plane, bper_all, &d_out[2]);
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(ref, d_out, sizeof(ref));
    ctp::GpuApi::Free(d_mesh);
  }

  // ---- Paged path. -------------------------------------------------------
  const u32 tags = 24;
  gv::Vector<unsigned long long> mesh(
      "gv_gmx_mesh", {0}, page_bytes, blocks, tags, nmesh,
      clio::run::PoolId::GetNull(), 0, 1, 0,
      cap == 0 ? static_cast<u32>(K + 2) : cap);
  mesh.EnableStats();
  auto dmesh = mesh.GetDevice(0);
  YieldRunner runner(blocks, threads);

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    gx::LaunchZero(g, b, gpu, dmesh, K, plane, zper, zbase, zend, vw, sv);
  });
  ctp::GpuApi::Synchronize();

  // ---- cross-node reduction client -------------------------------------
  // After the runtime is up, and only when there is a reduction to do.
  // The reduction tag is NOT per-node: the partial gate totals must meet.
  std::unique_ptr<clio::cte::core::Client> cte_red;
  clio::cte::core::TagId red_tag{};
  u64 red_round = 0;
  if (nodes > 1) {
    cte_red = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_red->AsyncGetOrCreateTag("gv_gmx_red");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "GMX ERROR: could not create reduction tag\n");
      return 1;
    }
    red_tag = t->tag_id_;
  }

  double t_spread = 0.0, t_gather = 0.0;
  unsigned long long got[4] = {0, 0, 0, 0};
  for (int r = 0; r < repeat; ++r) {
    if (r != 0) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        gx::LaunchZero(g, b, gpu, dmesh, K, plane, zper, zbase, zend, vw,
                       sv);
      });
      ctp::GpuApi::Synchronize();
    }
    const double t0 = NowMs();
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchSpread(g, b, gpu, dmesh, d_ax, d_ay, d_az, d_aq, d_bs, K,
                       plane, zper, zbase, zend, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    const double t1 = NowMs();
    t_spread += t1 - t0;
    ctp::GpuApi::Memset(d_out, 0, 4 * sizeof(unsigned long long));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchSum(g, b, gpu, dmesh, K, plane, zper, d_out, zbase, zend,
                    vw, sv);
    });
    ctp::GpuApi::Synchronize();
    // SPREAD IS PLANE-LOCAL; GATHER IS NOT. Gathering bin b reads mesh
    // planes b..b+3, which at a slab edge belong to a PEER. Those pages
    // are only readable once the peer has written them back -- a page it
    // still holds resident is invisible to everyone else, so this node
    // reads a stale or empty plane and the energy comes out LOW with every
    // counter clean. Measured: paged 36255318214 vs dense 37132459647,
    // 2.4% short, while CONSERVATION and MESH (both plane-local) passed.
    // Flush, then barrier: the flush makes this node's slab visible and
    // the barrier stops anyone gathering before every peer has flushed.
    if (nodes > 1) {
      mesh.FlushResidentToCte();
      if (!clio_bench_dist::Barrier(*cte_red, red_tag, node, nodes,
                                    red_round++, "gxbar")) {
        std::fprintf(stderr, "GMX ERROR: spread/gather barrier failed\n");
        return 1;
      }
    }
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchGather(g, b, gpu, dmesh, d_ax, d_ay, d_az, d_aq, d_bs, K,
                       plane, bper, &d_out[2], zbase, zend, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    t_gather += NowMs() - t1;
    ctp::GpuApi::Memcpy(got, d_out, sizeof(got));
    // Each node summed only its own slab of planes and its own bins, so
    // the four gate totals are partials. They are EXACT integers and
    // integer addition commutes, so combining them keeps CONSERVATION,
    // MESH and GATHER bit-equal at any node count -- the same property
    // the MPI edition relies on. Reduced as u64: a double would drop the
    // low bits of a fixed-point total above 2^53 and silently weaken
    // three exact gates into approximate ones.
    if (nodes > 1 &&
        !clio_bench_dist::ReduceSumU64(*cte_red, red_tag, node, nodes,
                                       red_round++, got, 4, "gxred")) {
      std::fprintf(stderr, "GMX ERROR: gate reduction failed\n");
      return 1;
    }
  }

  const auto st = mesh.ReadStats(0);
  std::printf("  paging: faults=%llu evicts=%llu puts=%llu get_errors=%llu "
              "put_errors=%llu\n",
              (unsigned long long)st.faults, (unsigned long long)st.evicts,
              (unsigned long long)st.puts, (unsigned long long)st.get_errors,
              (unsigned long long)st.put_errors);
  std::printf("  spread %.1f ms (dense %.1f)  gather+sum %.1f ms\n",
              t_spread / repeat, t_ref_spread, t_gather / repeat);

  // ---- The three exact gates. --------------------------------------------
  int rc = 0;
  const unsigned long long want_q =
      static_cast<unsigned long long>(q_total);
  if (got[0] != want_q) {
    std::printf("  CONSERVATION GATE: FAIL (mesh total %llu != input %llu)\n",
                got[0], want_q);
    rc = 1;
  } else {
    std::printf("  CONSERVATION GATE: PASS (mesh total == input charge, "
                "exact)\n");
  }
  if (no_dense) {
    std::printf("  MESH/GATHER GATES: SKIPPED (--no-dense; conservation "
                "only)\n");
  } else {
    if (got[0] != ref[0] || got[1] != ref[1]) {
      std::printf("  MESH GATE: FAIL (paged q=%llu ck=%llu vs dense q=%llu "
                  "ck=%llu)\n", got[0], got[1], ref[0], ref[1]);
      rc = 1;
    } else {
      std::printf("  MESH GATE: PASS (checksum bit-equal to dense "
                  "reference)\n");
    }
    if (got[2] != ref[2]) {
      std::printf("  GATHER GATE: FAIL (paged %llu vs dense %llu)\n", got[2],
                  ref[2]);
      rc = 1;
    } else {
      std::printf("  GATHER GATE: PASS (interpolation energy bit-equal)\n");
    }
  }
  std::printf("%s\n", rc == 0 ? "GMX BENCH: ALL GATES PASS"
                              : "GMX BENCH: GATE FAILURE");
  BenchFlushData();
  return rc;
}
#endif  // !CTP_IS_DEVICE_PASS
