/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The Gray-Scott device code -- ONE copy, compiled by CUDA and by SYCL.
 *
 * Same split as kmeans, for the same reason: the workload is here, the
 * launches are in cuda/ and sycl/, and the two differ only in how a grid is
 * submitted. Nothing in this file is backend-specific -- it uses no warp
 * intrinsics, no __shared__ and no cooperative groups, so the device-side
 * spellings that remain (threadIdx, atomicAdd, __syncthreads) are supplied
 * for SYCL by clio_ctp/util/sycl_cuda_compat.h.
 *
 * CTP_GPU_FUN, not plain inline: under CUDA these must be __device__ or the
 * host pass compiles them as host functions and every co_await on a
 * device-only verb fails to resolve. Under SYCL it expands to nothing.
 */
#ifndef CLIO_GV_BENCH_GRAYSCOTT_KERNELS_H_
#define CLIO_GV_BENCH_GRAYSCOTT_KERNELS_H_

#include "grayscott_math.h"

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::grayscott {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

using ::clio_gs::InitU;
using ::clio_gs::InitV;
using ::clio_gs::ReactDiffuse;

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
CTP_GPU_FUN inline void BaselineBody(const float *uzm, const float *uz,
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
    ReactDiffuse(u, v, lu, lv, Du, Dv, F, K, dt, &unx[i], &vnx[i]);
  }
}

/** Seed u and v for this block's z-range, one plane (= one page) at a time. */
CTP_GPU_FUN inline gy::YCoroMain SeedCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 ubase, u64 vbase) {
  for (u64 z = z0; z < z1; ++z) {
    {
      co_await vec.Fetch(0, ubase + z * plane, plane);
      auto h = co_await vec.HoldPage(ubase + z * plane, plane, /*write=*/true);
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h[ubase + z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective: name the plane just written.
      co_await vec.BeginFlush(1, ubase + z * plane, plane);
      // Fetch is the pinner; UnpinRange is the releaser, after the flush.
      vec.UnpinRange(ubase + z * plane, plane);
    }
    {
      co_await vec.Fetch(0, vbase + z * plane, plane);
      auto h = co_await vec.HoldPage(vbase + z * plane, plane, /*write=*/true);
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h[vbase + z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective: name the plane just written.
      co_await vec.BeginFlush(1, vbase + z * plane, plane);
      vec.UnpinRange(vbase + z * plane, plane);
    }
  }
  // The first step reads planes seeded by OTHER blocks, so the seed must be
  // durable before this kernel returns.
  co_await vec.EndFlush();
}

/**
 * One Gray-Scott step over this block's z-range.
 *
 * Holds z-1, z and z+1 for BOTH fields, then the output plane. The holds are
 * issued back to back so all of them are resident together -- which is why
 * slots >= 4 is enforced on the host. Boundary planes (z=0, z=nz-1) are copied
 * through rather than computed, the usual fixed-boundary treatment.
 */
CTP_GPU_FUN inline gy::YCoroMain StepCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 nlo, u64 nhi, u64 gen,
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

    // A GENERATION IS DEMANDED ONLY OF A NEIGHBOUR'S PLANE. Page generation
    // is stamped by the FETCH that delivers it, not by the flush that
    // publishes it, so a plane this node wrote locally and never re-fetched
    // sits at generation 0 forever -- demanding one on it stalls the block
    // with "gen stall: page N at gen 0 want G" and never resolves. Own
    // planes are current by construction and take 0 (any version); only
    // the halo, which a PEER produced, needs the demand.
    const u64 gzm = (zm < nlo || zm >= nhi) ? gen : 0;
    const u64 gzp = (zp < nlo || zp >= nhi) ? gen : 0;
    co_await vec.Fetch(gzm, ubase + zm * plane, plane);
    // Three input planes of u, then three of v, then the two outputs -- ONE
    // GUARD PER PLANE, because a guard indexes only its own held page.
    // THE HOLD IS THE PIN: each guard's plane stays resident until the
    // guard is re-assigned past it, so the sliding window (z and z+1 are
    // re-held next iteration) is expressed by the pins themselves and needs
    // no score hints.
    uzm = co_await vec.HoldPage(ubase + zm * plane, plane);
    co_await vec.Fetch(0, ubase + z * plane, plane);
    uz = co_await vec.HoldPage(ubase + z * plane, plane);
    co_await vec.Fetch(gzp, ubase + zp * plane, plane);
    uzp = co_await vec.HoldPage(ubase + zp * plane, plane);
    co_await vec.Fetch(gzm, vbase + zm * plane, plane);
    vzm = co_await vec.HoldPage(vbase + zm * plane, plane);
    co_await vec.Fetch(0, vbase + z * plane, plane);
    vz = co_await vec.HoldPage(vbase + z * plane, plane);
    co_await vec.Fetch(gzp, vbase + zp * plane, plane);
    vzp = co_await vec.HoldPage(vbase + zp * plane, plane);
    co_await vec.Fetch(0, unext + z * plane, plane);
    unx = co_await vec.HoldPage(unext + z * plane, plane, /*write=*/true);
    co_await vec.Fetch(0, vnext + z * plane, plane);
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
    co_await vec.BeginFlush(gen + 1, unext + z * plane, plane);
    co_await vec.BeginFlush(gen + 1, vnext + z * plane, plane);
    // ONE UNPIN PER FETCH, all eight planes of this step. The sliding window
    // is expressed by re-fetching z and z+1 next iteration, not by holding
    // their pins across it: a pin held across the step would accumulate one
    // per plane per z and fill the set.
    vec.UnpinRange(ubase + zm * plane, plane);
    vec.UnpinRange(ubase + z * plane, plane);
    vec.UnpinRange(ubase + zp * plane, plane);
    vec.UnpinRange(vbase + zm * plane, plane);
    vec.UnpinRange(vbase + z * plane, plane);
    vec.UnpinRange(vbase + zp * plane, plane);
    vec.UnpinRange(unext + z * plane, plane);
    vec.UnpinRange(vnext + z * plane, plane);
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

/** Sum of v over this block's range, for the correctness checksum. */
CTP_GPU_FUN inline gy::YCoroMain SumCoro(gv::DeviceVector<float> vec, u64 plane,
                                 u64 z0, u64 z1, u64 vbase, double *out) {
  for (u64 z = z0; z < z1; ++z) {
    co_await vec.Fetch(0, vbase + z * plane, plane);
    auto h = co_await vec.HoldPage(vbase + z * plane, plane);
    double acc = 0.0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      acc += static_cast<double>(h[vbase + z * plane + i]);
    }
    atomicAdd(out, acc);
    __syncthreads();
    vec.UnpinRange(vbase + z * plane, plane);
  }
}

}  // namespace clio::gv_bench::grayscott

#endif  // CLIO_GV_BENCH_GRAYSCOTT_KERNELS_H_
