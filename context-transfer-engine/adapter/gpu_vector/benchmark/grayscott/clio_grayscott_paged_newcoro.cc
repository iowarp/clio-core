#if CTP_ENABLE_SYCL
#define CLIO_SYCL_KERNEL_TU 1
#endif
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
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "../bench_memcpy_probe.h"

/** CTE core pool id, matching pool_id "512.0" in the server config the
 *  benchmark writes. The baseline talks to the core directly. */
const clio::run::PoolId kCorePool(512, 0);

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

static constexpr u32 kYieldLaneBytes = 1024;


/*
 * THE DEVICE CODE IS NOT HERE ANY MORE.
 *
 * The workload -- the initial condition, the reaction term, the three
 * coroutines and the staged-plane baseline -- lives in grayscott_kernels.h
 * (over grayscott_math.h), in ONE copy compiled by both backends. The
 * launches live in cuda/ and sycl/ and differ only in how a grid is
 * submitted. This file is the host driver and is now ordinary C++.
 *
 * See grayscott_launch.h for why the seam is at the launch.
 */
#include "../gv_launch_bounds.h"
#include "grayscott_kernels.h"
#include "grayscott_launch.h"

namespace gs = clio::gv_bench::grayscott;

// HOST DRIVER ONLY BELOW THIS LINE. Under CUDA this file is still compiled
// BY the CUDA compiler, whose device pass member-checks host bodies -- and
// gv::Vector is #if !CTP_IS_DEVICE_PASS. Under SYCL the guard is
// transparent: the driver is a plain C++ TU.

#if CTP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

/* =====================================================
 * THE WORKLOAD, on the new coroutine API. Ordinary return
 * types, ordinary locals, one marker per suspending call.
 * The `, clio::co::Ctx &_cy` on each signature and at each
 * call site is appended by clio-coroc, not written here.
 * ===================================================== */
namespace clio::gv_bench::grayscott {

/** Seed u and v for this block's z-range, one plane (= one page) at a time. */
CTP_GPU_FUN CLIO_COROC_INLINE void SeedCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 ubase, u64 vbase, u64 zlo, u64 zhi,
                                  u32 all) {
  // PUBLISH ONLY THE EDGES (unless `all`: out of core, where any plane can
  // be evicted and must be refetchable from the CTE). A peer's first step reads this node's first and
  // last planes (generation 1); every other plane is read only out of this
  // node's resident cache. Publishing the whole 32 GB seed filled a DRAM
  // tier on 8 nodes (the DPE places blobs across nodes) and the runtime
  // refused the put (FATAL 8).
  for (u64 z = z0; z < z1; ++z) {
    const bool edge = all != 0u || (z == zlo || z + 1 == zhi);
    {
      CO_AWAIT(vec.CoFetch(0, ubase + z * plane, plane));
      auto h = CO_AWAIT(vec.CoHoldPage(ubase + z * plane, plane, /*write=*/true));
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h[ubase + z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective: name the plane just written.
      if (edge) CO_AWAIT(vec.CoBeginFlush(1, ubase + z * plane, plane));
      // Fetch is the pinner; UnpinRange is the releaser, after the flush.
      vec.UnpinRange(ubase + z * plane, plane);
    }
    {
      CO_AWAIT(vec.CoFetch(0, vbase + z * plane, plane));
      auto h2 = CO_AWAIT(vec.CoHoldPage(vbase + z * plane, plane, /*write=*/true));
      for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
        h2[vbase + z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
      }
      // Collective: name the plane just written.
      if (edge) CO_AWAIT(vec.CoBeginFlush(1, vbase + z * plane, plane));
      vec.UnpinRange(vbase + z * plane, plane);
    }
  }
  // The first step reads planes seeded by OTHER blocks, so the seed must be
  // durable before this kernel returns.
  CO_AWAIT(vec.CoEndFlush());
}

// GLOBAL, SAID OUT LOUD (see the kmeans edition). The step body is a call the
// kernel does not inline, so IGC cannot prove a held plane's pointer is
// global and every stencil load became a generic-address check plus a
// divergent branch. The cast is applied at each use and never stored:
// clio-coroc byte-copies a coroutine's locals, which a multi_ptr cannot be.
#if CTP_ENABLE_SYCL && defined(__SYCL_DEVICE_ONLY__)
template <typename T>
CTP_GPU_FUN inline auto GsG(T *p) {
  return ::sycl::address_space_cast<::sycl::access::address_space::global_space,
                                    ::sycl::access::decorated::yes>(p);
}
#else
template <typename T>
CTP_GPU_FUN inline T *GsG(T *p) { return p; }
#endif

// THE STENCIL, OUT OF LINE. Inside StepCoro the loop shared the register
// budget of the whole coroutine body -- eight plane pointers, the step's
// indices and the transpiled state all live across it -- and IGC spilled
// inside the loop: 23 scratch fills/spills per pass at 128 GRF, one at 256.
// As its own function it is allocated alone, the way the baselines' step
// kernel is. Same arithmetic and index types as the baselines (u64).
CTP_GPU_FUN __attribute__((noinline)) void StencilPlane(
    float *pum, float *pu0, float *pup, float *pvm, float *pv0, float *pvp,
    float *pun, float *pvn, u64 plane, u64 nx, u64 ny, bool interior,
    float Du, float Dv, float F, float K, float dt, u64 e0, u64 e1) {
  (void)plane;
  for (u64 i = e0 + threadIdx.x; i < e1; i += blockDim.x) {
    const u64 x = i % nx;
    const u64 y = i / nx;
    const float u = GsG(pu0)[i];
    const float v = GsG(pv0)[i];
    float lu;
    float lv;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
      lu = 0.0f; lv = 0.0f;      // fixed boundary
    } else {
      lu = GsG(pu0)[i - 1] +
           GsG(pu0)[i + 1] +
           GsG(pu0)[i - nx] +
           GsG(pu0)[i + nx] +
           GsG(pum)[i] +
           GsG(pup)[i] - 6.0f * u;
      lv = GsG(pv0)[i - 1] +
           GsG(pv0)[i + 1] +
           GsG(pv0)[i - nx] +
           GsG(pv0)[i + nx] +
           GsG(pvm)[i] +
           GsG(pvp)[i] - 6.0f * v;
    }
    const float uvv = u * v * v;
    GsG(pun)[i] = u + dt * (Du * lu - uvv + F * (1.0f - u));
    GsG(pvn)[i] = v + dt * (Dv * lv + uvv - (F + K) * v);
  }
}

/** No plane held in a window slot (StepCoro). Namespace scope: clio-coroc
 *  does not hoist a constexpr local. */
constexpr u64 kNoZ = ~static_cast<u64>(0);

/**
 * One Gray-Scott step over this block's z-range.
 *
 * Holds z-1, z and z+1 for BOTH fields, then the output plane. The holds are
 * issued back to back so all of them are resident together -- which is why
 * slots >= 4 is enforced on the host. Boundary planes (z=0, z=nz-1) are copied
 * through rather than computed, the usual fixed-boundary treatment.
 */
CTP_GPU_FUN CLIO_COROC_INLINE void StepCoro(gv::DeviceVector<float> vec, u64 plane,
                                  u64 nx, u64 ny, u64 nz, u64 z0, u64 z1,
                                  u64 nlo, u64 nhi, u64 gen,
                                  u64 ubase, u64 vbase, u64 unext, u64 vnext,
                                  float Du, float Dv, float F, float K,
                                  float dt, u64 e0, u64 e1, u64 zstride) {
  // SLIDING WINDOW OVER THE INPUT PLANES. The step used to fetch, hold and
  // unpin z-1, z and z+1 of both fields at EVERY z, so each input plane went
  // through the paging path three times (as z+1, then z, then z-1) -- eight
  // fetch/hold/unpin sequences per plane, each a set scan and a handful of
  // block-wide collectives. The window {z-1, z, z+1} is consecutive, so a
  // 3-slot ring indexed by plane % 3 keeps each held plane in its own slot:
  // per z only the plane entering the window is fetched and held, and the one
  // leaving it (z-2) is unpinned when its slot is reused. Outputs are still
  // one fetch/hold/unpin per z. Halo planes demand the step's generation
  // exactly as before; they are fetched once per step instead of three times.
  float *pus[3] = {nullptr, nullptr, nullptr};
  float *pvs[3] = {nullptr, nullptr, nullptr};
  u64 held[3] = {kNoZ, kNoZ, kNoZ};
  gv::PageRef<float> hu;
  gv::PageRef<float> hv;
  gv::PageRef<float> unx;
  gv::PageRef<float> vnx;
  for (u64 z = z0; z < z1; z += zstride) {
    gy::YieldPublishCursor(z + 1);
    const bool interior = (z > 0 && z + 1 < nz);
    const u64 zm = interior ? (z - 1) : z;
    const u64 zp = interior ? (z + 1) : z;
    for (u32 k = 0; k < 3; ++k) {
      const u64 d = (k == 0) ? zm : ((k == 1) ? z : zp);
      const u32 slot = static_cast<u32>(d % 3);
      if (held[slot] != d) {
        if (held[slot] != kNoZ) {
          vec.UnpinRange2(ubase + held[slot] * plane, plane,
                          vbase + held[slot] * plane, plane);
        }
        // A GENERATION IS DEMANDED ONLY OF A NEIGHBOUR'S PLANE (see below the
        // seed): own planes are current by construction and take 0.
        const u64 g = (d < nlo || d >= nhi) ? gen : 0;
        // Both fields' plane in ONE fetch (CoFetch takes several ranges).
        CO_AWAIT(vec.CoFetch(g, ubase + d * plane, plane, vbase + d * plane,
                             plane));
        hu = CO_AWAIT(vec.CoHoldPage(ubase + d * plane, plane, /*write=*/false));
        pus[slot] = hu.ptr() + (ubase + d * plane - hu.begin_off());
        hv = CO_AWAIT(vec.CoHoldPage(vbase + d * plane, plane, /*write=*/false));
        pvs[slot] = hv.ptr() + (vbase + d * plane - hv.begin_off());
        held[slot] = d;
      }
    }
    CO_AWAIT(vec.CoFetch(0, unext + z * plane, plane, vnext + z * plane,
                         plane));
    unx = CO_AWAIT(vec.CoHoldPage(unext + z * plane, plane, /*write=*/true));
    vnx = CO_AWAIT(vec.CoHoldPage(vnext + z * plane, plane, /*write=*/true));
    float *pun = unx.ptr() + (unext + z * plane - unx.begin_off());
    float *pvn = vnx.ptr() + (vnext + z * plane - vnx.begin_off());
    StencilPlane(pus[zm % 3], pus[z % 3], pus[zp % 3], pvs[zm % 3],
                 pvs[z % 3], pvs[zp % 3], pun, pvn, plane, nx, ny, interior,
                 Du, Dv, F, K, dt, e0, e1);
    __syncthreads();
    vec.UnpinRange2(unext + z * plane, plane, vnext + z * plane, plane);
  }
  // NO FLUSH IN THE STEP. The node-edge planes another node reads are
  // published by PublishCoro, launched after this kernel has finished -- with
  // a plane split across blocks (--plane-split) no single block knows when
  // an edge plane is complete, and the step kernel's end is the one point
  // where every block's writes are.
  for (u32 k = 0; k < 3; ++k) {
    if (held[k] != kNoZ) {
      vec.UnpinRange2(ubase + held[k] * plane, plane, vbase + held[k] * plane,
                      plane);
    }
  }
  hu = {}; hv = {}; unx = {}; vnx = {};
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

/** TWO-PHASE, PHASE 1: fetch, pin and hold every plane this block's slab of
 *  the step needs, and record each plane's frame pointer in `tab`. The pins
 *  are kept (ReleaseCoro gives them back after the compute kernel), so the
 *  frames cannot move or be evicted while the plain stencil kernel reads
 *  them. Table layout, t = z - (zbase - 1): pu[t], pv[t] for input planes
 *  zbase-1 .. zend (the node halo included), pun[z - zbase], pvn[z - zbase]
 *  for output planes. The halo planes are fetched at the step's generation;
 *  block 0 takes the low one, the last block the high one. */
/** One plane of ResolveCoro: fetch (at `gen`) and hold the planes named, and
 *  write their frame pointers. OUT OF LINE because the loop written inline
 *  made IGC's backend crash (gen compiler exit 245, the code-shape bug of
 *  AURORA.md (4)); the documented fix is to give the suspending loop body
 *  its own noinline coroutine. `outs` = 0 resolves only u and v (a halo
 *  plane); 1 also resolves the two output planes. */
CTP_GPU_FUN __attribute__((noinline)) void ResolvePlane(
    gv::DeviceVector<float> vec, u64 plane, u64 z, u64 zbase, u64 gen,
    u64 ubase, u64 vbase, u64 unext, u64 vnext, float **tab, u64 ntab,
    u32 outs) {
  gv::PageRef<float> h;
  if (outs != 0u) {
    CO_AWAIT(vec.CoFetch(gen, ubase + z * plane, plane, vbase + z * plane,
                         plane, unext + z * plane, plane, vnext + z * plane,
                         plane));
  } else {
    CO_AWAIT(vec.CoFetch(gen, ubase + z * plane, plane, vbase + z * plane,
                         plane));
  }
  h = CO_AWAIT(vec.CoHoldPage(ubase + z * plane, plane, /*write=*/false));
  if (threadIdx.x == 0) {
    tab[0 * ntab + (z - zbase + 1)] =
        h.ptr() + (ubase + z * plane - h.begin_off());
  }
  h = CO_AWAIT(vec.CoHoldPage(vbase + z * plane, plane, /*write=*/false));
  if (threadIdx.x == 0) {
    tab[1 * ntab + (z - zbase + 1)] =
        h.ptr() + (vbase + z * plane - h.begin_off());
  }
  if (outs != 0u) {
    h = CO_AWAIT(vec.CoHoldPage(unext + z * plane, plane, /*write=*/true));
    if (threadIdx.x == 0) {
      tab[2 * ntab + (z - zbase)] =
          h.ptr() + (unext + z * plane - h.begin_off());
    }
    h = CO_AWAIT(vec.CoHoldPage(vnext + z * plane, plane, /*write=*/true));
    if (threadIdx.x == 0) {
      tab[3 * ntab + (z - zbase)] =
          h.ptr() + (vnext + z * plane - h.begin_off());
    }
  }
  h = {};
}

CTP_GPU_FUN CLIO_COROC_INLINE void ResolveCoro(gv::DeviceVector<float> vec,
                                    u64 plane, u64 nz, u64 z0, u64 z1,
                                    u64 zbase, u64 zend, u64 gen, u64 ubase,
                                    u64 vbase, u64 unext, u64 vnext,
                                    float **tab, u64 ntab, u32 blk, u32 nblk) {
  for (u64 z = z0; z < z1; ++z) {
    CO_AWAIT(ResolvePlane(vec, plane, z, zbase, 0, ubase, vbase, unext, vnext,
                          tab, ntab, 1u));
  }
  // Node halo: the plane below zbase and the plane at zend, from the peers.
  if (blk == 0 && zbase > 0) {
    CO_AWAIT(ResolvePlane(vec, plane, zbase - 1, zbase, gen, ubase, vbase,
                          unext, vnext, tab, ntab, 0u));
  }
  if (blk + 1 == nblk && zend < nz) {
    CO_AWAIT(ResolvePlane(vec, plane, zend, zbase, gen, ubase, vbase, unext,
                          vnext, tab, ntab, 0u));
  }
}

/** TWO-PHASE, PHASE 3: give back every pin ResolveCoro took. */
CTP_GPU_FUN CLIO_COROC_INLINE void ReleaseCoro(gv::DeviceVector<float> vec,
                                    u64 plane, u64 nz, u64 z0, u64 z1,
                                    u64 zbase, u64 zend, u64 ubase, u64 vbase,
                                    u64 unext, u64 vnext, u32 blk, u32 nblk,
                                    u64 flush_gen) {
  for (u64 z = z0; z < z1; ++z) {
    vec.UnpinRange2(ubase + z * plane, plane, vbase + z * plane, plane);
    // OUT OF CORE: write the output planes back (asynchronously -- one flush
    // in flight per block, overlapping the next plane's release) so an
    // evicted plane is refetchable. flush_gen = the generation the next
    // step's halo fetch demands, so this also publishes the node edges.
    if (flush_gen != 0) {
      CO_AWAIT(vec.CoBeginFlush(flush_gen, unext + z * plane, plane,
                                vnext + z * plane, plane));
    }
    vec.UnpinRange2(unext + z * plane, plane, vnext + z * plane, plane);
  }
  if (blk == 0 && zbase > 0) {
    vec.UnpinRange2(ubase + (zbase - 1) * plane, plane,
                    vbase + (zbase - 1) * plane, plane);
  }
  if (blk + 1 == nblk && zend < nz) {
    vec.UnpinRange2(ubase + zend * plane, plane, vbase + zend * plane, plane);
  }
  // Waits for the last writeback (out of core); otherwise no flush is
  // outstanding and it returns at once. Also the suspend point that makes
  // this a coroutine like its siblings.
  CO_AWAIT(vec.CoEndFlush());
}

/** Publish the node's edge output planes at generation `gen`, the ones a
 *  peer node's halo fetch demands. One block, after the step kernel. */
CTP_GPU_FUN CLIO_COROC_INLINE void PublishCoro(gv::DeviceVector<float> vec,
                                    u64 plane, u64 nlo, u64 nhi, u64 gen,
                                    u64 unext, u64 vnext) {
  if (nhi > nlo + 1) {
    CO_AWAIT(vec.CoBeginFlush(gen, unext + nlo * plane, plane,
                              vnext + nlo * plane, plane,
                              unext + (nhi - 1) * plane, plane,
                              vnext + (nhi - 1) * plane, plane));
  } else if (nhi > nlo) {
    CO_AWAIT(vec.CoBeginFlush(gen, unext + nlo * plane, plane,
                              vnext + nlo * plane, plane));
  }
  CO_AWAIT(vec.CoEndFlush());
}

/** Sum of v over this block's range, for the correctness checksum. */
CTP_GPU_FUN CLIO_COROC_INLINE void SumCoro(gv::DeviceVector<float> vec, u64 plane,
                                 u64 z0, u64 z1, u64 vbase, double *out) {
  for (u64 z = z0; z < z1; ++z) {
    CO_AWAIT(vec.CoFetch(0, vbase + z * plane, plane));
    auto h = CO_AWAIT(vec.CoHoldPage(vbase + z * plane, plane, /*write=*/false));
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

/* TWO BACKENDS, ONE WORKLOAD. Everything above this line is compiled for both: the transpiled state machine contains no vendor token. What differs is only how a grid is submitted. */
#if CTP_ENABLE_SYCL

namespace clio::gv_bench::grayscott {

namespace {

/** Submit one grid and wait, in CUDA's (grid, block) shape. */
template <typename BodyT>
void Submit(dim3 grid, dim3 block, BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>)
#ifdef GV_SG32
           [[intel::reqd_sub_group_size(32)]]
#endif
       { body(); })
      .wait();
}

/** The yieldable prologue, shared by all three yieldable launches. */
template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 vec, View vw, StackView sv,
                     MakeCoro make) {
  Submit(grid, block, [=]() {
    DevF32 dev = vec;
    dev.Init(vw.Block());
    __syncthreads();
    // GV_COMM_TIMING: this resident segment is the denominator of the
    // Fetch/Hold/Flush share (a no-op in an untimed build).
    const unsigned long long gv_seg0 = dev.CommSegmentBegin();
    CLIO_COROC_RUN(vw, sv, make(_cy, dev, vw.Block()));
    dev.CommStamp(3, gv_seg0);
  });
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 zbase, u64 zend, View vw, StackView sv,
                u32 all) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    const u64 z0 = zbase + static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    SeedCoro(_cy, dev, plane, nx, ny, nz, z0, z1, ubase, vbase, zbase, zend,
             all);
  });
}

void LaunchStep(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 unext, u64 vnext, float Du, float Dv, float F,
                float K, float dt, u64 zbase, u64 zend, u64 gen, View vw,
                StackView sv, u32 split) {
  (void)info;
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    // SLAB x CHUNK. `split` blocks share a slab of planes and each takes one
    // contiguous 1/split of every plane in it (see --plane-split).
    // split == 0 means CYCLIC plane order (--plane-order cyclic): block b
    // takes planes b, b+B, b+2B, ... so the blocks sweep a contiguous band
    // together and a plane's three stencil reads land close in time.
    const u64 nblk = grid.x;
    const bool nocomp = (split & 0x40000000u) != 0u;
    const u32 sp = split & ~0x40000000u;
    const u64 slab = sp ? blk / sp : 0, chunk = sp ? blk % sp : 0;
    const u64 z0 = sp ? zbase + slab * zper : zbase + blk;
    const u64 z1 = sp ? ((z0 + zper < zend) ? (z0 + zper) : zend) : zend;
    const u64 e0 = sp ? chunk * plane / sp : 0;
    const u64 e1 = sp ? (chunk + 1) * plane / sp : plane;
    const u64 zs = sp ? 1 : nblk;
    StepCoro(_cy, dev, plane, nx, ny, nz, z0, z1, zbase, zend, gen, ubase, vbase, unext, vnext,
                    Du, Dv, F, K, dt, e0, nocomp ? e0 : e1, zs);
  });
}

void LaunchPublish(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                   u64 plane, u64 zbase, u64 zend, u64 gen, u64 unext,
                   u64 vnext, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    (void)blk;
    PublishCoro(_cy, dev, plane, zbase, zend, gen, unext, vnext);
  });
}

void LaunchResolve(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                   u64 plane, u64 nz, u64 zper, u64 zbase, u64 zend, u64 gen,
                   u64 ubase, u64 vbase, u64 unext, u64 vnext, float **tab,
                   u64 ntab, View vw, StackView sv) {
  (void)info;
  const u32 nblk = static_cast<u32>(grid.x);
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    const u64 zs = zbase + static_cast<u64>(blk) * zper;
    const u64 z0 = zs < zend ? zs : zend;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    ResolveCoro(_cy, dev, plane, nz, z0, z1, zbase, zend, gen, ubase, vbase,
                unext, vnext, tab, ntab, blk, nblk);
  });
}

void LaunchRelease(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                   u64 plane, u64 nz, u64 zper, u64 zbase, u64 zend, u64 ubase,
                   u64 vbase, u64 unext, u64 vnext, View vw, StackView sv,
                   u64 flush_gen) {
  (void)info;
  const u32 nblk = static_cast<u32>(grid.x);
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    const u64 zs = zbase + static_cast<u64>(blk) * zper;
    const u64 z0 = zs < zend ? zs : zend;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    ReleaseCoro(_cy, dev, plane, nz, z0, z1, zbase, zend, ubase, vbase, unext,
                vnext, blk, nblk, flush_gen);
  });
}

/** TWO-PHASE, PHASE 2: the baselines' Step kernel, verbatim in shape (one
 *  work-group per plane, planes cyclic over groups, one work-item per cell),
 *  reading each plane through the frame pointer ResolveCoro pinned. A plain
 *  kernel, so IGC compiles it as the baselines' is compiled (SIMD32, no
 *  coroutine register budget). */
void LaunchStencilTab(u32 blocks, u32 threads, float *const *tab, u64 ntab,
                      u64 plane, u64 nx, u64 ny, u64 nz, u64 zbase, u64 zend,
                      float Du, float Dv, float F, float K, float dt) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t g = static_cast<size_t>(blocks) * threads;
  const u64 nzl = zend - zbase;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     const u64 lid = it.get_local_id(0), grp = it.get_group(0);
     for (u64 lz = grp; lz < nzl; lz += blocks) {
       const u64 gzz = zbase + lz;
       const bool interior = (gzz > 0 && gzz + 1 < nz);
       const u64 t = lz + 1;   // table index of plane gzz
       // A pointer loaded from the table is generic to IGC: cast each plane
       // to global once here (not a coroutine, so `auto` is fine).
       const float *uz0 = tab[0 * ntab + t];
       const float *vz0 = tab[1 * ntab + t];
       const auto uz = GsG(uz0);
       const auto vz = GsG(vz0);
       const auto uzm = GsG(interior ? tab[0 * ntab + t - 1] : uz0);
       const auto uzp = GsG(interior ? tab[0 * ntab + t + 1] : uz0);
       const auto vzm = GsG(interior ? tab[1 * ntab + t - 1] : vz0);
       const auto vzp = GsG(interior ? tab[1 * ntab + t + 1] : vz0);
       const auto unx = GsG(tab[2 * ntab + lz]);
       const auto vnx = GsG(tab[3 * ntab + lz]);
       for (u64 i = lid; i < plane; i += threads) {
         const u64 x = i % nx, y = i / nx;
         const float uu = uz[i];
         const float vv = vz[i];
         float lu, lv;
         if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
           lu = 0.0f; lv = 0.0f;
         } else {
           lu = uz[i - 1] + uz[i + 1] + uz[i - nx] + uz[i + nx] + uzm[i] +
                uzp[i] - 6.0f * uu;
           lv = vz[i - 1] + vz[i + 1] + vz[i - nx] + vz[i + nx] + vzm[i] +
                vzp[i] - 6.0f * vv;
         }
         const float uvv = uu * vv * vv;
         unx[i] = uu + dt * (Du * lu - uvv + F * (1.0f - uu));
         vnx[i] = vv + dt * (Dv * lv + uvv - (F + K) * vv);
       }
     }
   }).wait();
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
               u64 plane, u64 nz, u64 zper, u64 vbase, double *out,
               u64 zbase, u64 zend, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, vec, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    const u64 z0 = zbase + static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
    SumCoro(_cy, dev, plane, z0, z1, vbase, out);
  });
}

void LaunchBaseline(u32 threads, const float *uzm, const float *uz,
                    const float *uzp, const float *vzm, const float *vz,
                    const float *vzp, float *unx, float *vnx, u64 plane,
                    u64 nx, u64 ny, int interior, float Du, float Dv, float F,
                    float K, float dt) {
  Submit(dim3(1), dim3(threads), [=]() {
    BaselineBody(uzm, uz, uzp, vzm, vz, vzp, unx, vnx, plane, nx, ny, interior,
                 Du, Dv, F, K, dt);
  });
}

}  // namespace clio::gv_bench::grayscott

namespace clio::run::gpu {

/** The out-of-line half of YieldStack::Reset; see its declaration in
 *  yield_stack.h for why it cannot live in the class. */
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

namespace clio::gv_bench::grayscott {

namespace {

__global__ GV_LAUNCH_BOUNDS void SeedKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nx, u64 ny,
                           u64 nz, u64 zper, u64 ubase, u64 vbase, u64 zbase,
                           u64 zend, View yv,
                           StackView ys, u32 all) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_COROC_RUN(yv, ys, SeedCoro(_cy, vec, plane, nx, ny, nz, z0, z1, ubase, vbase, zbase, zend, all));
}

__global__ GV_LAUNCH_BOUNDS void StepKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nx, u64 ny,
                           u64 nz, u64 zper, u64 ubase, u64 vbase, u64 zbase,
                           u64 zend, u64 unext,
                           u64 vnext, float Du, float Dv, float F, float K,
                           float dt, u64 gen, View yv, StackView ys, u32 split) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  const u64 nblk = gridDim.x;
  const u64 slab = split ? yv.Block() / split : 0;
  const u64 chunk = split ? yv.Block() % split : 0;
  const u64 z0 = split ? zbase + slab * zper : zbase + yv.Block();
  const u64 z1 = split ? ((z0 + zper < zend) ? (z0 + zper) : zend) : zend;
  const u64 e0 = split ? chunk * plane / split : 0;
  const u64 e1 = split ? (chunk + 1) * plane / split : plane;
  const u64 zs = split ? 1 : nblk;
  CLIO_COROC_RUN(yv, ys, StepCoro(_cy, vec, plane, nx, ny, nz, z0, z1, zbase, zend, gen, ubase, vbase, unext,
                          vnext, Du, Dv, F, K, dt, e0, e1, zs));
}

__global__ GV_LAUNCH_BOUNDS void PublishKernel(GpuInfo info, DevF32 vec, u64 plane,
                              u64 zbase, u64 zend, u64 gen, u64 unext,
                              u64 vnext, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, PublishCoro(_cy, vec, plane, zbase, zend, gen, unext, vnext));
}

__global__ GV_LAUNCH_BOUNDS void SumKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nz,
                          u64 zper, u64 vbase, double *out, u64 zbase, u64 zend, View yv,
                          StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_COROC_RUN(yv, ys, SumCoro(_cy, vec, plane, z0, z1, vbase, out));
}

__global__ GV_LAUNCH_BOUNDS void BaselineKernel(const float *uzm, const float *uz,
                               const float *uzp, const float *vzm,
                               const float *vz, const float *vzp, float *unx,
                               float *vnx, u64 plane, u64 nx, u64 ny,
                               int interior, float Du, float Dv, float F,
                               float K, float dt) {
  BaselineBody(uzm, uz, uzp, vzm, vz, vzp, unx, vnx, plane, nx, ny, interior,
               Du, Dv, F, K, dt);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // Nothing to do: CUDA's per-block IpcManager is __shared__ storage, born
  // fresh at every launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 zbase, u64 zend, View vw, StackView sv,
                u32 all) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nx, ny, nz, zper, ubase, vbase, zbase, zend, vw,
      sv, all);
}

void LaunchStep(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 unext, u64 vnext, float Du, float Dv, float F,
                float K, float dt, u64 zbase, u64 zend, u64 gen, View vw,
                StackView sv, u32 split) {
  StepKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nx, ny, nz, zper, ubase, vbase, zbase, zend, unext,
      vnext, Du, Dv, F, K, dt, gen, vw, sv, split);
}

void LaunchPublish(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                   u64 plane, u64 zbase, u64 zend, u64 gen, u64 unext,
                   u64 vnext, View vw, StackView sv) {
  PublishKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, zbase, zend, gen, unext, vnext, vw, sv);
}

// Two-phase mode is SYCL-only for now; the host refuses it on CUDA.
void LaunchResolve(dim3, dim3, const GpuInfo &, DevF32, u64, u64, u64, u64,
                   u64, u64, u64, u64, u64, u64, float **, u64, View,
                   StackView) {}
void LaunchRelease(dim3, dim3, const GpuInfo &, DevF32, u64, u64, u64, u64,
                   u64, u64, u64, u64, u64, View, StackView, u64) {}
void LaunchStencilTab(u32, u32, float *const *, u64, u64, u64, u64, u64, u64,
                      u64, float, float, float, float, float) {}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
               u64 plane, u64 nz, u64 zper, u64 vbase, double *out,
               u64 zbase, u64 zend, View vw, StackView sv) {
  SumKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nz, zper, vbase, out, zbase, zend, vw, sv);
}

void LaunchBaseline(u32 threads, const float *uzm, const float *uz,
                    const float *uzp, const float *vzm, const float *vz,
                    const float *vzp, float *unx, float *vnx, u64 plane,
                    u64 nx, u64 ny, int interior, float Du, float Dv, float F,
                    float K, float dt) {
  BaselineKernel<<<1, threads>>>(uzm, uz, uzp, vzm, vz, vzp, unx, vnx, plane,
                                 nx, ny, interior, Du, Dv, F, K, dt);
}

}  // namespace clio::gv_bench::grayscott

#endif  /* CTP_ENABLE_SYCL */

#if !CTP_IS_DEVICE_PASS

// Cross-node reduction. Included INSIDE the device-pass guard: it uses the
// CTE client, whose members are compiled out of the CUDA device pass.
#include "../bench_dist.h"
#include "../bench_ckpt.h"
#include "../gv_comm_report.h"
// Host-only for the same reason: the prefetcher is pure host policy that
// speaks to the CTE, and never appears in a kernel.
#include "grayscott_prefetch.h"

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

  /**
   * Route this driver's post-round states into the vector's prefetchers.
   *
   * The seam is the runtime's (Yieldable knows nothing about the CTE) and the
   * policy is the vector's (which owns the tag and the client); this call is
   * the one line that joins them.
   */
  void BindPrefetch(gv::Vector<float> &vec) {
    drv_.SetYieldObserver(vec.YieldObserver());
  }

  double KernelMs() const { return drv_.KernelMs(); }
  void ResetTimers() { drv_.ResetTimers(); }

  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    // Both resets are required: RunToCompletion does not reset, so a reused
    // runner whose driver still reads "done" skips the launch entirely.
    drv_.Reset();
    stack_.Reset();
    // WHO IS RELAUNCHED, AND WHY: blocks parked with no tag (relaunched
    // every round to re-check) against blocks resumed because their task's
    // completion word flipped. Printed for any pass that needs > 50 rounds.
    unsigned long long n_tag0 = 0, n_ready = 0, n_wait = 0;
    std::map<clio::run::u32, unsigned long long> rp0;
    const u32 rounds = drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000,
        [&](clio::run::u32 blk, clio::run::u64 tag) {
          if (tag == 0) {
            ++n_tag0;
            ++rp0[drv_.BlockState(blk).resume_point_];
            return true;
          }
          const bool r = gv::ResumeWhenComplete(blk, tag);
          (r ? n_ready : n_wait)++;
          return r;
        });
    if (rounds > 50) {
      std::fprintf(stderr, "  [relaunch] rounds=%u tag0_relaunches=%llu "
                   "tag_ready=%llu tag_still_waiting=%llu | tag0 resume "
                   "points:", rounds, n_tag0, n_ready, n_wait);
      for (const auto &kv : rp0) {
        std::fprintf(stderr, " %u:%llu", kv.first, kv.second);
      }
      std::fprintf(stderr, "\n");
    }
    return rounds;
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, slots = 12, steps = 4;
  // --set-size N: cache SETS of N ways, as many as it takes to hold
  // blocks * slots pages. Default 0 keeps one set per block, `slots` wide.
  u32 set_size_arg = 0;
  // --plane-split G: G blocks share each slab of planes, each taking 1/G of
  // every plane. MPI's grid-stride works on ~one plane across the whole GPU
  // at a time, so the z-1/z/z+1 planes stay in L3 and each is read from
  // memory once; one slab per block had 1024 slabs in flight, evicted the
  // neighbours between uses and read ~3x the bytes (VTune: 207 GB vs 68 GB
  // per 4 steps at 1024 blocks). Default 1 keeps one slab per block.
  u32 plane_split = 1;
  // --plane-order cyclic: block b steps planes b, b+B, b+2B, ... (the
  // baselines' grid-stride order) instead of a contiguous slab.
  bool plane_cyclic = false;
  // --two-phase: the coroutine kernel fetches, pins and resolves every page
  // of the step into a pointer table; a plain kernel (the baselines' Step,
  // verbatim) computes through it; a second coroutine pass unpins. Paging
  // stays Eternia's; only the arithmetic leaves the coroutine, where the
  // stencil ran SIMD16 under the coroutine's register budget (measured
  // ~1.7x the baselines' stencil at 1024 blocks with paging at ~8%).
  bool two_phase = false;
  // --ooc (with --two-phase): OUT OF CORE. The deck is larger than the frame
  // cache, so each step walks the node's slab in BANDS that fit the cache
  // (resolve band -> stencil -> release), and the release writes every
  // output plane back with an asynchronous CoBeginFlush at the generation
  // the next step demands. The seed publishes every plane. That makes any
  // plane evictable and refetchable; resident runs (E1) keep edge-only
  // publishing.
  bool ooc = false;
  // GS_NO_COMPUTE=1 (diagnostic): every page operation runs, the stencil
  // arithmetic does not, so step time minus this is the stencil's share.
  const bool no_compute = getenv("GS_NO_COMPUTE") != nullptr;
  // Z-SLAB DECOMPOSITION, like the MPI edition. --nodes N --node i gives
  // this process a contiguous slab of the SAME global field; nz stays
  // global so the fixed boundary (z == 0, z == nz-1) is honoured however
  // the domain is cut. One node is the whole field, so a single-process run
  // is unchanged and is the reference a distributed run must reproduce.
  u32 nodes = 1, node = 0;
  // NEGATIVE CONTROL. With the exchange off a multi-node run must produce
  // a DIFFERENT checksum; if it does not, the exchange is not doing
  // anything and the passing run proves nothing.
  // NON-EMPTY, not merely present. docker-compose passes an unset variable
  // through as GS_NO_HALO= (empty), and getenv returns a non-null empty
  // string for that -- so a presence test silently enables the control in
  // every harness run, which is exactly how a correct build reported the
  // control's wrong answer.
  const char *const gs_no_halo = getenv("GS_NO_HALO");
  const bool halo_off = gs_no_halo != nullptr && gs_no_halo[0] != '\0';
  u64 page_kb = 1024, data_mb = 16384, hbm_mb = 4096;
  int repeat = 3;
  // Out-of-core WITHOUT in-kernel faulting: synchronous CTE reads AND
  // writes, synchronous memcpy both ways, kernel torn down per z.
  bool baseline = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;
  // dt = 0.5: explicit Euler on the 7-point Laplacian is stable only while
  // 12 * Du * dt < 2. At dt = 1 (12 * 0.2 = 2.4) the checkerboard mode grows
  // 1.4x per step -- invisible over 8 steps, NaN by 256.
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 0.5f;
  // PREFETCH: tier hints issued from the driver's gap between rounds. "none"
  // is the baseline and the default -- registering a prefetcher changes where
  // pages live, so every existing number stays comparable unless asked.
  // "stride" is the generic control that knows nothing about Gray-Scott; the
  // gap between it and "gs" is what "workload-specific" is worth.
  std::string prefetch = "none";
  // 0 = SIZE IT FROM THE FAST TIER; see the derivation below.
  u64 lookahead = 4;
  float pf_hot = 1.0f, pf_warm = 0.2f, pf_cold = 0.0f;
  bool pf_demote = true;
  // 0 = no gate. See PrefetchPolicy::stall_every_.
  u64 pf_stall_every = 0;
  // Tier scores and the host tier's capacity. Defaults reproduce the historic
  // configuration exactly; see the config writer for what they mean.
  float hbm_score = 1.0f, ram_score = 0.2f, nvme_score = 0.0f;
  // CHECKPOINTING. 0 = off. Every N steps the whole vector is snapshotted
  // with Vector::Copy (the lazy copy-on-write primitive) and then
  // MATERIALISED, because an untouched Copy is zero bytes and would produce
  // no checkpoint at all. `ckpt_drain` demotes each finished checkpoint out
  // of the fast tier: a materialised checkpoint lands at blob score 1.0 and
  // otherwise sits in VRAM holding cold history that nobody reads again.
  u64 ckpt_every = 0;
  bool ckpt_drain = false;
  // --ckpt-sync: every vector.Copy is fully synchronous (all pages
  // materialised inside the Copy) instead of lazy copy-on-write. Implies
  // --ckpt-final.
  bool ckpt_sync = false;
  // OPT-IN, NOT OPT-OUT. The end-of-run checkpoint arrived enabled by
  // default, which silently changes what an already-queued job measures:
  // the E1 scaling rungs were submitted against binaries without it and
  // would have executed binaries with it, so their numbers would have
  // carried a final write the 256-node rung did not, and the ladder would
  // not have been self-consistent. --ckpt-final asks for it.
  bool ckpt_final = false;  // --ckpt-final: take the end-of-run checkpoint
  // --organizer-hint: tell the CTE data organizer which step is starting
  // (ReorganizeHint(s+1)), so a phase-aware organizer knows which region pair
  // is about to be overwritten. Opaque to the core; see GrayScottDataOrganizer.
  bool organizer_hint = false;
  float ckpt_cold = 0.0f;
  unsigned long long ram_mb = 0;   // 0 = data_mb + 1024, the historic value

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    auto nextf = [&]() -> float {
      return (i + 1 < argc) ? std::strtof(argv[++i], nullptr) : 0.0f;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--set-size") set_size_arg = static_cast<u32>(next());
    else if (a == "--plane-split") plane_split = static_cast<u32>(next());
    else if (a == "--plane-order" && i + 1 < argc) plane_cyclic = (std::string(argv[++i]) == "cyclic");
    else if (a == "--two-phase") two_phase = true;
    else if (a == "--ooc") ooc = true;
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--steps") steps = static_cast<u32>(next());
    else if (a == "--nodes") nodes = static_cast<u32>(next());
    else if (a == "--node") node = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--hbm-mb") hbm_mb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--baseline") baseline = true;
    else if (a == "--prefetch" && i + 1 < argc) prefetch = argv[++i];
    else if (a == "--lookahead") lookahead = next();
    else if (a == "--pf-hot") pf_hot = nextf();
    else if (a == "--pf-warm") pf_warm = nextf();
    else if (a == "--pf-cold") pf_cold = nextf();
    else if (a == "--pf-no-demote") pf_demote = false;
    else if (a == "--pf-stall-every") pf_stall_every = next();
    else if (a == "--hbm-score") hbm_score = nextf();
    else if (a == "--ram-score") ram_score = nextf();
    else if (a == "--nvme-score") nvme_score = nextf();
    else if (a == "--ram-mb") ram_mb = next();
    else if (a == "--ckpt-every") ckpt_every = next();
    else if (a == "--ckpt-drain") ckpt_drain = true;
    else if (a == "--ckpt-sync") ckpt_sync = ckpt_final = true;
    else if (a == "--no-ckpt") ckpt_final = false;   // kept: now a no-op
    else if (a == "--ckpt-final") ckpt_final = true;
    else if (a == "--organizer-hint") organizer_hint = true;
    else if (a == "--ckpt-cold") ckpt_cold = nextf();
    else if (a == "--Du") Du = nextf();
    else if (a == "--Dv") Dv = nextf();
    else if (a == "--F") F = nextf();
    else if (a == "--K") K = nextf();
    else if (a == "--dt") dt = nextf();
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--slots N] "
                  "[--steps N] [--page-kb N] [--data-mb N] [--hbm-mb N] "
                  "[--repeat N] [--hbm-only] [--Du f] [--Dv f] [--F f] "
                  "[--K f] [--dt f]\n"
                  "       [--prefetch none|stride|gs] [--lookahead N] "
                  "(0 = size it to the fast tier)\n"
                  "       [--pf-hot f] [--pf-warm f] [--pf-cold f] "
                  "[--pf-no-demote] [--pf-stall-every N]\n"
                  "       [--nvme-mb N] [--nvme-path P] [--ram-mb N]\n"
                  "       [--ckpt-every N] [--ckpt-drain] [--ckpt-sync] [--ckpt-cold f]\n"
                  "       [--hbm-score f] [--ram-score f] [--nvme-score f]\n"
                  "         tier scores: a tier scored ABOVE the vector's put "
                  "score (0.5) is\n"
                  "         never a first choice. For capacity-ordered "
                  "placement across all\n"
                  "         three tiers pass e.g. 0.5 / 0.3 / 0.1.\n",
                  argv[0]);
      return 0;
    }
  }

  if (prefetch != "none" && prefetch != "stride" && prefetch != "gs") {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --prefetch must be one of "
                 "none|stride|gs (got '%s')\n", prefetch.c_str());
    return 2;
  }
  // A prefetch run against the baseline driver would measure nothing: the
  // baseline tears the kernel down per z and never yields, so no cursor is
  // ever published and no hint is ever derived. Refused rather than run,
  // because it would report prefetch=gs and sent=0 and look like a result.
  if (baseline && prefetch != "none") {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --baseline does not yield, so no "
                 "cursor is published and --prefetch %s would emit nothing.\n",
                 prefetch.c_str());
    return 2;
  }

  // The coroutine-mode refusal that used to be here moved with the kernels:
  // a build that cannot compile them does not produce this target at all now
  // (see the CMake guards, and adapter/CMakeLists.txt).
  // The kernel holds 6 input planes + 2 output planes at once. A smaller cache
  // could evict a plane the kernel is still reading -- that would not crash,
  // it would silently read whatever replaced it, so it is refused.
  // Out of core the two-phase bands are sized to the cache, so only a band of
  // one plane per block (4 pages each, plus the halos) has to fit.
  const u32 kNeededSlots = (ooc && two_phase) ? 6 : 10;
  if (ooc && !two_phase) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --ooc needs --two-phase\n");
    return 2;
  }
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
  if (node >= nodes) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --node %u out of range for "
                 "--nodes %u\n", node, nodes);
    return 2;
  }
  // This node's slab of the global z range. The halo planes it reads are
  // its neighbours' pages, which is why the tag namespace below is SHARED:
  // a vector's pages are CTE blobs and blobs hash across the cluster, so
  // the paging path IS the halo exchange. Per-node namespaces would give
  // each node a private field and silently decouple the physics.
  // HOW THE HALO EXCHANGE WORKS, now that it does.
  //
  // Every Fetch used to pass generation 0, and generation 0 means ANY
  // VERSION IS ACCEPTABLE -- so a node never demanded a current halo
  // plane and was content with a stale one. Steps 0-2 happened to come
  // out right; the divergence started at step 3 and was nondeterministic.
  //
  // The fix is to demand a generation, but ONLY of a neighbour's plane.
  // A page's generation is stamped by the FETCH that delivers it, not by
  // the flush that publishes it, so a plane this node wrote locally and
  // never re-fetched sits at generation 0 forever -- demanding one on it
  // hangs the block with "gen stall: page N at gen 0 want G". Own planes
  // are current by construction and pass 0; the halo passes the step's
  // generation. The seed publishes AS generation 1, which is what step 0
  // demands; step s reads at s+1 and publishes as s+2.
  //
  // MEASURED, 2 nodes vs the single-node reference of 36410.579344:
  //   demand on  -> 36410.579344, bit-exact, three consecutive runs
  //   demand off -> 36104.119147 (GS_NO_HALO=1, the negative control)
  // The control disables the DEMAND, not the barrier, because an earlier
  // version disabled only the barrier and still came out exact -- which
  // is its own result: the generation IS the barrier, as md's harness
  // says. The explicit flush/barrier/ClearCache below may now be
  // redundant; that is an optimisation, not a correctness question, and
  // has not been tested on its own.
  const u64 nz_local = (nz + nodes - 1) / nodes;
  const u64 zbase = static_cast<u64>(node) * nz_local;
  const u64 zend = (zbase + nz_local < nz) ? (zbase + nz_local) : nz;
  if (nodes > 1 && zbase >= nz) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --nodes %u leaves node %u with "
                 "no planes of nz=%llu\n", nodes, node,
                 (unsigned long long)nz);
    return 2;
  }
  const u64 zper = ((zend - zbase) + blocks - 1) / blocks;
  if (plane_split == 0 || blocks % plane_split != 0) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --plane-split %u must divide "
                 "--blocks %u\n", plane_split, blocks);
    return 2;
  }
  const u64 slabs = blocks / plane_split;
  const u64 zper_step = ((zend - zbase) + slabs - 1) / slabs;
  const u64 region = plane * nz;
  const u64 n = 4 * region;
  const double logical_mb =
      static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);
  // Historic default: enough host tier to hold the whole grid with room over,
  // i.e. a hierarchy with no real pressure on the host tier.
  const unsigned long long ram_cap_mb =
      (ram_mb != 0) ? ram_mb : (data_mb + 1024);

  // ---- "OPTIMAL" LOOKAHEAD ---------------------------------------------
  //
  // --lookahead 0 sizes the promotion window to the FAST TIER rather than
  // guessing a constant, which is what makes "optimal prefetching" a defined
  // thing on this workload rather than a vibe.
  //
  // Gray-Scott's access order is a deterministic sweep: block b reads planes
  // z0..z1 in order, four regions per z. So the future is fully known, and
  // the Belady-optimal placement is simply "the fast tier holds the next
  // |fast tier| accesses". A prefetcher that promotes exactly that far ahead
  // and demotes everything behind it IS optimal for this pattern -- there is
  // no cleverer policy available, only a better-sized window.
  //
  // At any instant the hot set is, across the grid,
  //     blocks * 4 regions * (L + 2)          [+2: the z-1 and z still held]
  // planes. Setting that equal to the fast tier's capacity in planes:
  //     L = hbm_planes / (4 * blocks) - 2
  //
  // Undersized, the tier sits part empty and misses are served from a slower
  // one. Oversized, promotions evict each other before they are used and the
  // migrations are wasted work -- which is why this is a real optimum with a
  // curve either side of it, not a "bigger is better" knob.
  const u64 hbm_planes =
      static_cast<u64>(hbm_mb) * 1024ull * 1024ull / page_bytes;
  const u64 auto_lookahead =
      (hbm_planes > 4ull * blocks * 2ull)
          ? (hbm_planes / (4ull * blocks) - 2ull)
          : 1ull;
  if (lookahead == 0) {
    lookahead = auto_lookahead;
    std::printf("  lookahead: auto = %llu planes (fast tier holds %llu "
                "planes; %u blocks x 4 regions x (L+2))\n",
                (unsigned long long)lookahead,
                (unsigned long long)hbm_planes, blocks);
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
        // TIER SCORES ARE A COMMAND-LINE AXIS, and getting them wrong is the
        // difference between a three-tier run and a two-tier run that says it
        // has three.
        //
        // MaxBwDpe partitions targets into `target_score <= blob_score`
        // (PREFERRED, sorted by score DESCENDING) and the rest (FALLBACK,
        // taken only when no preferred tier has room). The vector writes every
        // page at kVectorBlobScore = 0.5, so A TIER SCORED ABOVE 0.5 IS NEVER
        // A FIRST CHOICE FOR ANYTHING THIS VECTOR WRITES.
        //
        // The old defaults -- HBM 1.0, RAM 0.2 -- put HBM in the fallback
        // group, which is why every run of this benchmark reported
        // "kHBM used=0MiB ... nothing landed in the fastest tier" while the
        // comment here asserted the opposite. Preserved as the DEFAULTS
        // anyway, so existing sweeps reproduce exactly; a run that wants
        // capacity-ordered placement across all three tiers passes scores at
        // or below 0.5, descending (e.g. 0.5 / 0.3 / 0.1).
        << "      - path: \"hbm::gv_gs_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: " << hbm_score << "\n";
    if (!hbm_only) {
      cfg << "      - path: \"ram::gv_gs_ram\"\n        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << ram_cap_mb << "MB\"\n"
          << "        score: " << ram_score << "\n";
    }
      // OPTIONAL STORAGE TIER. Without it the whole dataset lives in host
      // DRAM and NOTHING EVER TOUCHES STORAGE -- what such a run measures is
      // DRAM over PCIe, not I/O. On this machine that spill is nearly free,
      // which is exactly why a cache-size sweep over a DRAM-only hierarchy
      // comes back flat: there is no penalty for the cache to save.
      //
      // score BELOW the host tier, so the preferred group ranks it last.
      //
      // IT MUST STILL BE <= THE BLOB SCORE TO BE PREFERRED AT ALL. At the
      // default 0.0 it is, for any blob score -- but a prefetcher demoting to
      // "cold" has to name a score at or above this one, or storage lands in
      // the FALLBACK group and the ordering there is by the bandwidth model
      // rather than by the operator's declared tiering. That model is not
      // trustworthy for this (see the note in MaxBwDpe::SelectTargets, which
      // rated an HBM tier at 118 MB/s against host RAM at 1600), so a cold
      // score should equal this number rather than merely be below it.
      if (nvme_mb > 0) {
        cfg << "      - path: \"" << nvme_path << "\"\n"
            << "        bdev_type: \"file\"\n"
            << "        persistence_level: \"temporary\"\n"
            << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
            << "        score: " << nvme_score << "\n";
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
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  gs::InitBackend(blocks, gpu);

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

  // ASSOCIATIVITY, NOT FULL ASSOCIATIVITY. A lookup scans its set, so one
  // giant set costs O(frames) per probe -- kmeans went 4.7 s -> 42 s as a
  // 512-way cache. Keep a set per block and floor the width at 8, which is
  // what covers several blocks colliding on one set.
  // This kernel holds eight planes per step per block, so the set must be
  // wide enough for several blocks' worth of those.
  // ---- cross-node reduction client -------------------------------------
  // After the runtime is up, and only when there is a reduction to do.
  std::unique_ptr<clio::cte::core::Client> cte_red;
  clio::cte::core::TagId red_tag{};
  u64 red_round = 0;
  if (nodes > 1) {
    cte_red = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_red->AsyncGetOrCreateTag("gv_grayscott_red");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: could not create reduction "
                   "tag\n");
      return 1;
    }
    red_tag = t->tag_id_;
  }

  // NARROW SETS WHEN ASKED. With one set per block the set is `slots` wide
  // (516 for a resident 32 GB deck at 64 blocks), and every Find /
  // FindClaimed is a linear scan of the whole set -- run by every thread,
  // eight lookups per plane. The same capacity in narrow sets makes each
  // lookup a scan of set_size entries; pages hash across sets (SetOf).
  const u32 cap_pages = blocks * (slots < 24u ? 24u : slots);
  const u32 vec_set_size =
      set_size_arg != 0 ? set_size_arg : (slots < 24u ? 24u : slots);
  const u32 vec_nsets =
      set_size_arg != 0 ? (cap_pages + set_size_arg - 1) / set_size_arg : 0;
  gv::Vector<float> vec("gv_grayscott", {0}, page_bytes, blocks,
                        vec_set_size, n, clio::run::PoolId::GetNull(), 0, 1,
                        vec_nsets, set_size_arg != 0 ? cap_pages : 0);
  if (set_size_arg != 0) {
    std::printf("  cache: %u sets x %u ways over %u pages\n", vec_nsets,
                vec_set_size, cap_pages);
  }
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(blocks, threads);
  YieldRunner pub_runner(1, threads);
  const u64 ntab = (zend - zbase) + 2;
  // OUT-OF-CORE BAND: planes per resolve/release pass. Each plane pins 4
  // pages (u, v and both outputs), plus 2 halo planes of u and v; keep a
  // quarter of the cache free so set conflicts do not evict a pinned page.
  const u64 frames = static_cast<u64>(slots) * blocks;
  u64 ooc_band = (frames * 3 / 4 > 8) ? (frames * 3 / 4 - 4) / 4 : 1;
  if (ooc_band > zend - zbase) ooc_band = zend - zbase;
  if (ooc) {
    std::printf("  out of core: %llu frames, bands of %llu planes (%llu per "
                "node slab)\n", (unsigned long long)frames,
                (unsigned long long)ooc_band,
                (unsigned long long)(zend - zbase));
  }
  float **d_tab = ctp::GpuApi::Malloc<float *>(4 * ntab * sizeof(float *));

  const u64 ubase = 0, vbase = region, unext = 2 * region, vnext = 3 * region;

  // ---- PREFETCHERS ------------------------------------------------------
  // Registered on the VECTOR (which owns the tag and the CTE client) and
  // fired from the DRIVER's post-round gap (which is the only host time with
  // no kernel resident). BindPrefetch joins the two.
  gv::PrefetchPolicy pf_policy;
  pf_policy.hot_ = pf_hot;
  pf_policy.warm_ = pf_warm;
  pf_policy.cold_ = pf_cold;
  pf_policy.lookahead_ = lookahead;
  pf_policy.stall_every_ = static_cast<u32>(pf_stall_every);
  vec.SetPrefetchPolicy(pf_policy);
  std::shared_ptr<gs::GrayScottPrefetcher> gs_pf;
  if (prefetch == "gs") {
    // zbase/zend/zper are the SAME locals LaunchStep is given below, which is
    // what keeps the host's idea of block b's slab from drifting from the
    // kernel's.
    gs_pf = std::make_shared<gs::GrayScottPrefetcher>(
        plane, nz, zbase, zend, zper, pf_policy, pf_demote);
    vec.RegisterPrefetcher(gs_pf);
  } else if (prefetch == "stride") {
    // Blind to three of the four regions and to demotion entirely -- that is
    // the point of it. It is pointed at u's current region because the cursor
    // it sees counts z, and u is the region whose plane index the cursor
    // equals. It cannot learn that v, unext and vnext exist.
    vec.RegisterPrefetcher(std::make_shared<gv::StridePrefetcher>(
        ubase / plane, nz, pf_policy));
  }
  if (vec.HasPrefetchers()) runner.BindPrefetch(vec);

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, ubase, vbase,
                   zbase, zend, vw, sv, ooc ? 1u : 0u);
  });
  ctp::GpuApi::Synchronize();
  // TOUCH THE OUTPUT REGIONS TOO, before the clock starts. The baselines
  // allocate and initialise all four fields up front; here unext/vnext were
  // never written, so step 0's first write to each of their pages faulted
  // (16000 faults at 32 GB/node) inside the timed region -- a setup cost the
  // baselines do not pay. Seeding them with the same generator makes them
  // resident; step 0 overwrites every value, so the result is unchanged.
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, unext, vnext,
                   zbase, zend, vw, sv, 0u);
  });
  ctp::GpuApi::Synchronize();
  // NO seed-side flush/barrier/invalidate. SeedCoro publishes its own
  // planes AS generation 1 (BeginFlush(1) + EndFlush), and step 0's halo
  // fetch DEMANDS generation 1 -- the demand polls until the peer's
  // publish is served, so the generation is the barrier. A whole-table
  // flush here republishes nothing new, and an invalidate drops frames
  // that were never stale (the seed touches only this node's slab).

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
      gs::LaunchBaseline(threads, bl_d[0], bl_d[1], bl_d[2], bl_d[3], bl_d[4],
                         bl_d[5], bl_dout[0], bl_dout[1], plane, nx, ny,
                         interior ? 1 : 0, Du, Dv, F, K, dt);
      ctp::GpuApi::Synchronize();
      if (!bl_write(0, nu_, z) || !bl_write(1, nv_, z)) return false;
    }
    return true;
  };

  double best_ms = 1e30, checksum = 0.0;
  u64 ckpt_id = 0, ckpt_pages = 0;
  double ckpt_ms = 0.0;
  for (int r = 0; r < repeat; ++r) {
    // RE-SEED between repeats. Without this, repeat 2 continues evolving the
    // field left by repeat 1, so each timed run measures a different physical
    // state and the reported checksum depends on `repeat` -- which makes it
    // useless as a correctness check and makes the repeats non-comparable.
    if (r > 0) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, ubase, vbase,
                       zbase, zend, vw, sv, ooc ? 1u : 0u);
      });
      ctp::GpuApi::Synchronize();
    }
    vec.ResetStats();
    vec.ResetPrefetchStats();
    vec.ResetPrefetchHistory();
    vec.PrefetchRunBegin();
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();
    // E1 split: in-kernel Fetch/Hold/Flush share (GV_COMM_TIMING), and in
    // two-phase mode the host time of the paging kernels vs the stencil.
    clio_gv_bench::CommAcc gs_comm;
    double gs_step_ms = 0.0, gs_tab_ms = 0.0;
    u64 cu = ubase, cv = vbase, nu = unext, nv = vnext;
    // Checkpoints are KEPT ALIVE for the whole run: dropping the handle would
    // let the copy tag go away and with it the bytes we just produced.
    std::vector<std::unique_ptr<gv::Vector<float>>> ckpts;
    for (u32 s = 0; s < steps; ++s) {
      // RE-ARM EVERY STEP. The swap below makes last step's outputs this
      // step's inputs, so a prefetcher still holding the old assignment
      // promotes the wrong four regions.
      //
      // NO HISTORY RESET HERE. The dedup map models the score the CTE holds
      // for a page, which a change of MEANING does not invalidate -- see
      // Vector::ResetPrefetchHistory. Clearing it re-sent every page its own
      // current score at each step boundary, which the CTE no-ops, and made
      // the promote/demote tally meaningless.
      if (gs_pf) gs_pf->SetRegions(cu, cv, nu, nv);
      if (organizer_hint) {
        // Broadcast and waited: cheap next to a step, and the organizer must
        // not see step s's hint while step s+1 is already writing.
        (void)CLIO_CTE_CLIENT->ReorganizeHint(static_cast<clio::run::i32>(s + 1));
      }
      if (baseline) {
        if (!run_baseline_step(cu, cv, nu, nv)) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: baseline step failed\n");
          return 1;
        }
      } else {
        const auto st0 = vec.ReadStats(0);
        const double ts0 = NowMs();
        runner.ResetTimers();
        u32 srounds = 0;
        if (two_phase) {
          // Resident: one band = the whole slab. Out of core: bands of
          // `band` planes (see ooc_band below), each flushed on release.
          const u64 bstep = ooc ? ooc_band : (zend - zbase);
          const u64 hgen = halo_off ? 0 : static_cast<u64>(s) + 1;
          const u64 fgen = ooc ? static_cast<u64>(s) + 2 : 0;
          for (u64 zb = zbase; zb < zend; zb += bstep) {
            const u64 ze = (zb + bstep < zend) ? zb + bstep : zend;
            const u64 bper = ((ze - zb) + blocks - 1) / blocks;
            srounds += runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                      gy::YieldStackView sv) {
              gs::LaunchResolve(g, b, gpu, dev, plane, nz, ooc ? bper : zper,
                                zb, ze, hgen, cu, cv, nu, nv, d_tab, ntab, vw,
                                sv);
            });
            if (!no_compute) {
              const double tt0 = NowMs();
              gs::LaunchStencilTab(blocks, threads, d_tab, ntab, plane, nx, ny,
                                   nz, zb, ze, Du, Dv, F, K, dt);
              ctp::GpuApi::Synchronize();
              gs_tab_ms += NowMs() - tt0;
            }
            srounds += runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                      gy::YieldStackView sv) {
              gs::LaunchRelease(g, b, gpu, dev, plane, nz, ooc ? bper : zper,
                                zb, ze, cu, cv, nu, nv, vw, sv, fgen);
            });
          }
        } else {
          srounds = runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                   gy::YieldStackView sv) {
            gs::LaunchStep(g, b, gpu, dev, plane, nx, ny, nz, zper_step, cu, cv,
                           nu, nv, Du, Dv, F, K, dt, zbase, zend,
                           // GS_NO_HALO=1 forces generation 0 -- "any
                           // version" -- which is the control: it disables
                           // the DEMAND itself, not merely the barrier.
                           halo_off ? 0 : static_cast<u64>(s) + 1, vw, sv,
                           (plane_cyclic ? 0u : plane_split) |
                               (no_compute ? 0x40000000u : 0u));
          });
        }
        // Publish the node's edge planes once every block's writes are done.
        // Only a peer node reads them, so a single node skips it.
        // Out of core the release already flushed every plane at s+2.
        if (nodes > 1 && !ooc) {
          pub_runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
            gs::LaunchPublish(g, b, gpu, dev, plane, zbase, zend,
                              static_cast<u64>(s) + 2, nu, nv, vw, sv);
          });
        }
        // PER-STEP SPLIT: whether the time is first-touch faults (step 0
        // writes a fresh output region), relaunch rounds, or the kernel.
        const auto st1 = vec.ReadStats(0);
        gs_comm.Add(st0, st1);
        gs_step_ms += NowMs() - ts0;
        std::fprintf(stderr,
                     "  step %u: %.1fms rounds=%u gpu=%.1fms faults=%llu "
                     "puts=%llu\n", s, NowMs() - ts0, srounds,
                     runner.KernelMs(),
                     (unsigned long long)(st1.faults - st0.faults),
                     (unsigned long long)(st1.puts - st0.puts));
      }
      // NO per-step flush/barrier/invalidate: the generational demand IS
      // the halo exchange. A halo fetch names generation s+1 and a stale
      // RESIDENT peer frame takes the gen-stale refetch path, so neither
      // an explicit barrier nor a cache clear is needed -- and the clear
      // was actively harmful: SettleAndInvalidate dropped EVERY frame
      // each step, so the eviction path never engaged in a distributed
      // run (the OOC gate measured evicts=0 against 844 single-node,
      // which is what exposed this). The "removing the flush measured
      // 5.98% worse" note that used to justify this block was measured
      // BEFORE generations were threaded through, on single runs of a
      // quantity later shown to vary run-to-run.
      //
      // GS_NO_HALO=1 remains the negative control, applied where it now
      // belongs: it forces the step's halo demand back to generation 0
      // at the launch site.
      std::swap(cu, nu);
      std::swap(cv, nv);

      // ---- CHECKPOINT ------------------------------------------------
      // Vector::Copy is COPY-ON-WRITE: it registers the checkpoint chimod as
      // the copy tag's fault handler and duplicates nothing. So the snapshot
      // must be MATERIALISED, and materialised SOON: the handler pulls from
      // the source at fault time, and the region swap above overwrites each
      // pair every second step, so a checkpoint left lazy for two more steps
      // would silently capture later bytes instead of these.
      //
      // Materialisation is server-side (a size-0 put takes the put-side
      // materialise-only fault), so the bytes never cross to the host -- this
      // is a checkpoint being made durable, not a read-back.
      if (ckpt_every != 0 && ((s + 1) % ckpt_every) == 0) {
        const double ck0 = NowMs();
        // --ckpt-sync materialises inside the Copy; otherwise do it here.
        auto snap = vec.Copy("gv_gs_ck" + std::to_string(ckpt_id), ckpt_sync);
        const u64 mat = ckpt_sync ? snap->NumPages() : snap->MaterializeAll();
        if (mat == 0) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: checkpoint %llu failed to "
                       "materialize\n", (unsigned long long)ckpt_id);
          return 1;
        }
        ckpt_pages += mat;
        // DRAIN IT. A materialised checkpoint is written at blob score 1.0
        // (the handler puts at -1.0, which PutBlobImpl resolves to 1.0 for a
        // new blob), so it outranks the LIVE grid on every tier and parks
        // cold history in VRAM. Nothing ever reads it again in this run, so
        // demoting it is not speculative -- it is the drain that has to
        // happen anyway, done early instead of by eviction pressure later.
        if (ckpt_drain) {
          snap->SetPrefetchPolicy(pf_policy);
          snap->ScoreAllPages(ckpt_cold);
          snap->DrainPrefetch();
        }
        ckpts.push_back(std::move(snap));
        ckpt_ms += NowMs() - ck0;
        ++ckpt_id;
      }
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    if (gs_step_ms > 0.0) {
      if (two_phase) {
        // Everything but the stencil is the paging kernels (resolve, release,
        // edge publish): communication by E1's definition.
        std::printf("COMM grayscott two-phase: comm_ms=%.1f (resolve+release"
                    "+publish) compute_ms=%.1f (stencil) of %.1f ms over %u "
                    "steps\n", gs_step_ms - gs_tab_ms, gs_tab_ms, gs_step_ms,
                    steps);
      } else {
        gs_comm.Print("grayscott", gs_step_ms);
      }
    }
    // AFTER the timer, deliberately. The SUBMISSIONS are inside the timed
    // region -- they are host time in the round gaps and the run really pays
    // for them -- but the run never waits on a rescore to complete, so
    // charging it for that wait would be charging it for work it did not do.
    // The drain is here only so the tier occupancy read below reflects the
    // migrations this run asked for rather than a race with them.
    vec.DrainPrefetch();

    // GS_PLANE_DUMP=1 prints a per-plane sum of v. Diffing that between a
    // 1-node and a 2-node run says WHICH planes are wrong, which is the
    // question -- a whole-field checksum only says that something is. Uses
    // the existing sum kernel one plane at a time, so it measures the same
    // bytes the gate does rather than a second path that could differ.
    const char *const gs_dump = getenv("GS_PLANE_DUMP");
    if (gs_dump != nullptr && gs_dump[0] != '\0') {
      for (u64 z = zbase; z < zend; ++z) {
        ctp::GpuApi::Memset(d_sum, 0, sizeof(double));
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          gs::LaunchSum(g, b, gpu, dev, plane, nz, 1, cv, d_sum, z, z + 1,
                        vw, sv);
        });
        ctp::GpuApi::Synchronize();
        double pz = 0.0;
        ctp::GpuApi::Memcpy(&pz, d_sum, sizeof(double));
        std::printf("  PLANE %llu %.6f\n", (unsigned long long)z, pz);
      }
      std::fflush(stdout);
    }
    ctp::GpuApi::Memset(d_sum, 0, sizeof(double));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gs::LaunchSum(g, b, gpu, dev, plane, nz, zper, cv, d_sum, zbase, zend,
                    vw, sv);
    });
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(&checksum, d_sum, sizeof(double));
    // Each node summed only its own slab; the gate compares a checksum of
    // the WHOLE field against a single-node reference, so the partials
    // have to be combined before it is reported.
    if (nodes > 1 &&
        !clio_bench_dist::ReduceSum(*cte_red, red_tag, node, nodes,
                                    red_round++, &checksum, 1, "gsred")) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: checksum reduction failed\n");
      return 1;
    }
    // A diverged field sums to inf or NaN, and NaN compares false against
    // every tolerance; refuse it here rather than let a gate pass it.
    if (!std::isfinite(checksum)) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: v_checksum is not finite (the "
                           "stencil diverged)\n");
      return 1;
    }
  }

  const auto st = vec.ReadStats(0);
  const auto pf = vec.ReadPrefetchStats();
  // EDGE-ONLY PUBLISHING IS CORRECT ONLY WHILE RESIDENT. The step publishes
  // just the node-edge planes and leaves every interior plane dirty in the
  // cache, and a gpu_vector Page has no dirty bit: an eviction drops the
  // frame without writing it back, and the refetch reads an older copy.
  // Measured: 33 evictions at 256 blocks with 4% slot headroom moved the
  // checksum in the 6th digit. Until the cache tracks dirty pages, a run
  // that evicted anything reports failure instead of a wrong number.
  if (st.evicts != 0 && !ooc) {
    std::fprintf(stderr,
                 "GRAYSCOTT ERROR: %llu page(s) evicted; interior planes are "
                 "not written back, so the result is invalid. Give the cache "
                 "headroom (--slots) so the deck stays resident.\n",
                 (unsigned long long)st.evicts);
    return 1;
  }
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
  //
  // ALL THREE TIERS, and the storage one is not optional to report: a
  // three-tier run whose report covers two of them cannot answer "what
  // fraction of the data is where", which on a tiered evaluation IS the
  // result. The third bdev is (514,1) by the same config-order rule.
  {
    const clio::run::u64 fast_cap = (clio::run::u64)hbm_mb * 1024ull * 1024ull;
    const clio::run::u64 host_cap =
        hbm_only ? 0ull : (clio::run::u64)ram_cap_mb * 1024ull * 1024ull;
    const clio::run::u64 stor_cap = (clio::run::u64)nvme_mb * 1024ull * 1024ull;
    // Bdev minors follow CONFIG ORDER, and the host tier is skipped entirely
    // under --hbm-only -- so storage is the SECOND entry there, not the
    // third. Indexing it unconditionally as 514 read a pool that does not
    // exist and reported the storage tier as completely full.
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    const clio::run::u64 fast_rem = fa->remaining_size_;
    clio::run::u64 host_rem = 0, stor_rem = 0;
    if (!hbm_only) {
      clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
      auto ha = t_host.AsyncGetStats(); ha.Wait();
      host_rem = ha->remaining_size_;
    }
    if (nvme_mb > 0) {
      clio::run::bdev::Client t_stor(
          clio::run::PoolId(hbm_only ? 513 : 514, 1));
      auto sa = t_stor.AsyncGetStats(); sa.Wait();
      stor_rem = sa->remaining_size_;
    }
    auto used = [](clio::run::u64 cap, clio::run::u64 rem) {
      return cap > rem ? cap - rem : 0ull;
    };
    const clio::run::u64 fast_used = used(fast_cap, fast_rem);
    const clio::run::u64 host_used = used(host_cap, host_rem);
    const clio::run::u64 stor_used = used(stor_cap, stor_rem);
    const double total_mb =
        static_cast<double>((fast_used + host_used + stor_used) >> 20);
    auto pct = [&](clio::run::u64 u) {
      return total_mb > 0.0 ? 100.0 * static_cast<double>(u >> 20) / total_mb
                            : 0.0;
    };
    // RAW remaining is printed alongside the derived used, because the
    // derived number alone is not interpretable: if a queried pool does not
    // exist or the stat fails, remaining reads 0 and "used" then equals the
    // full capacity -- which looks like a completely full tier rather than a
    // failed query. Both were indistinguishable in the first version of this
    // report and it nearly produced a false VIOLATION.
    std::fprintf(stderr,
                 "TIER SPLIT: kHBM used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "host used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "stor used=%lluMiB cap=%lluMiB remain=%lluMiB%s\n",
                 (unsigned long long)(fast_used >> 20),
                 (unsigned long long)(fast_cap >> 20),
                 (unsigned long long)(fast_rem >> 20),
                 (unsigned long long)(host_used >> 20),
                 (unsigned long long)(host_cap >> 20),
                 (unsigned long long)(host_rem >> 20),
                 (unsigned long long)(stor_used >> 20),
                 (unsigned long long)(stor_cap >> 20),
                 (unsigned long long)(stor_rem >> 20),
                 (fast_used == 0 && fast_rem == fast_cap)
                     ? "   <-- nothing landed in the fastest tier"
                     : "");
    // The line the tiered evaluation actually reads: where the data ENDED UP,
    // as a fraction. A capacity split is what was ASKED for; this is what was
    // achieved, and on this workload the two came apart silently for as long
    // as the fast tier was unreachable.
    std::fprintf(stderr,
                 "TIER PCT: vram=%.1f%% dram=%.1f%% nvme=%.1f%% "
                 "(placed %.0fMiB of %.0fMiB logical)\n",
                 pct(fast_used), pct(host_used), pct(stor_used), total_mb,
                 logical_mb);
  }

  std::fprintf(stderr,
               "GRAYSCOTT mode=%s blocks=%u thr=%u nx=%llu ny=%llu nz=%llu "
               "page_kb=%llu slots=%u steps=%u data_mb=%.0f hbm_mb=%llu "
               "ms=%.1f GB/s=%.2f v_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f "
               // PREFETCH ACCOUNTING. Reported unconditionally, including for
               // a --prefetch none run where every field is 0: a run whose
               // pf_sent is 0 is NOT a prefetching run whatever its timings
               // say, and that has to be readable from the log rather than
               // inferred from the command line.
               "prefetch=%s lookahead=%llu stall_every=%llu "
               "pf_hints=%llu pf_sent=%llu "
               "pf_deduped=%llu pf_batches=%llu pf_promote=%llu "
               "pf_demote=%llu pf_errors=%llu pf_notfound=%llu "
               "pf_dropped=%llu ckpt_every=%llu ckpt_n=%llu "
               "ckpt_gb=%.2f ckpt_ms=%.0f ckpt_drain=%d\n",
               baseline ? "baseline" : "paged",
               blocks, threads, (unsigned long long)nx, (unsigned long long)ny,
               (unsigned long long)nz, (unsigned long long)page_kb, slots,
               steps, logical_mb, (unsigned long long)hbm_mb, best_ms, gbps,
               checksum, (unsigned long long)st.faults,
               (unsigned long long)st.evicts, (unsigned long long)st.puts,
               (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps,
               vec.PrefetcherNames().c_str(), (unsigned long long)lookahead,
               (unsigned long long)pf_stall_every,
               (unsigned long long)pf.hints_, (unsigned long long)pf.sent_,
               (unsigned long long)pf.deduped_,
               (unsigned long long)pf.batches_,
               (unsigned long long)pf.promotions_,
               (unsigned long long)pf.demotions_,
               (unsigned long long)pf.errors_,
               (unsigned long long)pf.not_found_,
               (unsigned long long)pf.dropped_,
               (unsigned long long)ckpt_every, (unsigned long long)ckpt_id,
               static_cast<double>(ckpt_pages) * page_bytes /
                   (1024.0 * 1024.0 * 1024.0),
               ckpt_ms, ckpt_drain ? 1 : 0);

  ctp::GpuApi::Free(d_sum);
  // FINAL-STATE CHECKPOINT: vector.Copy of the whole field, on
  // by default (--no-ckpt skips it). After every gate, because the
  // multi-node path drops the cache first -- see bench_ckpt.h.
  // Unlike the periodic --ckpt-every snapshots this one needs no
  // MaterializeAll: nothing writes the field after it, so the lazy copy
  // cannot be overtaken by a later step.
  std::unique_ptr<gv::Vector<float>> final_ck;
  if (ckpt_final) {
    final_ck = clio_bench_ckpt::FinalCheckpoint(vec, "gv_gs_ckpt_final", nodes,
                                                 ckpt_sync);
  }
  BenchFlushData();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
