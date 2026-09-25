#if CTP_ENABLE_SYCL
#define CLIO_SYCL_KERNEL_TU 1
#endif
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
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "../bench_memcpy_probe.h"

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
static constexpr u32 kYieldLaneBytes = 1024;


/*
 * THE DEVICE CODE IS NOT HERE ANY MORE.
 *
 * The workload -- PointVal, SeedCoro, AssignCoro, the baseline body, the
 * centroid update -- lives in kmeans_kernels.h, in ONE copy compiled by both
 * backends. The launches live in cuda/ and sycl/, which differ only in how a
 * grid is submitted. This file is the host driver and is now ordinary C++:
 * no kernels, no `<<<>>>`, no device pass to keep out of.
 *
 * See kmeans_launch.h for why the seam is at the launch and not somewhere
 * tidier.
 */
#include "../gv_launch_bounds.h"
#include "kmeans_kernels.h"
#include "kmeans_launch.h"

namespace clio::gv_bench::kmeans {
/** The assignment pass with block-private accumulators (see AssignCoro):
 *  bsums holds grid.x * k * dims floats and bcounts grid.x * k counters,
 *  both zeroed by the caller before every pass. Yieldable. */
void LaunchAssignTiled(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v,
                       u64 per, u64 page_elems, u32 dims, u32 k,
                       const float *cent, float *sums, unsigned *counts,
                       float *bsums, unsigned *bcounts, View vw, StackView sv);
}  // namespace clio::gv_bench::kmeans

namespace kb = clio::gv_bench::kmeans;
using kb::PointVal;

// HOST DRIVER ONLY BELOW THIS LINE.
//
// There is no device code left in this file, but under CUDA it is still
// compiled BY the CUDA compiler (add_cuda_executable compiles every source
// as CUDA), and that device pass member-checks host bodies -- gv::Vector and
// the CTE client are `#if !CTP_IS_DEVICE_PASS` and simply do not exist
// there. Under SYCL this guard is transparent: the driver is a plain C++ TU,
// so CTP_IS_DEVICE_PASS is 0 and everything below compiles once.
#if CTP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

/* =====================================================
 * THE WORKLOAD, on the new coroutine API. Ordinary return
 * types, ordinary locals, one marker per suspending call.
 * The `, clio::co::Ctx &_cy` on each signature and at each
 * call site is appended by clio-coroc, not written here.
 * ===================================================== */
namespace clio::gv_bench::kmeans {

/** Seed: write the point set into the vector, one page at a time.
 *
 *  `base_idx` is this node's offset into the GLOBAL point set. The vector
 *  index stays local -- each node's vector holds only its own shard -- but the
 *  VALUE written must come from the global index, or the union of the shards
 *  is not the single-node point set and the distributed run is solving a
 *  different problem than the reference it is gated against. Zero for a
 *  single-node run, which is therefore unchanged.
 */
CTP_GPU_FUN CLIO_COROC_INLINE void SeedCoro(gv::DeviceVector<float> v, u64 per,
                              u64 page_elems, u32 dims, u32 k, u64 base_idx,
                              u32 block, u32 publish) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    CO_AWAIT(v.CoFetch(0, base + off, n));
    auto h = CO_AWAIT(v.CoHoldPage(base + off, n, /*write=*/true));
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = PointVal(base_idx + base + off + i, dims, k);
    }
    // Collective: name the page just written -- only when asked. A node's
    // points are private and a resident cache holds them all, so writing the
    // whole seed back is 32 GB of puts per node that nothing reads; on 4
    // nodes it exhausted node 0's 64 copy streams and hung setup.
    if (publish != 0) CO_AWAIT(v.CoBeginFlush(0, base + off, n));
    // Fetch is the pinner; UnpinRange is the releaser.
    v.UnpinRange(base + off, n);
  }
  // Collect every flush started above: only explicit flushes write data back
  // now (drops refuse dirty pages), so a seeded page left in flight or left
  // to eviction would simply be lost.
  if (publish != 0) CO_AWAIT(v.CoEndFlush());
}

// ---- WORK-GROUP TILE (SYCL) ---------------------------------------------
//
// THE BASELINES TILE IN LOCAL MEMORY AND THIS DID NOT. Their AssignTiled
// accumulates each point into a work-group tile in local memory and flushes
// the tile to global memory once per group; AssignCoro added every point into
// a per-block slice in GLOBAL memory, 33 device-scope atomics per point. With
// the deck resident (0 faults) that alone left Eternia 2.2x behind MPI at the
// same 1024 x 256 grid -- a different algorithm, not the cost of paging.
//
// The tile is claimed AT THE POINT OF USE, per page, and flushed before the
// next suspend point, so no local-memory pointer is ever stored in a
// coroutine frame or carried across a relaunch. The point loop between
// CoHoldPage and UnpinRange never suspends, which is what makes that legal.
// CUDA keeps the global path (KmTile returns null): its launch bounds and
// __shared__ budget are a separate question and nothing here changes it.
constexpr u32 kKmTileFloats = 1024;   // k * dims ceiling for the tile
constexpr u32 kKmTileCounts = 64;     // k ceiling for the tile
#if CTP_ENABLE_SYCL && defined(__SYCL_DEVICE_ONLY__)
CTP_GPU_FUN inline float *KmTileSums() {
  auto g = ::sycl::ext::oneapi::this_work_item::get_work_group<1>();
  return *::sycl::ext::oneapi::group_local_memory_for_overwrite<
      float[kKmTileFloats]>(g);
}
CTP_GPU_FUN inline unsigned *KmTileCounts() {
  auto g = ::sycl::ext::oneapi::this_work_item::get_work_group<1>();
  return *::sycl::ext::oneapi::group_local_memory_for_overwrite<
      unsigned[kKmTileCounts]>(g);
}
template <typename T>
CTP_GPU_FUN inline void KmLocalAdd(T *p, T v) {
  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                     ::sycl::memory_scope::work_group,
                     ::sycl::access::address_space::local_space>(*p)
      .fetch_add(v);
}
// GLOBAL, SAID OUT LOUD. The assign body runs inside a call the kernel does
// not inline (the coroutine functions are stack calls on SYCL), so IGC cannot
// prove its pointers are global and emits a GENERIC load for every access:
// mask the high address bits, compare against the local window, branch
// divergently to load.ugm or load.slm, join. Measured in the distance loop,
// that was one check-and-branch per coordinate read, ~10x the baselines'
// instruction count at 71% XVE-active. A pointer cast to the global address
// space loads with a single load.ugm, as the inlined baseline does.
template <typename T>
CTP_GPU_FUN inline auto KmGlobal(T *p) {
  return ::sycl::address_space_cast<::sycl::access::address_space::global_space,
                                    ::sycl::access::decorated::yes>(p);
}
#else
CTP_GPU_FUN inline float *KmTileSums() { return nullptr; }
CTP_GPU_FUN inline unsigned *KmTileCounts() { return nullptr; }
template <typename T>
CTP_GPU_FUN inline void KmLocalAdd(T *p, T v) { atomicAdd(p, v); }
template <typename T>
CTP_GPU_FUN inline T *KmGlobal(T *p) { return p; }
#endif
// Applied AT THE USE, never stored: clio-coroc hoists a coroutine's locals
// into a frame it byte-copies, and a multi_ptr is neither hoistable as `auto`
// nor trivially copyable. The cast itself is free.

/**
 * One Lloyd assignment pass over this block's slice.
 *
 * Centroid sums are accumulated with global atomics rather than a
 * shared-memory reduction: k*dims is small, the loop is dominated by paging,
 * and a shared tile would have to be sized at compile time for the largest k
 * this accepts. (It is also why there is no __shared__ anywhere in this
 * benchmark, and so why one kernel body serves both backends.)
 */
CTP_GPU_FUN CLIO_COROC_INLINE void AssignCoro(gv::DeviceVector<float> v, u64 per,
                                u64 page_elems, u32 dims, u32 k,
                                const float *cent, float *sums,
                                unsigned *counts, float *bsums,
                                unsigned *bcounts, u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  // BLOCK-PRIVATE ACCUMULATORS. With every point's atomics landing in the
  // one global sums/counts, the winning cluster's entries serialize the
  // whole device as Lloyd's converges (the 32 GB deck's steps grew from
  // 20 s to 116 s). bsums/bcounts are per-block slices in global memory
  // (no __shared__, so the one body still serves both backends): a block
  // contends only with its own threads, and flushes each entry to the
  // global accumulators once at the end. Null slices keep the old path.
  float *ts = bsums ? bsums + static_cast<u64>(block) * k * dims : sums;
  unsigned *tc = bcounts ? bcounts + static_cast<u64>(block) * k : counts;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    CO_AWAIT(v.CoFetch(0, base + off, n));
    // Read-only pass: no write intent, so every page stays clean and the
    // oversubscribed streaming read sheds pages without writeback.
    auto h = CO_AWAIT(v.CoHoldPage(base + off, n, /*write=*/false));
    // Whole pages hold whole points (enforced on the host), so a page is
    // exactly n/dims points and no point straddles a page boundary.
    const u64 npts = n / dims;
    // Local tile for this page (SYCL; null elsewhere or if k*dims is too big).
    const u32 kdt = k * dims;
    float *ls = (bsums && kdt <= kKmTileFloats && k <= kKmTileCounts)
                    ? KmTileSums() : nullptr;
    unsigned *lc = ls ? KmTileCounts() : nullptr;
    if (ls) {
      for (u32 i = threadIdx.x; i < kdt; i += blockDim.x) ls[i] = 0.0f;
      for (u32 i = threadIdx.x; i < k; i += blockDim.x) lc[i] = 0u;
      __syncthreads();
    }
    // A RAW POINTER, RESOLVED ONCE PER PAGE. The loop used to read every
    // coordinate through a shim holding a REFERENCE to `h`, and `h` is a
    // coroutine local: taking its address pins it in the frame, which lives
    // in global memory, so each of the k*dims reads per point reloaded the
    // handle's data pointer and base from the frame before touching the
    // point -- where the baselines read a plain `pts + p * dims`. Resident,
    // one launch and zero faults, the kernel still ran ~2x the baselines'
    // time, which is what pointed here. Nothing below can suspend, so a
    // pointer into the held frame is valid until UnpinRange.
    const float *const pg = h.ptr() + (base + off - h.begin_off());
    for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
      const float *const pt = pg + p * dims;
      const u32 bestk =
          ::clio_km::NearestCentroid(KmGlobal(pt), KmGlobal(cent), dims, k);
      if (ls) {
        for (u32 i = 0; i < dims; ++i) {
          KmLocalAdd(&ls[bestk * dims + i], KmGlobal(pt)[i]);
        }
        KmLocalAdd(&lc[bestk], 1u);
      } else {
        for (u32 i = 0; i < dims; ++i) {
          atomicAdd(&ts[bestk * dims + i], KmGlobal(pt)[i]);
        }
        atomicAdd(&tc[bestk], 1u);
      }
    }
    __syncthreads();
    if (ls) {
      // Flush the page's tile into the block's slice: k*dims + k atomics per
      // page instead of (dims + 1) per point. Done before UnpinRange and the
      // next CoFetch, so the tile never outlives this stretch of the kernel.
      for (u32 i = threadIdx.x; i < kdt; i += blockDim.x) {
        if (ls[i] != 0.0f) atomicAdd(&ts[i], ls[i]);
      }
      for (u32 i = threadIdx.x; i < k; i += blockDim.x) {
        if (lc[i] != 0u) atomicAdd(&tc[i], lc[i]);
      }
      __syncthreads();
    }
    // NO RELEASE HINT HERE. Telling the cache "this page is dead after use"
    // was measured to be actively WRONG: the page IS re-read on the next
    // Lloyd pass, and the frequency policy was retaining ~10% of them across
    // passes. Releasing guaranteed a 0% hit rate and cost ~6% in rescore
    // traffic on top. The unpin below is NOT that hint -- it gives back the
    // fetch's reservation and leaves the page resident.
    v.UnpinRange(base + off, n);
  }
  if (bsums) {
    __syncthreads();
    const u32 kd = k * dims;
    for (u32 i = threadIdx.x; i < kd; i += blockDim.x) {
      if (ts[i] != 0.0f) atomicAdd(&sums[i], ts[i]);
    }
    for (u32 i = threadIdx.x; i < k; i += blockDim.x) {
      if (tc[i] != 0u) atomicAdd(&counts[i], tc[i]);
    }
  }
}

}  // namespace clio::gv_bench::kmeans

/* TWO BACKENDS, ONE WORKLOAD. Everything above this line is compiled for both: the transpiled state machine contains no vendor token. What differs is only how a grid is submitted. */
#if CTP_ENABLE_SYCL

namespace clio::gv_bench::kmeans {

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
       // THE SYCL ANALOGUE OF LAUNCH BOUNDS. IGC picks the SIMD width itself
       // and chose SIMD16 for the coroutine kernels, where the baselines
       // compile SIMD32 -- half the lanes per instruction on the same work.
           [[intel::reqd_sub_group_size(32)]]
#endif
       { body(); })
      .wait();
}

/** The yieldable prologue, shared by both yieldable launches below. */
template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 v, View vw, StackView sv,
                     MakeCoro make) {
  Submit(grid, block, [=]() {
    DevF32 dev = v;
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

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, u64 base_idx, View vw,
                StackView sv, u32 publish) {
  // gpu_info is already stamped into every block's record by InitBackend, so
  // unlike CUDA there is no per-launch CLIO_GPU_INIT store.
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    SeedCoro(_cy, dev, per, page_elems, dims, k, base_idx, blk, publish);
  });
}

void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    AssignCoro(_cy, dev, per, page_elems, dims, k, cent, sums, counts, nullptr,
               nullptr, blk);
  });
}

void LaunchAssignTiled(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v,
                       u64 per, u64 page_elems, u32 dims, u32 k,
                       const float *cent, float *sums, unsigned *counts,
                       float *bsums, unsigned *bcounts, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](clio::co::Ctx &_cy, DevF32 dev, u32 blk) {
    AssignCoro(_cy, dev, per, page_elems, dims, k, cent, sums, counts, bsums,
               bcounts, blk);
  });
}

void LaunchBaseline(u32 threads, const float *tile, u64 n, u32 dims, u32 k,
                    const float *cent, float *sums, unsigned *counts) {
  Submit(dim3(1), dim3(threads),
         [=]() { BaselineBody(tile, n, dims, k, cent, sums, counts); });
}

void LaunchUpdate(float *cent, const float *sums, const unsigned *counts,
                  u32 dims, u32 k) {
  Submit(dim3((k + 63) / 64), dim3(64),
         [=]() { UpdateBody(cent, sums, counts, dims, k); });
}

}  // namespace clio::gv_bench::kmeans

namespace clio::run::gpu {

/** The out-of-line half of YieldStack::Reset; see (3) in the file comment
 *  and the declaration in yield_stack.h. */
void SyclYieldStackReset(const YieldStackView &view, clio::run::u32 nlanes,
                         char *smem_base) {
  auto &q = ctp::GpuApi::SyclQueue();
  YieldStackView v = view;
  q.parallel_for(sycl::range<1>(nlanes), [=](sycl::id<1> i) {
     auto *h = reinterpret_cast<YieldLaneHeader *>(
         v.base_ + static_cast<clio::run::u64>(i[0]) * v.bytes_per_lane_);
     // sp_ starts AFTER the header: the header is not frame space.
     h->sp_ = sizeof(YieldLaneHeader);
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

namespace clio::gv_bench::kmeans {

namespace {

/** The prologue every yieldable kernel repeats: bind this block to its page
 *  table, publish the lane stack, then run the chain. Identical to the SYCL
 *  side's, which is the point. */
__global__ GV_LAUNCH_BOUNDS void SeedKernel(GpuInfo info, DevF32 v, u64 per,
                                            u64 page_elems, u32 dims, u32 k,
                                            u64 base_idx, View yv,
                                            StackView ys, u32 publish) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, SeedCoro(_cy, v, per, page_elems, dims, k, base_idx, yv.Block(), publish));
}

__global__ GV_LAUNCH_BOUNDS void AssignKernel(GpuInfo info, DevF32 v, u64 per, u64 page_elems,
                             u32 dims, u32 k, const float *cent, float *sums,
                             unsigned *counts, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, AssignCoro(_cy, v, per, page_elems, dims, k, cent, sums, counts,
                            nullptr, nullptr, yv.Block()));
}

__global__ GV_LAUNCH_BOUNDS void AssignTiledKernel(
    GpuInfo info, DevF32 v, u64 per, u64 page_elems, u32 dims, u32 k,
    const float *cent, float *sums, unsigned *counts, float *bsums,
    unsigned *bcounts, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, AssignCoro(_cy, v, per, page_elems, dims, k, cent, sums, counts,
                            bsums, bcounts, yv.Block()));
}

__global__ GV_LAUNCH_BOUNDS void BaselineKernel(const float *tile, u64 n, u32 dims, u32 k,
                               const float *cent, float *sums,
                               unsigned *counts) {
  BaselineBody(tile, n, dims, k, cent, sums, counts);
}

__global__ GV_LAUNCH_BOUNDS void UpdateKernel(float *cent, const float *sums,
                             const unsigned *counts, u32 dims, u32 k) {
  UpdateBody(cent, sums, counts, dims, k);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // Nothing to do: CUDA's per-block IpcManager is __shared__ storage, born
  // fresh at every launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, u64 base_idx, View vw,
                StackView sv, u32 publish) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, dims, k, base_idx, vw, sv, publish);
}

void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv) {
  AssignKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, dims, k, cent, sums, counts, vw, sv);
}

void LaunchAssignTiled(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v,
                       u64 per, u64 page_elems, u32 dims, u32 k,
                       const float *cent, float *sums, unsigned *counts,
                       float *bsums, unsigned *bcounts, View vw, StackView sv) {
  AssignTiledKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, dims, k, cent, sums, counts, bsums, bcounts,
      vw, sv);
}

void LaunchBaseline(u32 threads, const float *tile, u64 n, u32 dims, u32 k,
                    const float *cent, float *sums, unsigned *counts) {
  BaselineKernel<<<1, threads>>>(tile, n, dims, k, cent, sums, counts);
}

void LaunchUpdate(float *cent, const float *sums, const unsigned *counts,
                  u32 dims, u32 k) {
  UpdateKernel<<<(k + 63) / 64, 64>>>(cent, sums, counts, dims, k);
}

}  // namespace clio::gv_bench::kmeans

#endif  /* CTP_ENABLE_SYCL */

#if !CTP_IS_DEVICE_PASS

// Cross-node reduction. Included INSIDE the device-pass guard: it uses the
// CTE client, whose members are compiled out of the device pass, so at file
// scope it breaks the CUDA build of this driver and not the SYCL one.
#include "../bench_dist.h"
#include "../gv_comm_report.h"
#include "../bench_ckpt.h"


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
  /** Where the driver's time went since the last ResetTimers: GPU time in
   *  launch+sync, the per-round D2H state copy, the pending-list upload. */
  double KernelMs() const { return drv_.KernelMs(); }
  double CopyMs() const { return drv_.CopyMs(); }
  double UploadMs() const { return drv_.UploadMs(); }
  void ResetTimers() { drv_.ResetTimers(); }
  const std::vector<std::pair<double, u32>> &RoundLog() const {
    return drv_.RoundLog();
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
  // --ckpt-final: vector.Copy of the point set after the run (see
  // bench_ckpt.h); --ckpt-sync makes that Copy fully synchronous.
  bool ckpt = false, ckpt_sync = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;
  // DATA-PARALLEL DECOMPOSITION. --nodes N --node i gives this process the
  // i'th contiguous shard of the SAME global point set --data-mb describes,
  // matching the MPI baseline's split exactly (last node absorbs the
  // remainder). One node, the default, is the whole set, so a single-process
  // run is unchanged -- and is the reference a distributed run must reproduce.
  //
  // Only the centroid reduction crosses nodes. Each node's points are private,
  // so its vector gets its own tag namespace; two processes sharing one CTE
  // would otherwise both create "gv_kmeans" and page into each other's blobs.
  u32 nodes = 1, node = 0;
  // --publish-seed writes the seeded points back to the CTE (the old
  // behaviour); by default a resident run keeps them only in its frames.
  bool publish_seed = false;

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
    else if (a == "--publish-seed") publish_seed = true;
    else if (a == "--iters") iters = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--hbm-mb") hbm_mb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--ckpt-final") ckpt = true;
    else if (a == "--ckpt-sync") ckpt = ckpt_sync = true;
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--baseline") baseline = true;
    else if (a == "--nodes") nodes = static_cast<u32>(next());
    else if (a == "--node") node = static_cast<u32>(next());
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--dims N] "
                  "[--clusters N] [--slots N] [--iters N] [--page-kb N] "
                  "[--data-mb N] [--hbm-mb N] [--repeat N] [--hbm-only] "
                  "[--nodes N --node I]\n",
                  argv[0]);
      return 0;
    }
  }

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

  if (node >= nodes) {
    std::fprintf(stderr, "KMEANS ERROR: --node %u is out of range for "
                 "--nodes %u\n", node, nodes);
    return 2;
  }
  // --data-mb is the GLOBAL problem, split across the nodes; this is strong
  // scaling, like the MPI baseline. Required to divide evenly: a rounded shard
  // makes the union of the shards something other than the single-node point
  // set, and the gate would then be comparing two different problems.
  const u64 global_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  if (nodes > 1 && (data_mb % nodes) != 0) {
    std::fprintf(stderr, "KMEANS ERROR: --data-mb %llu does not divide evenly "
                 "across --nodes %u; choose a multiple.\n",
                 (unsigned long long)data_mb, nodes);
    return 2;
  }
  const u64 total_elems = global_elems / nodes;
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
  // `per` is truncated to a whole number of pages, so n can be less than
  // total_elems -- but it is the SAME n on every node, which makes the shards
  // contiguous and non-overlapping at exactly node*n. Deriving the base from
  // the requested size instead of the realised one would leave a gap at every
  // node boundary and quietly change the point set.
  const u64 base_idx = static_cast<u64>(node) * n;
  const double logical_mb =
      static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);

  // THE BENCH OWNS ITS CONFIG ONLY WHEN NOBODY ELSE SUPPLIED ONE. Writing
  // one and Setenv-ing it with overwrite=1 unconditionally makes it
  // impossible to point this bench at a cluster: any CLIO_SERVER_CONF the
  // caller exported is clobbered a line later, so every node stands up its
  // own single-host runtime and the reduction can never meet. A distributed
  // harness needs exactly that config -- one naming a hostfile and the other
  // nodes -- so an already-set CLIO_SERVER_CONF is left alone.
  if (getenv("CLIO_SERVER_CONF") != nullptr) {
    std::printf("  runtime: using CLIO_SERVER_CONF=%s (not writing one)\n",
                getenv("CLIO_SERVER_CONF"));
  } else {
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
            << "        persistence_level: \"temporary\"\n"
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
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  kb::InitBackend(blocks, gpu);

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

  // ASSOCIATIVITY, NOT FULL ASSOCIATIVITY. A lookup scans its set, so one
  // giant set costs O(frames) per probe -- kmeans went 4.7 s -> 42 s as a
  // 512-way cache. Keep a set per block and floor the width at 8, which is
  // what covers several blocks colliding on one set.
  // Per-node tag namespace: the points are private to this node, and two
  // processes on one CTE would otherwise share the blob names.
  const std::string region =
      (nodes > 1) ? ("gv_kmeans_n" + std::to_string(node)) : "gv_kmeans";
  gv::Vector<float> vec(region.c_str(), {0}, page_bytes, blocks,
                        slots < 16u ? 16u : slots, n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(blocks, threads);

  // ---- seed the point set -------------------------------------------------
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    kb::LaunchSeed(g, b, gpu, dev, per, page_elems, dims, k, base_idx, vw,
                   sv, publish_seed ? 1u : 0u);
  });
  ctp::GpuApi::Synchronize();

  // ---- device state -------------------------------------------------------
  float *d_cent = nullptr, *d_sums = nullptr;
  unsigned *d_counts = nullptr;
  d_cent = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_cent)>>(k * dims * sizeof(float));
  d_sums = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sums)>>(k * dims * sizeof(float));
  d_counts = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_counts)>>(k * sizeof(unsigned));
  // Per-block slices for the assignment (see AssignCoro).
  const size_t bsums_n = static_cast<size_t>(blocks) * k * dims;
  const size_t bcounts_n = static_cast<size_t>(blocks) * k;
  float *d_bsums = ctp::GpuApi::Malloc<float>(bsums_n * sizeof(float));
  unsigned *d_bcounts =
      ctp::GpuApi::Malloc<unsigned>(bcounts_n * sizeof(unsigned));

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
      kb::LaunchBaseline(threads, bl_dev, page_elems, dims, k, d_cent, d_sums,
                         d_counts);
      ctp::GpuApi::Synchronize();
    }
    return true;
  };

  // ---- cross-node centroid reduction ------------------------------------
  // Built AFTER the runtime is up (a client constructed before CLIO_INIT
  // dereferences a runtime that does not exist) and ONLY when there is a
  // reduction to do -- an unconditional second CTE client crashes the
  // single-node run.
  //
  // The reduction tag is deliberately NOT per-node: the point shards are
  // private, but the partial sums have to meet somewhere.
  std::unique_ptr<clio::cte::core::Client> cte_red;
  clio::cte::core::TagId red_tag{};
  if (nodes > 1) {
    cte_red = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_red->AsyncGetOrCreateTag("gv_kmeans_red");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "KMEANS ERROR: could not create the reduction "
                   "tag\n");
      return 1;
    }
    red_tag = t->tag_id_;
  }
  // Lloyd's update needs the sums and counts over the WHOLE point set, not
  // this node's shard, or every node converges to its own shard's centroids
  // and the run silently solves a different problem. Reduced as doubles: the
  // partials are float on the device, but summing N nodes' partials in float
  // would add a node-count-dependent rounding term to a value the gate
  // compares against a single-node reference.
  std::vector<double> red_buf(static_cast<size_t>(k) * dims + k);
  std::vector<float> h_sums(static_cast<size_t>(k) * dims);
  std::vector<unsigned> h_counts(k);
  u64 red_round = 0;
  const auto reduce_centroids = [&]() -> bool {
    if (nodes <= 1) return true;
    ctp::GpuApi::Memcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(float));
    ctp::GpuApi::Memcpy(h_counts.data(), d_counts,
                        h_counts.size() * sizeof(unsigned));
    for (size_t i = 0; i < h_sums.size(); ++i) red_buf[i] = h_sums[i];
    for (u32 c = 0; c < k; ++c) red_buf[h_sums.size() + c] = h_counts[c];
    if (!clio_bench_dist::ReduceSum(*cte_red, red_tag, node, nodes,
                                    red_round++, red_buf.data(),
                                    static_cast<int>(red_buf.size()),
                                    "kmred")) {
      return false;
    }
    for (size_t i = 0; i < h_sums.size(); ++i) {
      h_sums[i] = static_cast<float>(red_buf[i]);
    }
    for (u32 c = 0; c < k; ++c) {
      h_counts[c] = static_cast<unsigned>(red_buf[h_sums.size() + c]);
    }
    ctp::GpuApi::Memcpy(d_sums, h_sums.data(), h_sums.size() * sizeof(float));
    ctp::GpuApi::Memcpy(d_counts, h_counts.data(),
                        h_counts.size() * sizeof(unsigned));
    return true;
  };

  double best_ms = 1e30;
  std::vector<float> h_final(static_cast<size_t>(k) * dims);
  for (int r = 0; r < repeat; ++r) {
    ctp::GpuApi::Memcpy(d_cent, h_cent.data(), h_cent.size() * sizeof(float));
    vec.ResetStats();
    ctp::GpuApi::Synchronize();
    const auto km_c0 = vec.ReadStats(0);
    double km_host_ms = 0.0;
    const double t0 = NowMs();
    double t_iter0 = t0;
    for (u32 it = 0; it < iters; ++it) {
      u32 it_rounds = 0;
      ctp::GpuApi::Memset(d_sums, 0, k * dims * sizeof(float));
      ctp::GpuApi::Memset(d_counts, 0, k * sizeof(unsigned));
      if (baseline) {
        if (!run_baseline_pass()) {
          std::fprintf(stderr, "KMEANS ERROR: baseline tile loop failed\n");
          return 1;
        }
      } else {
        ctp::GpuApi::Memset(d_bsums, 0, bsums_n * sizeof(float));
        ctp::GpuApi::Memset(d_bcounts, 0, bcounts_n * sizeof(unsigned));
        runner.ResetTimers();
        it_rounds = runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          kb::LaunchAssignTiled(g, b, gpu, dev, per, page_elems, dims, k,
                                d_cent, d_sums, d_counts, d_bsums, d_bcounts,
                                vw, sv);
        });
      }
      // Combine the shards BEFORE the update, so every node divides the same
      // global sums by the same global counts and they stay in lockstep.
      ctp::GpuApi::Synchronize();
      const double t_kernel = NowMs();
      if (!reduce_centroids()) {
        std::fprintf(stderr, "KMEANS ERROR: centroid reduction failed\n");
        return 1;
      }
      const double t_reduce = NowMs();
      km_host_ms += t_reduce - t_kernel;
      kb::LaunchUpdate(d_cent, d_sums, d_counts, dims, k);
      // PER-ITERATION SPLIT, on stderr with the rest of the diagnostics: the
      // two-node runs were slow with almost no faults, and the total alone
      // could not say whether the time was in the kernel or in the
      // cross-node reduction that follows it.
      {
        const auto is = vec.ReadStats(0);
        const auto &rl = runner.RoundLog();
        std::fprintf(stderr,
                     "  iter %u: kernel=%.1fms reduce=%.1fms faults=%llu "
                     "evicts=%llu alloc_waits=%llu rounds=%u "
                     "drv_gpu=%.1fms drv_copy=%.1fms drv_upload=%.1fms "
                     "r0=%.1fms/%u r1=%.1fms/%u rlast=%.1fms/%u\n",
                     it, t_kernel - t_iter0, t_reduce - t_kernel,
                     (unsigned long long)is.faults,
                     (unsigned long long)is.evicts,
                     (unsigned long long)is.alloc_waits, it_rounds,
                     runner.KernelMs(), runner.CopyMs(), runner.UploadMs(),
                     rl.size() > 0 ? rl[0].first : 0.0,
                     rl.size() > 0 ? rl[0].second : 0u,
                     rl.size() > 1 ? rl[1].first : 0.0,
                     rl.size() > 1 ? rl[1].second : 0u,
                     rl.empty() ? 0.0 : rl.back().first,
                     rl.empty() ? 0u : rl.back().second);
      }
      t_iter0 = NowMs();
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    {
      clio_gv_bench::CommAcc km_comm;
      km_comm.Add(km_c0, vec.ReadStats(0));
      km_comm.AddHost(km_host_ms);
      km_comm.Print("kmeans", ms);
    }
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
               "alloc_waits=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
               baseline ? "baseline" : "paged",
               blocks, threads, dims, k, iters, (unsigned long long)page_kb,
               slots, logical_mb, (unsigned long long)hbm_mb,
               (unsigned long long)npoints, best_ms, gbps, csum,
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.alloc_waits,
               (unsigned long long)st.puts, (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps);
  if (!baseline && !publish_seed && st.evicts != 0) {
    std::fprintf(stderr, "KMEANS ERROR: %llu evictions of unpublished seed "
                 "pages (the cache is not resident); rerun with "
                 "--publish-seed or more --slots\n",
                 (unsigned long long)st.evicts);
    return 1;
  }

  // FINAL-STATE CHECKPOINT, after every gate that reads the live cache (the
  // multi-node path drops it first).
  std::unique_ptr<gv::Vector<float>> vec_ck;
  if (ckpt && !baseline) {
    vec_ck = clio_bench_ckpt::FinalCheckpoint(vec, region + "_ckpt", nodes,
                                              ckpt_sync);
  }
  // EXIT BARRIER. Each rank embeds its node's runtime, so a rank that exits
  // takes that node's containers with it. A slower rank whose checkpoint tag
  // is owned by an exited node then waits forever in GetOrCreateTag (64
  // nodes: 5 ranks hung). No node leaves until every node has checkpointed.
  if (ckpt && !baseline && nodes > 1 &&
      !clio_bench_dist::Barrier(*cte_red, red_tag, node, nodes, red_round++,
                                "kmdone", 600)) {
    std::fprintf(stderr, "KMEANS ERROR: exit barrier failed\n");
  }

  ctp::GpuApi::Free(d_cent); ctp::GpuApi::Free(d_sums); ctp::GpuApi::Free(d_counts);
  ctp::GpuApi::Free(d_bsums); ctp::GpuApi::Free(d_bcounts);
  BenchFlushData();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
