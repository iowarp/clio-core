/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The kmeans device code -- ONE copy, compiled by CUDA and by SYCL.
 *
 * WHAT IS HERE AND WHAT IS NOT
 *
 * Everything in this header is the WORKLOAD: the coordinate generator, the
 * Lloyd assignment pass, the centroid update, the staged-tile baseline. None
 * of it is backend-specific, and none of it is duplicated -- `cuda/` and
 * `sycl/` hold only the launch wrappers, which are about ten lines each.
 *
 * That split is possible because of what the kernels do NOT use. Measured
 * across this benchmark: zero warp intrinsics, zero __shared__, zero
 * cooperative groups. The only CUDA-isms were `__global__` and `<<<>>>`,
 * and those live entirely in the launch, not in the body. The device-side
 * spellings that remain (threadIdx, atomicAdd, __syncthreads) are supplied
 * for SYCL by clio_ctp/util/sycl_cuda_compat.h.
 *
 * The coroutines are the reason this works at all: they are already
 * backend-neutral, because DeviceVector and the yield machinery are.
 */
#ifndef CLIO_GV_BENCH_KMEANS_KERNELS_H_
#define CLIO_GV_BENCH_KMEANS_KERNELS_H_

// The science, shared with the MPI/NVSHMEM/BaM baselines, which cannot
// include any clio header. See kmeans_math.h.
#include "kmeans_math.h"

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::kmeans {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

using ::clio_km::PointVal;

/* ------------------------------------------------------------------ */
/* Paged path: the coroutines.                                          */
/*                                                                      */
/* CTP_GPU_FUN, not plain inline: under CUDA these must be __device__ or */
/* the host pass compiles them as host functions and every co_await on a */
/* device-only verb fails to resolve. Under SYCL it expands to nothing,  */
/* which is correct -- there are no execution-space attributes there.    */
/* ------------------------------------------------------------------ */
/* The bodies are byte-for-byte what the CUDA-only file had -- they     */
/* were already portable, which is the whole point.                     */
/* ------------------------------------------------------------------ */

/** Seed: write the point set into the vector, one page at a time.
 *
 *  `base_idx` is this node's offset into the GLOBAL point set. The vector
 *  index stays local -- each node's vector holds only its own shard -- but the
 *  VALUE written must come from the global index, or the union of the shards
 *  is not the single-node point set and the distributed run is solving a
 *  different problem than the reference it is gated against. Zero for a
 *  single-node run, which is therefore unchanged.
 */
CTP_GPU_FUN inline gy::YCoroMain SeedCoro(gv::DeviceVector<float> v, u64 per,
                              u64 page_elems, u32 dims, u32 k, u64 base_idx,
                              u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    co_await v.Fetch(0, base + off, n);
    auto h = co_await v.HoldPage(base + off, n, /*write=*/true);
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = PointVal(base_idx + base + off + i, dims, k);
    }
    // Collective: name the page just written.
    co_await v.BeginFlush(0, base + off, n);
    // Fetch is the pinner; UnpinRange is the releaser.
    v.UnpinRange(base + off, n);
  }
  // Collect every flush started above: only explicit flushes write data back
  // now (drops refuse dirty pages), so a seeded page left in flight or left
  // to eviction would simply be lost.
  co_await v.EndFlush();
}

/**
 * One Lloyd assignment pass over this block's slice.
 *
 * Centroid sums are accumulated with global atomics rather than a
 * shared-memory reduction: k*dims is small, the loop is dominated by paging,
 * and a shared tile would have to be sized at compile time for the largest k
 * this accepts. (It is also why there is no __shared__ anywhere in this
 * benchmark, and so why one kernel body serves both backends.)
 */
CTP_GPU_FUN inline gy::YCoroMain AssignCoro(gv::DeviceVector<float> v, u64 per,
                                u64 page_elems, u32 dims, u32 k,
                                const float *cent, float *sums,
                                unsigned *counts, u32 block) {
  const u64 base = static_cast<u64>(block) * per;
  for (u64 off = 0; off < per; off += page_elems) {
    const u64 n = (off + page_elems <= per) ? page_elems : (per - off);
    co_await v.Fetch(0, base + off, n);
    // Read-only pass: no write intent, so every page stays clean and the
    // oversubscribed streaming read sheds pages without writeback.
    auto h = co_await v.HoldPage(base + off, n);
    // Whole pages hold whole points (enforced on the host), so a page is
    // exactly n/dims points and no point straddles a page boundary.
    const u64 npts = n / dims;
    for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
      const u64 pbase = base + off + p * dims;
      // The held page is indexed by ABSOLUTE element offset, so hand
      // NearestCentroid a shim that closes over that base rather than a raw
      // pointer -- the arithmetic is then provably the baselines'.
      struct PageAt {
        const decltype(h) &hh;
        u64 b;
        CTP_GPU_FUN float operator[](u32 i) const { return hh[b + i]; }
      } pt{h, pbase};
      const u32 bestk = ::clio_km::NearestCentroid(pt, cent, dims, k);
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
    // traffic on top. The unpin below is NOT that hint -- it gives back the
    // fetch's reservation and leaves the page resident.
    v.UnpinRange(base + off, n);
  }
}

/* ------------------------------------------------------------------ */
/* Plain (non-paged) kernel bodies. These are ordinary functions; the   */
/* per-backend wrapper supplies the launch geometry.                    */
/* ------------------------------------------------------------------ */

/**
 * BASELINE BODY -- the out-of-core model WITHOUT in-kernel faulting.
 *
 * The control for the GPU vector's claim: storage I/O synchronous, HBM<->DRAM
 * copy synchronous, and the kernel torn down for every transfer.
 *
 * Reads its points from a STAGED TILE rather than through the vector, but
 * performs the identical assignment and the identical atomicAdds, so the
 * centroid checksum is directly comparable with the paged path.
 */
CTP_GPU_FUN inline void BaselineBody(const float *tile, u64 n, u32 dims, u32 k,
                                     const float *cent, float *sums,
                                     unsigned *counts) {
  const u64 npts = n / dims;
  for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
    const float *pt = tile + p * dims;
    const u32 bestk = ::clio_km::NearestCentroid(pt, cent, dims, k);
    for (u32 i = 0; i < dims; ++i) {
      atomicAdd(&sums[bestk * dims + i], pt[i]);
    }
    atomicAdd(&counts[bestk], 1u);
  }
}

/** centroid = sum / count, leaving an empty cluster where it was. */
CTP_GPU_FUN inline void UpdateBody(float *cent, const float *sums,
                                   const unsigned *counts, u32 dims, u32 k) {
  const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  ::clio_km::UpdateCentroid(cent, sums, counts, dims, c);
}

}  // namespace clio::gv_bench::kmeans

#endif  // CLIO_GV_BENCH_KMEANS_KERNELS_H_
