/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The weights device code -- ONE copy, compiled by CUDA and by SYCL.
 *
 * Same split as the other workloads: the workload is here, the launches are
 * in cuda/ and sycl/. Nothing here is backend-specific -- no warp
 * intrinsics, no __shared__, no cooperative groups.
 *
 * CTP_GPU_FUN, not plain inline: under CUDA these must be __device__ or the
 * host pass compiles them as host functions and every co_await on a
 * device-only verb fails to resolve.
 */
#ifndef CLIO_GV_BENCH_WEIGHTS_KERNELS_H_
#define CLIO_GV_BENCH_WEIGHTS_KERNELS_H_

#include "weights_math.h"

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::weights {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;

using ::clio_wt::Activation;
using ::clio_wt::kFlatGranuleElems;
using ::clio_wt::PageIsFlat;
using ::clio_wt::Weight;


/**
 * BASELINE KERNEL -- the textbook out-of-core GPU model, for comparison.
 *
 * The GPU vector's whole claim is that a kernel can fault on data it does not
 * have and keep running. The honest control for that claim is how everyone
 * does it WITHOUT such a mechanism:
 *
 *   1. storage I/O is SYNCHRONOUS   -- the host blocks on a CTE GetBlob
 *   2. HBM<->DRAM copy is SYNCHRONOUS -- a blocking cudaMemcpy per tile
 *   3. THE KERNEL TERMINATES for I/O -- one launch per tile, and the grid is
 *      torn down and rebuilt between every transfer
 *
 * Those three are one design, not three: once a kernel must exit to get data,
 * the transfer is necessarily host-driven and serialised behind it. Nothing
 * overlaps -- read, copy, compute and teardown all run end to end.
 *
 * It computes EXACTLY the contributions the paged kernel does (same integer
 * arithmetic, same per-page accounting), so the existing got-vs-want checksum
 * validates the baseline rather than a second implementation being trusted on
 * inspection. Integer accumulation means order does not change the total.
 */
CTP_GPU_FUN inline void BaselineBody(const ::clio::run::u32 *tile,
                                      ::clio::run::u64 gbase, ::clio::run::u64 n,
                                      unsigned long long *sum,
                                      unsigned long long *page_sum,
                                      unsigned *page_visits,
                                      ::clio::run::u64 page_elems) {
  unsigned long long r = 0;
  for (::clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
    r += static_cast<unsigned long long>(tile[i]) * Activation(gbase + i);
  }
  atomicAdd(sum, r);
  atomicAdd(&page_sum[gbase / page_elems], r);
  if (threadIdx.x == 0) {
    atomicAdd(&page_visits[gbase / page_elems], 1u);
  }
}

/**
 * Seeding is the phase that actually wedges, so it yields too.
 *
 * Every page it writes is dirty, so making room for the next one flushes the
 * previous one -- and on the kHbm tier that writeback is a device copy that
 * cannot schedule while this kernel sits on the SM waiting for it. Suspending
 * lets the copy run.
 */
/**
 * The seed pass as a per-lane C++20 coroutine (clang-CUDA builds). Ordinary
 * locals -- `off` and the guard cross suspends and the COMPILER puts them in
 * the frame; compare the CLIO_YLOCAL_INIT bookkeeping in the macro version
 * below. Suspends stay block-collective: every yield site votes.
 */
CTP_GPU_FUN inline gy::YCoroMain SeedLaneCoro(gv::DeviceVector<::clio::run::u32> v,
                                      ::clio::run::u64 per,
                                      ::clio::run::u64 page_elems,
                                      ::clio::run::u32 flat_pct,
                                      ::clio::run::u64 base_idx,
                                      ::clio::run::u32 block) {
  const ::clio::run::u64 base = static_cast<::clio::run::u64>(block) * per;
  for (::clio::run::u64 off = 0; off < per; off += page_elems) {
    co_await v.Fetch(0, base + off,
                     (off + page_elems <= per) ? page_elems : (per - off));
    const ::clio::run::u64 n =
        (off + page_elems <= per) ? page_elems : (per - off);
    {
      auto h = co_await v.HoldPage(base + off, n, /*write=*/true);
    for (::clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = Weight(base_idx + base + off + i, flat_pct);
    }
    __syncthreads();
    }
    // FLUSH AS WE GO. A dirty frame is not evictable, so deferring every
    // writeback to the end dirties the whole table and the next fetch has
    // nowhere to land. Ranged, so this sends only what was just written.
    co_await v.BeginFlush(0, base + off, n);
    // Fetch is the pinner; UnpinRange is the releaser.
    v.UnpinRange(base + off, n);
  }
  co_await v.EndFlush();
}

/**
 * The measured pass: a weighted sum over this block's slice of the model,
 * SUSPENDING on a miss. Every lane votes on whether its page is resident,
 * the whole block suspends if ANY lane is waiting, and the kernel
 * exits so the fetch can land. On resume the page is resident for everyone, so
 * the read is an ordinary parallel loop with no fault path in it at all.
 */
/** The measured pass as a per-lane coroutine; same shape as SeedLaneCoro. */
CTP_GPU_FUN inline gy::YCoroMain WeightsLaneCoro(gv::DeviceVector<::clio::run::u32> v,
                                         ::clio::run::u64 per,
                                         ::clio::run::u64 page_elems,
                                         unsigned long long *sum,
                                         unsigned long long *page_sum,
                                         unsigned *page_visits,
                                         ::clio::run::u64 base_idx,
                                         ::clio::run::u32 block) {
  const ::clio::run::u64 base = static_cast<::clio::run::u64>(block) * per;
  unsigned long long acc = 0;
  for (::clio::run::u64 off = 0; off < per; off += page_elems) {
    co_await v.Fetch(0, base + off,
                     (off + page_elems <= per) ? page_elems : (per - off));
    auto h = co_await v.HoldPage(
        base + off, (off + page_elems <= per) ? page_elems : (per - off));
    const ::clio::run::u64 n =
        (off + page_elems <= per) ? page_elems : (per - off);
    unsigned long long r = 0;                     // register, not the frame
    for (::clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
      r += static_cast<unsigned long long>(h[base + off + i]) *
           Activation(base_idx + base + off + i);
    }
    acc += r;
    atomicAdd(&page_sum[(base + off) / page_elems], r);
    if (threadIdx.x == 0) {
      atomicAdd(&page_visits[(base + off) / page_elems], 1u);
    }
    v.UnpinRange(base + off, n);
  }
  atomicAdd(sum, acc);
}

}  // namespace clio::gv_bench::weights

#endif  // CLIO_GV_BENCH_WEIGHTS_KERNELS_H_
