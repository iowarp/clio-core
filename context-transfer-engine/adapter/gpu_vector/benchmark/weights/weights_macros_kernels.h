/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weights paged kernels, MACRO (Duff's device) form.
 *
 * Same workload as weights_kernels.h with suspension expressed through
 * yield_stack.h's macros instead of C++20 device coroutines, which clang
 * cannot compile for SPIR-V. See kmeans_macros_kernels.h for the full
 * rationale; the constraint is upstream in LLVM, not a flag.
 *
 * THE ONE RULE. Anything live across a yield is a frame local (CLIO_YLOCAL*).
 * `acc` is the one that matters here -- it accumulates across every page and
 * therefore across every suspension, so as an automatic it would read back
 * garbage after the first resume. `r` stays an ordinary register: it is born
 * and consumed between two yields.
 */
#ifndef CLIO_GV_BENCH_WEIGHTS_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_WEIGHTS_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "weights_kernels.h"

namespace clio::gv_bench::weights_macros {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;
// The plain (non-coroutine) helpers and kernel bodies beside the coroutines
// -- Weight, Activation, BaselineBody -- are ordinary device functions and are
// reused as-is. Only the coroutine spellings are unavailable on SPIR-V, and
// those are now behind CLIO_HAS_YCORO.
using namespace ::clio::gv_bench::weights;

using DevU32 = gv::DeviceVector<u32>;

/** Macro-form SeedLaneCoro. */
CTP_GPU_FUN inline void SeedLaneMacro(DevU32 v, u64 per, u64 page_elems,
                                      u32 flat_pct, u64 base_idx, u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, base, static_cast<u64>(block) * per);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, n, 0);
  CLIO_YLOCAL(gv::Held<u32>, h);
  CLIO_YBEGIN();
  base = static_cast<u64>(block) * per;
  for (; off < per; off += page_elems) {
    n = (off + page_elems <= per) ? page_elems : (per - off);
    CLIO_YCALL(v.MFetch(0, base + off, n));
    CLIO_YCALL(v.MHoldPage(&h, base + off, n, /*write=*/true));
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = Weight(base_idx + base + off + i, flat_pct);
    }
    __syncthreads();
    // FLUSH AS WE GO: a dirty frame is not evictable, so deferring every
    // writeback dirties the whole table and the next fetch has nowhere to
    // land. Ranged, so this sends only what was just written.
    CLIO_YCALL(v.MBeginFlush(0, base + off, n));
    // Fetch is the pinner; UnpinRange is the releaser.
    v.UnpinRange(base + off, n);
  }
  CLIO_YCALL(v.MEndFlush());
  CLIO_YEND();
}

/** Macro-form WeightsLaneCoro: the measured weighted sum. */
CTP_GPU_FUN inline void WeightsLaneMacro(DevU32 v, u64 per, u64 page_elems,
                                         unsigned long long *sum,
                                         unsigned long long *page_sum,
                                         unsigned *page_visits, u64 base_idx,
                                         u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, base, static_cast<u64>(block) * per);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, n, 0);
  // Accumulates across every page, so it spans every suspension.
  CLIO_YLOCAL_INIT(unsigned long long, acc, 0ull);
  CLIO_YLOCAL(gv::Held<u32>, h);
  CLIO_YBEGIN();
  base = static_cast<u64>(block) * per;
  for (; off < per; off += page_elems) {
    n = (off + page_elems <= per) ? page_elems : (per - off);
    CLIO_YCALL(v.MFetch(0, base + off, n));
    CLIO_YCALL(v.MHoldPage(&h, base + off, n));
    unsigned long long r = 0;                     // register, not the frame
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
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
  CLIO_YEND();
}

}  // namespace clio::gv_bench::weights_macros

#endif  // CLIO_GV_BENCH_WEIGHTS_MACROS_KERNELS_H_
