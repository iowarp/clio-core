/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * kmeans paged kernels, MACRO (Duff's device) form.
 *
 * WHY A SECOND SPELLING. kmeans_kernels.h expresses suspension with C++20
 * device coroutines, which clang cannot compile for SPIR-V at all -- its
 * EmitCoroutineBody builds the frame PHI as ptr addrspace(0) and the whole
 * llvm.coro.* family is declared llvm_ptr_ty, while spir64's datalayout
 * carries -G1. That is an upstream LLVM constraint, not a flag: measured on
 * Aurora oneAPI 2025.2/2025.3.2 and on intel/llvm nightly-2026-09-18.
 *
 * These bodies are the SAME workload with suspension expressed through
 * yield_stack.h's macros, which need no compiler coroutine support and so
 * reach Intel GPUs. The coroutine file is untouched and stays the CUDA path.
 *
 * THE ONE RULE. The switch re-enters mid-function, so anything live across a
 * yield must be a frame local (CLIO_YLOCAL*), not an automatic. Every local
 * below that spans a CLIO_YCALL is declared that way for exactly that reason.
 */
#ifndef CLIO_GV_BENCH_KMEANS_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_KMEANS_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "kmeans_math.h"
/* The non-coroutine bodies beside the coroutines (BaselineBody, UpdateBody)
 * are plain device functions and are perfectly usable here -- only the
 * coroutine spellings are unavailable on SPIR-V, and those are now behind
 * CLIO_HAS_YCORO. Reuse them rather than keeping a second copy in step. */
#include "kmeans_kernels.h"

namespace clio::gv_bench::kmeans_macros {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;
using ::clio_km::PointVal;
using ::clio::gv_bench::kmeans::BaselineBody;
using ::clio::gv_bench::kmeans::UpdateBody;

/** Macro-form SeedCoro: write the point set in, one page at a time. */
CTP_GPU_FUN inline void SeedMacro(gv::DeviceVector<float> v, u64 per,
                                  u64 page_elems, u32 dims, u32 k,
                                  u64 base_idx, u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, base, static_cast<u64>(block) * per);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, n, 0);
  CLIO_YLOCAL(gv::Held<float>, h);
  CLIO_YBEGIN();
  base = static_cast<u64>(block) * per;
  for (; off < per; off += page_elems) {
    n = (off + page_elems <= per) ? page_elems : (per - off);
    CLIO_YCALL(v.MFetch(0, base + off, n));
    CLIO_YCALL(v.MHoldPage(&h, base + off, n, /*write=*/true));
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = PointVal(base_idx + base + off + i, dims, k);
    }
    // Collective: name the page just written.
    CLIO_YCALL(v.MBeginFlush(0, base + off, n));
    // Fetch is the pinner; UnpinRange is the releaser.
    v.UnpinRange(base + off, n);
  }
  // Only explicit flushes write back, so a page left in flight would be lost.
  CLIO_YCALL(v.MEndFlush());
  CLIO_YEND();
}

/** Macro-form AssignCoro: one Lloyd assignment pass over this block's slice. */
CTP_GPU_FUN inline void AssignMacro(gv::DeviceVector<float> v, u64 per,
                                    u64 page_elems, u32 dims, u32 k,
                                    const float *cent, float *sums,
                                    unsigned *counts, u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, base, static_cast<u64>(block) * per);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, n, 0);
  CLIO_YLOCAL(gv::Held<float>, h);
  CLIO_YBEGIN();
  base = static_cast<u64>(block) * per;
  for (; off < per; off += page_elems) {
    n = (off + page_elems <= per) ? page_elems : (per - off);
    CLIO_YCALL(v.MFetch(0, base + off, n));
    // Read-only: no write intent, so pages stay clean and shed without
    // writeback under oversubscription.
    CLIO_YCALL(v.MHoldPage(&h, base + off, n));
    const u64 npts = n / dims;
    for (u64 p = threadIdx.x; p < npts; p += blockDim.x) {
      const u64 pbase = base + off + p * dims;
      // The held page is indexed by ABSOLUTE element offset, so close over
      // that base rather than passing a raw pointer -- the arithmetic is then
      // provably the baselines'.
      struct PageAt {
        const gv::Held<float> &hh;
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
    v.UnpinRange(base + off, n);
  }
  CLIO_YEND();
}

}  // namespace clio::gv_bench::kmeans_macros

#endif  // CLIO_GV_BENCH_KMEANS_MACROS_KERNELS_H_
