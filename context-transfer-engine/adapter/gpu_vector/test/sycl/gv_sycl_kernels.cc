/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The gpu_vector SYCL kernels. Compiled with -fsycl; see gv_sycl_kernels.h
 * for why this is a translation unit of its own.
 *
 * THE COROUTINES BELOW ARE THE CUDA ONES, UNCHANGED. Compare FillCoro here
 * with FillCoro in ../test_gpu_vector_smoke_gpu.cc: same co_awaits, same
 * threadIdx/blockDim strided loop, same flush-as-you-go, same UnpinRange.
 * That is the result this port is for -- the paging code is backend-neutral
 * and only the launch is not.
 *
 * What differs, and all of it is in `Submit`:
 *   - a lambda on a queue instead of `Kernel<<<grid, block, smem>>>`;
 *   - no dynamic-shared-memory argument, because YieldTls keeps the block's
 *     state in global memory under SYCL (see yield_stack.h);
 *   - an nd_range whose work-group is the CUDA block, which is what makes
 *     threadIdx / __syncthreads mean the same thing (see sycl_cuda_compat.h).
 */

// THIS TU OWNS THE DEVICE GLOBALS. Must be defined before any clio header:
// it is what makes yield_stack.h and gpu_ipc_manager.h emit their
// device_global instances here and nowhere else (a second device image
// re-registering them aborts the DPC++ runtime at startup).
#define CLIO_SYCL_KERNEL_TU 1

#include "gv_sycl_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::cte::gpu_vector::sycl_test {

namespace gy = ::clio::run::gpu;

namespace {

/* ---------------------------------------------------------------- */
/* The coroutines -- identical to the CUDA smoke test's.             */
/* ---------------------------------------------------------------- */

gy::YCoroMain FillCoro(DevU32 v, clio::run::u64 n) {
  for (clio::run::u64 i = 0; i < n;) {
    clio::run::u64 run = 0;
    {
      co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
      auto h = co_await v.HoldPage(i, n - i, /*write=*/true);
      run = h.run();
      for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[i + k] = static_cast<clio::run::u32>((i + k) * 7 + 1);
      }
    }   // guard dies here: the page is unpinned but still DIRTY
    // Flush as we go: the vector never writes back on its own, so a dirty
    // page is unevictable until the caller flushes it.
    co_await v.Flush(0, i, run);
    // Give the pin back -- Fetch is the pinner.
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
}

/** `bad[0]` is the total; `bad[1 + page]` is that page's share. Splitting it
 *  per page is what turns "12288 wrong" into "pages 4..15 wrong", which is
 *  the difference between a number and a diagnosis. */
gy::YCoroMain CheckCoro(DevU32 v, clio::run::u64 n, unsigned long long *bad) {
  for (clio::run::u64 i = 0; i < n;) {
    co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
    auto h = co_await v.HoldPage(i, n - i);
    for (clio::run::u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      if (h[i + k] != static_cast<clio::run::u32>((i + k) * 7 + 1)) {
        atomicAdd(bad, 1ull);
        atomicAdd(bad + 1 + v.PageOf(i + k), 1ull);
      }
    }
    const clio::run::u64 run = h.run();
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
}

gy::YCoroMain MultiFillCoro(DevU32 v, clio::run::u64 per,
                            clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 i = 0; i < per;) {
    clio::run::u64 run = 0;
    {
      co_await v.Fetch(0, v.PageLo(base + i), v.PageSpan(base + i, 1));
      auto h = co_await v.HoldPage(base + i, per - i, /*write=*/true);
      run = h.run();
      for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[base + i + k] = static_cast<clio::run::u32>((base + i + k) * 7 + 1);
      }
    }
    co_await v.Flush(0, base + i, run);
    v.UnpinRange(v.PageLo(base + i), v.PageSpan(base + i, 1));
    i += run;
  }
}

gy::YCoroMain MultiCheckCoro(DevU32 v, clio::run::u64 per,
                             unsigned long long *bad, clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 i = 0; i < per;) {
    co_await v.Fetch(0, v.PageLo(base + i), v.PageSpan(base + i, 1));
    auto h = co_await v.HoldPage(base + i, per - i);
    for (clio::run::u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      if (h[base + i + k] !=
          static_cast<clio::run::u32>((base + i + k) * 7 + 1)) {
        atomicAdd(bad, 1ull);
      }
    }
    const clio::run::u64 run = h.run();
    v.UnpinRange(v.PageLo(base + i), v.PageSpan(base + i, 1));
    i += run;
  }
}

/**
 * Submit one round's grid and wait for it.
 *
 * `body(block)` receives the LOGICAL block id -- not blockIdx.x. The driver
 * relaunches a COMPACTED grid of whichever blocks are still pending, so
 * blockIdx.x is not stable across resumes and anything partitioned by block
 * (a page-table binding, a slice of the vector) must use the logical id.
 */
template <typename BodyT>
void Submit(dim3 grid, dim3 block, DevU32 v, View vw, StackView sv,
            BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>) {
         // No CLIO_GPU_INIT store here: SyclInitBlockIpcManagers already
         // stamped gpu_info into every block's record on the host, and
         // GetBlockIpcManager finds this block's by symbol lookup.
         DevU32 dev = v;
         dev.Init(vw.Block());
         gy::YieldTlsPublish(sv, vw.Y(), vw.Block());
         __syncthreads();
         body(dev, vw.Block());
       })
      .wait();
}

}  // namespace

void InitBlockIpc(clio::run::u32 max_blocks,
                  const clio::run::IpcManagerGpuInfo &gpu_info) {
  gy::SyclInitBlockIpcManagers(max_blocks, gpu_info);
}

void LaunchFill(dim3 grid, dim3 block, DevU32 v, clio::run::u64 n, View vw,
                StackView sv) {
  Submit(grid, block, v, vw, sv, [=](DevU32 dev, clio::run::u32) {
    CLIO_YCORO_RUN(FillCoro(dev, n));
  });
}

void LaunchCheck(dim3 grid, dim3 block, DevU32 v, clio::run::u64 n,
                 unsigned long long *bad, View vw, StackView sv) {
  Submit(grid, block, v, vw, sv, [=](DevU32 dev, clio::run::u32) {
    CLIO_YCORO_RUN(CheckCoro(dev, n, bad));
  });
}

void LaunchMultiFill(dim3 grid, dim3 block, DevU32 v, clio::run::u64 per,
                     View vw, StackView sv) {
  Submit(grid, block, v, vw, sv, [=](DevU32 dev, clio::run::u32 b) {
    CLIO_YCORO_RUN(MultiFillCoro(dev, per, b));
  });
}

void LaunchMultiCheck(dim3 grid, dim3 block, DevU32 v, clio::run::u64 per,
                      unsigned long long *bad, View vw, StackView sv) {
  Submit(grid, block, v, vw, sv, [=](DevU32 dev, clio::run::u32 b) {
    CLIO_YCORO_RUN(MultiCheckCoro(dev, per, bad, b));
  });
}

}  // namespace clio::cte::gpu_vector::sycl_test

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
     // sp_ starts AFTER the header: the header is not frame space.
     h->sp_ = sizeof(YieldLaneHeader);
     h->live_depth_ = 0;
     h->cur_depth_ = 0;
     h->error_ = kYieldErrNone;
     h->coro_resume_ = 0;
     h->coro_top_ = 0;
     h->coro_park_ = 0;
   }).wait();
  // Install the per-block YieldSmem base for THIS stack. Done per reset
  // rather than once at startup because a program may run several stacks,
  // and whichever one is about to launch owns the device_global.
  char *base = smem_base;
  q.copy(&base, g_yield_smem_dg, 1).wait();
}

}  // namespace clio::run::gpu
