/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * kmeans launches, SYCL. Compiled with -fsycl; the ONLY file in this
 * benchmark that is.
 *
 * Nothing here is workload: every body lives in ../kmeans_kernels.h, shared
 * verbatim with the CUDA side. Compare SeedKernel in
 * cuda/kmeans_launch_cuda.cc with the lambda below -- the prologue and the
 * coroutine call are identical, and only the submission differs.
 *
 * Three things this file must own that CUDA does not need:
 *
 *   1. THE DEVICE GLOBALS. It defines CLIO_SYCL_KERNEL_TU, which makes
 *      yield_stack.h and gpu_ipc_manager.h emit their device_global
 *      instances HERE and nowhere else. A second device image re-registering
 *      them aborts the DPC++ runtime at startup.
 *   2. The per-block IpcManager array (InitBackend), which CUDA gets from
 *      __shared__ storage at every launch.
 *   3. SyclYieldStackReset -- the out-of-line half of YieldStack::Reset,
 *      which submits a kernel and so cannot live in a header a plain host
 *      TU includes.
 *
 * The nd_range's work-group IS the CUDA block, which is what makes
 * threadIdx / blockIdx / __syncthreads mean the same thing on both sides
 * (see clio_ctp/util/sycl_cuda_compat.h). There is no dynamic-shared-memory
 * argument because YieldTls keeps the block's state in global memory under
 * SYCL.
 */

// Must precede every clio header: see (1) above.
#define CLIO_SYCL_KERNEL_TU 1

#include "../kmeans_launch.h"

#include "../kmeans_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::kmeans {

namespace {

/** Submit one grid and wait, in CUDA's (grid, block) shape. */
template <typename BodyT>
void Submit(dim3 grid, dim3 block, BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>) { body(); })
      .wait();
}

/** The yieldable prologue, shared by both yieldable launches below. */
template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 v, View vw, StackView sv,
                     MakeCoro make) {
  Submit(grid, block, [=]() {
    DevF32 dev = v;
    dev.Init(vw.Block());
    ::clio::run::gpu::YieldTlsPublish(sv, vw.Y(), vw.Block());
    __syncthreads();
    CLIO_YCORO_RUN(make(dev, vw.Block()));
  });
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, u64 base_idx, View vw,
                StackView sv) {
  // gpu_info is already stamped into every block's record by InitBackend, so
  // unlike CUDA there is no per-launch CLIO_GPU_INIT store.
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](DevF32 dev, u32 blk) {
    return SeedCoro(dev, per, page_elems, dims, k, base_idx, blk);
  });
}

void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](DevF32 dev, u32 blk) {
    return AssignCoro(dev, per, page_elems, dims, k, cent, sums, counts, blk);
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
