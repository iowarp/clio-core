/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Weights launches, SYCL. Compiled with -fsycl; the ONLY file in this
 * benchmark that is. Nothing here is workload -- every body lives in
 * ../weights_kernels.h, shared verbatim with the CUDA side.
 *
 * This TU also owns the yield device globals (CLIO_SYCL_KERNEL_TU) and
 * SyclYieldStackReset: exactly one -fsycl TU per linked program may, and
 * each benchmark is its own program.
 */

#define CLIO_SYCL_KERNEL_TU 1

#include "../weights_launch.h"

#include "../weights_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::weights {

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
void SubmitYieldable(dim3 grid, dim3 block, DevU32 v, View vw, StackView sv,
                     MakeCoro make) {
  Submit(grid, block, [=]() {
    DevU32 dev = v;
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

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v, u64 per,
                u64 page_elems, u32 flat_pct, View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, v, vw, sv, [=](DevU32 dev, u32 blk) {
    return SeedLaneCoro(dev, per, page_elems, flat_pct, blk);
  });
}

void LaunchWeights(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v,
                   u64 per, u64 page_elems, unsigned long long *sum,
                   unsigned long long *page_sum, unsigned *page_visits,
                   View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](DevU32 dev, u32 blk) {
    return WeightsLaneCoro(dev, per, page_elems, sum, page_sum, page_visits,
                           blk);
  });
}

void LaunchBaseline(u32 threads, const u32 *tile, u64 gbase, u64 n,
                    unsigned long long *sum, unsigned long long *page_sum,
                    unsigned *page_visits, u64 page_elems) {
  Submit(dim3(1), dim3(threads), [=]() {
    BaselineBody(tile, gbase, n, sum, page_sum, page_visits, page_elems);
  });
}

}  // namespace clio::gv_bench::weights

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
