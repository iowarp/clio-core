/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SYCL launches for the MACRO-form kmeans paged bench.
 *
 * Same structure as kmeans_launch_sycl.cc; the only difference is the
 * yieldable prologue. The coroutine version ends in CLIO_YCORO_RUN(make(...)),
 * which drives a coroutine handle. The macro form has no handle: the body IS
 * the resumable function, so the prologue is CLIO_YKERNEL_ENTER followed by a
 * direct call, and the switch inside the callee resumes it.
 *
 * This TU must be compiled with -fsycl and linked as a SHARED library --
 * device code in a static archive does not reach the executable's device
 * image, and the first kernel then aborts in ProgramManager.
 */
// Must precede every clio header: selects the SYCL kernel-TU spellings.
#define CLIO_SYCL_KERNEL_TU 1

#include "../kmeans_macros_launch.h"
#include "../kmeans_macros_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::kmeans_macros {
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
template <typename BodyT>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 v, View vw, StackView sv,
                     BodyT run) {
  Submit(grid, block, [=]() {
    DevF32 dev = v;
    dev.Init(vw.Block());
    // Publishes the block's stack base and zeroes the frame depth. This is
    // what CLIO_YCORO_RUN's prologue does too -- the mechanisms share one
    // stack and one host protocol.
    CLIO_YKERNEL_ENTER(vw, sv);
    run(dev, vw.Block());
  });
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, u64 base_idx, View vw,
                StackView sv) {
  // gpu_info is already stamped into every block's record by InitBackend.
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](DevF32 dev, u32 blk) {
    SeedMacro(dev, per, page_elems, dims, k, base_idx, blk);
  });
}

void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, v, vw, sv, [=](DevF32 dev, u32 blk) {
    AssignMacro(dev, per, page_elems, dims, k, cent, sums, counts, blk);
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

}  // namespace clio::gv_bench::kmeans_macros

namespace clio::run::gpu {

/** The out-of-line half of YieldStack::Reset (declared in yield_stack.h).
 *  Exactly one -fsycl TU per linked program may define it; for the macro-form
 *  bench that TU is this one, since it does not link the coroutine edition's
 *  launch library. */
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
