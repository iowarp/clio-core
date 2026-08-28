/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott launches, SYCL. Compiled with -fsycl; the ONLY file in this
 * benchmark that is.
 *
 * Nothing here is workload: every body lives in ../grayscott_kernels.h,
 * shared verbatim with the CUDA side. Compare StepKernel in
 * cuda/grayscott_launch_cuda.cc with the lambda below -- the prologue and the
 * coroutine call are identical; only the submission differs.
 *
 * NOTE the absence of a SyclYieldStackReset here. Exactly one -fsycl TU per
 * LINKED PROGRAM may define the yield device globals and that hook, and each
 * benchmark is its own program, so this file carries them -- see the
 * CLIO_SYCL_KERNEL_TU define below and the note in yield_stack.h about a
 * second device image re-registering a device_global.
 */

// Must precede every clio header.
#define CLIO_SYCL_KERNEL_TU 1

#include "../grayscott_launch.h"

#include "../grayscott_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::grayscott {

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

/** The yieldable prologue, shared by all three yieldable launches. */
template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 vec, View vw, StackView sv,
                     MakeCoro make) {
  Submit(grid, block, [=]() {
    DevF32 dev = vec;
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

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, vec, vw, sv, [=](DevF32 dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
    return SeedCoro(dev, plane, nx, ny, nz, z0, z1, ubase, vbase);
  });
}

void LaunchStep(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 unext, u64 vnext, float Du, float Dv, float F,
                float K, float dt, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, vec, vw, sv, [=](DevF32 dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
    return StepCoro(dev, plane, nx, ny, nz, z0, z1, ubase, vbase, unext, vnext,
                    Du, Dv, F, K, dt);
  });
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
               u64 plane, u64 nz, u64 zper, u64 vbase, double *out, View vw,
               StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, vec, vw, sv, [=](DevF32 dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < nz) ? (z0 + zper) : nz;
    return SumCoro(dev, plane, z0, z1, vbase, out);
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
