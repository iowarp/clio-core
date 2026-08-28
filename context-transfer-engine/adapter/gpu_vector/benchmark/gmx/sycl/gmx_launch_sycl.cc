/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PME launches, SYCL. Compiled with -fsycl; the ONLY file in this benchmark
 * that is.
 *
 * Nothing here is workload: every body lives in ../gmx_kernels.h, shared
 * verbatim with the CUDA side. This TU also owns the yield device globals
 * (CLIO_SYCL_KERNEL_TU) and SyclYieldStackReset -- exactly one -fsycl TU per
 * linked program may, and each benchmark is its own program.
 */

#define CLIO_SYCL_KERNEL_TU 1

#include "../gmx_launch.h"

#include "../gmx_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::gmx {

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
void SubmitYieldable(dim3 grid, dim3 block, DevMesh mesh, View vw,
                     StackView sv, MakeCoro make) {
  Submit(grid, block, [=]() {
    DevMesh dev = mesh;
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

void LaunchZero(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                u64 K, u64 plane, u64 zper, View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, mesh, vw, sv, [=](DevMesh dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
    return ZeroCoro(dev, plane, z0, z1);
  });
}

void LaunchSpread(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 zper, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](DevMesh dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
    return SpreadCoro(dev, ax, ay, az, aq, bin_start, K, plane, z0, z1);
  });
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh, u64 K,
               u64 plane, u64 zper, unsigned long long *out, View vw,
               StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](DevMesh dev, u32 blk) {
    const u64 z0 = static_cast<u64>(blk) * zper;
    const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
    return SumCoro(dev, K, plane, z0, z1, out);
  });
}

void LaunchGather(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 bper, unsigned long long *out, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, mesh, vw, sv, [=](DevMesh dev, u32 blk) {
    const u64 b0 = static_cast<u64>(blk) * bper;
    const u64 b1 = (b0 + bper < K) ? (b0 + bper) : K;
    return GatherCoro(dev, ax, ay, az, aq, bin_start, K, plane, b0, b1, out);
  });
}

void LaunchDenseSpread(u32 blocks, u32 threads, unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 zper) {
  Submit(dim3(blocks), dim3(threads), [=]() {
    DenseSpreadBody(mesh, ax, ay, az, aq, bin_start, K, plane, zper);
  });
}

void LaunchDenseSum(const unsigned long long *mesh, u64 n,
                    unsigned long long *out) {
  Submit(dim3(64), dim3(256), [=]() { DenseSumBody(mesh, n, out); });
}

void LaunchDenseGather(u32 blocks, u32 threads, const unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 bper, unsigned long long *out) {
  Submit(dim3(blocks), dim3(threads), [=]() {
    DenseGatherBody(mesh, ax, ay, az, aq, bin_start, K, plane, bper, out);
  });
}

}  // namespace clio::gv_bench::gmx

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
