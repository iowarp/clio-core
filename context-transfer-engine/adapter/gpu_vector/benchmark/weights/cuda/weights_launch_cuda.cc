/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Weights launches, CUDA. Compiled by clang-CUDA. Nothing here is workload:
 * every body lives in ../weights_kernels.h.
 */

#include "../weights_launch.h"
#include "../../gv_launch_bounds.h"

#include "../weights_kernels.h"

namespace clio::gv_bench::weights {

namespace {

__global__ GV_LAUNCH_BOUNDS void SeedKernel(GpuInfo info, DevU32 v, u64 per,
                                            u64 page_elems, u32 flat_pct,
                                            u64 base_idx, View yv,
                                            StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedLaneCoro(v, per, page_elems, flat_pct, base_idx,
                              yv.Block()));
}

__global__ GV_LAUNCH_BOUNDS void WeightsKernel(GpuInfo info, DevU32 v, u64 per, u64 page_elems,
                              unsigned long long *sum,
                              unsigned long long *page_sum,
                              unsigned *page_visits, u64 base_idx, View yv,
                              StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(WeightsLaneCoro(v, per, page_elems, sum, page_sum,
                                 page_visits, base_idx, yv.Block()));
}

__global__ GV_LAUNCH_BOUNDS void BaselineKernel(const u32 *tile, u64 gbase, u64 n,
                               unsigned long long *sum,
                               unsigned long long *page_sum,
                               unsigned *page_visits, u64 page_elems) {
  BaselineBody(tile, gbase, n, sum, page_sum, page_visits, page_elems);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // CUDA's per-block IpcManager is __shared__ storage, born fresh at every
  // launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v, u64 per,
                u64 page_elems, u32 flat_pct, u64 base_idx, View vw,
                StackView sv) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, flat_pct, base_idx, vw, sv);
}

void LaunchWeights(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v,
                   u64 per, u64 page_elems, unsigned long long *sum,
                   unsigned long long *page_sum, unsigned *page_visits,
                   u64 base_idx, View vw, StackView sv) {
  WeightsKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, sum, page_sum, page_visits, base_idx, vw, sv);
}

void LaunchBaseline(u32 threads, const u32 *tile, u64 gbase, u64 n,
                    unsigned long long *sum, unsigned long long *page_sum,
                    unsigned *page_visits, u64 page_elems) {
  BaselineKernel<<<1, threads>>>(tile, gbase, n, sum, page_sum, page_visits,
                                 page_elems);
}

}  // namespace clio::gv_bench::weights
