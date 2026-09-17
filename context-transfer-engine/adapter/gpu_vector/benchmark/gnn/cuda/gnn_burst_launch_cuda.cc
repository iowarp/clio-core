/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * GNN burst launch, CUDA. Compiled by clang-CUDA (the vector's device API is
 * C++20 coroutines, which nvcc cannot compile).
 *
 * Nothing here is workload: the body lives in ../gnn_burst_kernels.h. This
 * wrapper exists only because a CUDA kernel is a `__global__` function.
 */

#include "../gnn_burst_launch.h"
#include "../../gv_launch_bounds.h"

#include "../gnn_burst_kernels.h"

namespace clio::gv_bench::gnn_burst {

namespace {

__global__ GV_LAUNCH_BOUNDS void BurstKernel(GpuInfo info, DevF32 vec,
                                             const u64 *pages, u64 npages,
                                             u64 epp, u32 nblocks, double *out,
                                             u32 dim, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BurstCoro(vec, pages, npages, epp, nblocks, yv.Block(), out,
                           dim));
}

}  // namespace

void LaunchBurst(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                 const u64 *d_pages, u64 npages, u64 epp, u32 nblocks,
                 double *d_out, u32 dim, View vw, StackView sv) {
  BurstKernel<<<grid, block>>>(info, vec, d_pages, npages, epp, nblocks, d_out,
                               dim, vw, sv);
}

}  // namespace clio::gv_bench::gnn_burst
