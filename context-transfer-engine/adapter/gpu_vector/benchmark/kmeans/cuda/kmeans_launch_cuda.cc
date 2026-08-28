/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * kmeans launches, CUDA. Compiled by clang-CUDA (the vector's device API is
 * C++20 coroutines, which nvcc cannot compile).
 *
 * Nothing here is workload: every body lives in ../kmeans_kernels.h. These
 * wrappers exist only because a CUDA kernel is a `__global__` function and a
 * SYCL kernel is not. See ../kmeans_launch.h for the seam.
 */

#include "../kmeans_launch.h"

#include "../kmeans_kernels.h"

namespace clio::gv_bench::kmeans {

namespace {

/** The prologue every yieldable kernel repeats: bind this block to its page
 *  table, publish the lane stack, then run the chain. Identical to the SYCL
 *  side's, which is the point. */
__global__ void SeedKernel(GpuInfo info, DevF32 v, u64 per, u64 page_elems,
                           u32 dims, u32 k, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedCoro(v, per, page_elems, dims, k, yv.Block()));
}

__global__ void AssignKernel(GpuInfo info, DevF32 v, u64 per, u64 page_elems,
                             u32 dims, u32 k, const float *cent, float *sums,
                             unsigned *counts, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(AssignCoro(v, per, page_elems, dims, k, cent, sums, counts,
                            yv.Block()));
}

__global__ void BaselineKernel(const float *tile, u64 n, u32 dims, u32 k,
                               const float *cent, float *sums,
                               unsigned *counts) {
  BaselineBody(tile, n, dims, k, cent, sums, counts);
}

__global__ void UpdateKernel(float *cent, const float *sums,
                             const unsigned *counts, u32 dims, u32 k) {
  UpdateBody(cent, sums, counts, dims, k);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // Nothing to do: CUDA's per-block IpcManager is __shared__ storage, born
  // fresh at every launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, View vw, StackView sv) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, v, per, page_elems,
                                                     dims, k, vw, sv);
}

void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv) {
  AssignKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, v, per, page_elems, dims, k, cent, sums, counts, vw, sv);
}

void LaunchBaseline(u32 threads, const float *tile, u64 n, u32 dims, u32 k,
                    const float *cent, float *sums, unsigned *counts) {
  BaselineKernel<<<1, threads>>>(tile, n, dims, k, cent, sums, counts);
}

void LaunchUpdate(float *cent, const float *sums, const unsigned *counts,
                  u32 dims, u32 k) {
  UpdateKernel<<<(k + 63) / 64, 64>>>(cent, sums, counts, dims, k);
}

}  // namespace clio::gv_bench::kmeans
