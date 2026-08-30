/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PME launches, CUDA. Compiled by clang-CUDA (the vector's device API is
 * C++20 coroutines, which nvcc cannot compile).
 *
 * Nothing here is workload: every body lives in ../gmx_kernels.h.
 */

#include "../gmx_launch.h"
#include "../../gv_launch_bounds.h"

#include "../gmx_kernels.h"

namespace clio::gv_bench::gmx {

namespace {

__global__ GV_LAUNCH_BOUNDS void ZeroKernel(GpuInfo info, DevMesh mesh, u64 K, u64 plane,
                           u64 zper, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
  CLIO_YCORO_RUN(ZeroCoro(mesh, plane, z0, z1));
}

__global__ GV_LAUNCH_BOUNDS void SpreadKernel(GpuInfo info, DevMesh mesh, const float *ax,
                             const float *ay, const float *az,
                             const long long *aq, const u32 *bin_start, u64 K,
                             u64 plane, u64 zper, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
  CLIO_YCORO_RUN(SpreadCoro(mesh, ax, ay, az, aq, bin_start, K, plane, z0, z1));
}

__global__ GV_LAUNCH_BOUNDS void SumKernel(GpuInfo info, DevMesh mesh, u64 K, u64 plane,
                          u64 zper, unsigned long long *out, View yv,
                          StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
  CLIO_YCORO_RUN(SumCoro(mesh, K, plane, z0, z1, out));
}

__global__ GV_LAUNCH_BOUNDS void GatherKernel(GpuInfo info, DevMesh mesh, const float *ax,
                             const float *ay, const float *az,
                             const long long *aq, const u32 *bin_start, u64 K,
                             u64 plane, u64 bper, unsigned long long *out,
                             View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  mesh.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 b0 = static_cast<u64>(yv.Block()) * bper;
  const u64 b1 = (b0 + bper < K) ? (b0 + bper) : K;
  CLIO_YCORO_RUN(GatherCoro(mesh, ax, ay, az, aq, bin_start, K, plane, b0, b1,
                            out));
}

__global__ GV_LAUNCH_BOUNDS void DenseSpreadKernel(unsigned long long *mesh, const float *ax,
                                  const float *ay, const float *az,
                                  const long long *aq, const u32 *bin_start,
                                  u64 K, u64 plane, u64 zper) {
  DenseSpreadBody(mesh, ax, ay, az, aq, bin_start, K, plane, zper);
}

__global__ GV_LAUNCH_BOUNDS void DenseSumKernel(const unsigned long long *mesh, u64 n,
                               unsigned long long *out) {
  DenseSumBody(mesh, n, out);
}

__global__ GV_LAUNCH_BOUNDS void DenseGatherKernel(const unsigned long long *mesh,
                                  const float *ax, const float *ay,
                                  const float *az, const long long *aq,
                                  const u32 *bin_start, u64 K, u64 plane,
                                  u64 bper, unsigned long long *out) {
  DenseGatherBody(mesh, ax, ay, az, aq, bin_start, K, plane, bper, out);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // CUDA's per-block IpcManager is __shared__ storage, born fresh at every
  // launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchZero(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                u64 K, u64 plane, u64 zper, View vw, StackView sv) {
  ZeroKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, mesh, K, plane,
                                                     zper, vw, sv);
}

void LaunchSpread(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 zper, View vw, StackView sv) {
  SpreadKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, ax, ay, az, aq, bin_start, K, plane, zper, vw, sv);
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh, u64 K,
               u64 plane, u64 zper, unsigned long long *out, View vw,
               StackView sv) {
  SumKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, mesh, K, plane, zper,
                                                    out, vw, sv);
}

void LaunchGather(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 bper, unsigned long long *out, View vw, StackView sv) {
  GatherKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, mesh, ax, ay, az, aq, bin_start, K, plane, bper, out, vw, sv);
}

void LaunchDenseSpread(u32 blocks, u32 threads, unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 zper) {
  DenseSpreadKernel<<<blocks, threads>>>(mesh, ax, ay, az, aq, bin_start, K,
                                         plane, zper);
}

void LaunchDenseSum(const unsigned long long *mesh, u64 n,
                    unsigned long long *out) {
  DenseSumKernel<<<64, 256>>>(mesh, n, out);
}

void LaunchDenseGather(u32 blocks, u32 threads, const unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 bper, unsigned long long *out) {
  DenseGatherKernel<<<blocks, threads>>>(mesh, ax, ay, az, aq, bin_start, K,
                                         plane, bper, out);
}

}  // namespace clio::gv_bench::gmx
