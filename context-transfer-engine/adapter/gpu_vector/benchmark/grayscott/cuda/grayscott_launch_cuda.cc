/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott launches, CUDA. Compiled by clang-CUDA (the vector's device API
 * is C++20 coroutines, which nvcc cannot compile).
 *
 * Nothing here is workload: every body lives in ../grayscott_kernels.h.
 * These wrappers exist only because a CUDA kernel is a `__global__` function
 * and a SYCL kernel is not. See ../grayscott_launch.h.
 */

#include "../grayscott_launch.h"
#include "../../gv_launch_bounds.h"

#include "../grayscott_kernels.h"

namespace clio::gv_bench::grayscott {

namespace {

__global__ GV_LAUNCH_BOUNDS void SeedKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nx, u64 ny,
                           u64 nz, u64 zper, u64 ubase, u64 vbase, u64 zbase,
                           u64 zend, View yv,
                           StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_YCORO_RUN(SeedCoro(vec, plane, nx, ny, nz, z0, z1, ubase, vbase));
}

__global__ GV_LAUNCH_BOUNDS void StepKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nx, u64 ny,
                           u64 nz, u64 zper, u64 ubase, u64 vbase, u64 zbase,
                           u64 zend, u64 unext,
                           u64 vnext, float Du, float Dv, float F, float K,
                           float dt, u64 gen, View yv, StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_YCORO_RUN(StepCoro(vec, plane, nx, ny, nz, z0, z1, zbase, zend, gen, ubase, vbase, unext,
                          vnext, Du, Dv, F, K, dt));
}

__global__ GV_LAUNCH_BOUNDS void SumKernel(GpuInfo info, DevF32 vec, u64 plane, u64 nz,
                          u64 zper, u64 vbase, double *out, u64 zbase, u64 zend, View yv,
                          StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  const u64 z0 = zbase + static_cast<u64>(yv.Block()) * zper;
  const u64 z1 = (z0 + zper < zend) ? (z0 + zper) : zend;
  CLIO_YCORO_RUN(SumCoro(vec, plane, z0, z1, vbase, out));
}

__global__ GV_LAUNCH_BOUNDS void BaselineKernel(const float *uzm, const float *uz,
                               const float *uzp, const float *vzm,
                               const float *vz, const float *vzp, float *unx,
                               float *vnx, u64 plane, u64 nx, u64 ny,
                               int interior, float Du, float Dv, float F,
                               float K, float dt) {
  BaselineBody(uzm, uz, uzp, vzm, vz, vzp, unx, vnx, plane, nx, ny, interior,
               Du, Dv, F, K, dt);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // Nothing to do: CUDA's per-block IpcManager is __shared__ storage, born
  // fresh at every launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 zbase, u64 zend, View vw, StackView sv) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nx, ny, nz, zper, ubase, vbase, zbase, zend, vw,
      sv);
}

void LaunchStep(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 unext, u64 vnext, float Du, float Dv, float F,
                float K, float dt, u64 zbase, u64 zend, u64 gen, View vw,
                StackView sv) {
  StepKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nx, ny, nz, zper, ubase, vbase, zbase, zend, unext,
      vnext, Du, Dv, F, K, dt, gen, vw, sv);
}

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
               u64 plane, u64 nz, u64 zper, u64 vbase, double *out,
               u64 zbase, u64 zend, View vw, StackView sv) {
  SumKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
      info, vec, plane, nz, zper, vbase, out, zbase, zend, vw, sv);
}

void LaunchBaseline(u32 threads, const float *uzm, const float *uz,
                    const float *uzp, const float *vzm, const float *vz,
                    const float *vzp, float *unx, float *vnx, u64 plane,
                    u64 nx, u64 ny, int interior, float Du, float Dv, float F,
                    float K, float dt) {
  BaselineKernel<<<1, threads>>>(uzm, uz, uzp, vzm, vz, vzp, unx, vnx, plane,
                                 nx, ny, interior, Du, Dv, F, K, dt);
}

}  // namespace clio::gv_bench::grayscott
