/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The seven launches PME spread/gather needs, as a backend-neutral interface.
 *
 * THE SEAM, for the same reason as kmeans_launch.h and grayscott_launch.h: a
 * CUDA kernel is a __global__ function at namespace scope; a SYCL kernel is a
 * LAMBDA INSIDE ITS LAUNCHER, and DPC++ member-checks the whole translation
 * unit, so a TU holding both the kernels and gpu_vector.h's host-only Vector
 * cannot compile. The workload is in gmx_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_GMX_LAUNCH_H_
#define CLIO_GV_BENCH_GMX_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::gmx {

using ::clio::run::u32;
using ::clio::run::u64;
using DevMesh = ::clio::cte::gpu_vector::DeviceVector<unsigned long long>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/** Per-block device state the SYCL backend allocates once; no-op on CUDA. */
void InitBackend(u32 max_blocks, const GpuInfo &info);

/* ---- paged path (yieldable) ---- */
void LaunchZero(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                u64 K, u64 plane, u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv);

void LaunchSpread(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 zper, u64 zbase, u64 zend, View vw,
                  StackView sv);

void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh, u64 K,
               u64 plane, u64 zper, unsigned long long *out, u64 zbase, u64 zend, View vw,
               StackView sv);

void LaunchGather(dim3 grid, dim3 block, const GpuInfo &info, DevMesh mesh,
                  const float *ax, const float *ay, const float *az,
                  const long long *aq, const u32 *bin_start, u64 K, u64 plane,
                  u64 bper, unsigned long long *out, u64 zbase, u64 zend, View vw,
                  StackView sv);

/* ---- dense reference: the whole mesh resident, no paging ---- */
void LaunchDenseSpread(u32 blocks, u32 threads, unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 zper);

void LaunchDenseSum(const unsigned long long *mesh, u64 n,
                    unsigned long long *out);

void LaunchDenseGather(u32 blocks, u32 threads, const unsigned long long *mesh,
                       const float *ax, const float *ay, const float *az,
                       const long long *aq, const u32 *bin_start, u64 K,
                       u64 plane, u64 bper, unsigned long long *out);

}  // namespace clio::gv_bench::gmx

#endif  // CLIO_GV_BENCH_GMX_LAUNCH_H_
