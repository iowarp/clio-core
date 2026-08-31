/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The four launches Gray-Scott needs, as a backend-neutral interface.
 *
 * THE SEAM, for the same reason as kmeans_launch.h: a CUDA kernel is a
 * __global__ function at namespace scope, so host code may sit beside it and
 * be dropped from the device pass; a SYCL kernel is a LAMBDA INSIDE ITS
 * LAUNCHER, and DPC++ member-checks the whole translation unit. A TU holding
 * both the kernels and gpu_vector.h's host-only Vector cannot compile.
 *
 * The workload is in neither cuda/ nor sycl/ -- it is in
 * grayscott_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_GRAYSCOTT_LAUNCH_H_
#define CLIO_GV_BENCH_GRAYSCOTT_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::grayscott {

using ::clio::run::u32;
using ::clio::run::u64;
using DevF32 = ::clio::cte::gpu_vector::DeviceVector<float>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/** Per-block device state the SYCL backend allocates once; no-op on CUDA. */
void InitBackend(u32 max_blocks, const GpuInfo &info);

/** Seed u and v for every block's z-range, one plane per page. Yieldable. */
void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 zbase, u64 zend, View vw, StackView sv);

/** One Gray-Scott step over the paged field. Yieldable. */
void LaunchStep(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                u64 plane, u64 nx, u64 ny, u64 nz, u64 zper, u64 ubase,
                u64 vbase, u64 unext, u64 vnext, float Du, float Dv, float F,
                float K, float dt, u64 zbase, u64 zend, u64 gen, View vw,
                StackView sv);

/** Sum of v, for the correctness checksum. Yieldable. */
void LaunchSum(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
               u64 plane, u64 nz, u64 zper, u64 vbase, double *out,
               u64 zbase, u64 zend, View vw, StackView sv);

/** The staged-plane control: one launch per z, nothing overlapped. */
void LaunchBaseline(u32 threads, const float *uzm, const float *uz,
                    const float *uzp, const float *vzm, const float *vz,
                    const float *vzp, float *unx, float *vnx, u64 plane,
                    u64 nx, u64 ny, int interior, float Du, float Dv, float F,
                    float K, float dt);

}  // namespace clio::gv_bench::grayscott

#endif  // CLIO_GV_BENCH_GRAYSCOTT_LAUNCH_H_
