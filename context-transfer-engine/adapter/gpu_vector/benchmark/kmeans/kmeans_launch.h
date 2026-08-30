/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The four launches kmeans needs, as a backend-neutral interface.
 *
 * THIS HEADER IS THE SEAM. Above it, one host driver
 * (clio_kmeans_paged_bench.cc) -- ordinary C++, no kernels, no `<<<>>>`,
 * compiled once and identically for both backends. Below it, two tiny
 * translation units that differ only in how a grid is submitted:
 *
 *   cuda/kmeans_launch_cuda.cc    __global__ wrappers + Kernel<<<g, b, smem>>>
 *   sycl/kmeans_launch_sycl.cc    queue::parallel_for over an nd_range
 *
 * The WORKLOAD is in neither of them; it is in kmeans_kernels.h, once.
 *
 * Why the seam has to exist at all -- the thing that cannot be macro'd away:
 * a CUDA kernel is a `__global__` function at namespace scope, so host code
 * may sit beside it and the device pass drops that host code with
 * `#if !CTP_IS_DEVICE_PASS`. A SYCL kernel is a LAMBDA INSIDE ITS LAUNCHER,
 * and DPC++ member-checks the whole translation unit in its device pass --
 * so a TU holding both the kernels and gpu_vector.h's host-only `Vector`
 * cannot compile. Splitting at the launch keeps each side in a TU that can.
 */
#ifndef CLIO_GV_BENCH_KMEANS_LAUNCH_H_
#define CLIO_GV_BENCH_KMEANS_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::kmeans {

using ::clio::run::u32;
using ::clio::run::u64;
using DevF32 = ::clio::cte::gpu_vector::DeviceVector<float>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/**
 * One-time per-process setup the SYCL backend needs and CUDA does not: the
 * per-block IpcManager array that CUDA gets from __shared__ storage at every
 * launch. No-op under CUDA. Call once with the widest grid that will run.
 */
void InitBackend(u32 max_blocks, const GpuInfo &info);

/** Seed the point set into the vector, one page at a time. Yieldable. */
void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                u64 page_elems, u32 dims, u32 k, u64 base_idx, View vw,
                StackView sv);

/** One Lloyd assignment pass over the paged point set. Yieldable. */
void LaunchAssign(dim3 grid, dim3 block, const GpuInfo &info, DevF32 v, u64 per,
                  u64 page_elems, u32 dims, u32 k, const float *cent,
                  float *sums, unsigned *counts, View vw, StackView sv);

/** The staged-tile control: one launch per tile, nothing overlapped. */
void LaunchBaseline(u32 threads, const float *tile, u64 n, u32 dims, u32 k,
                    const float *cent, float *sums, unsigned *counts);

/** centroid = sum / count. */
void LaunchUpdate(float *cent, const float *sums, const unsigned *counts,
                  u32 dims, u32 k);

}  // namespace clio::gv_bench::kmeans

#endif  // CLIO_GV_BENCH_KMEANS_LAUNCH_H_
