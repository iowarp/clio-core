/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The three launches the weights benchmark needs, backend-neutral.
 *
 * THE SEAM, for the reason documented in kmeans_launch.h: a CUDA kernel is a
 * __global__ function at namespace scope; a SYCL kernel is a LAMBDA INSIDE
 * ITS LAUNCHER, and DPC++ member-checks the whole translation unit. The
 * workload is in weights_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_WEIGHTS_LAUNCH_H_
#define CLIO_GV_BENCH_WEIGHTS_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::weights {

using ::clio::run::u32;
using ::clio::run::u64;
using DevU32 = ::clio::cte::gpu_vector::DeviceVector<u32>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/** Per-block device state the SYCL backend allocates once; no-op on CUDA. */
void InitBackend(u32 max_blocks, const GpuInfo &info);

/** Write the synthetic model into the vector. Yieldable -- seeding is the
 *  phase that actually wedges, because every page it writes is dirty. */
void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v, u64 per,
                u64 page_elems, u32 flat_pct, View vw, StackView sv);

/** One inference sweep over the paged model. Yieldable. */
void LaunchWeights(dim3 grid, dim3 block, const GpuInfo &info, DevU32 v,
                   u64 per, u64 page_elems, unsigned long long *sum,
                   unsigned long long *page_sum, unsigned *page_visits,
                   View vw, StackView sv);

/** The staged-tile control: one launch per tile, nothing overlapped. */
void LaunchBaseline(u32 threads, const u32 *tile, u64 gbase, u64 n,
                    unsigned long long *sum, unsigned long long *page_sum,
                    unsigned *page_visits, u64 page_elems);

}  // namespace clio::gv_bench::weights

#endif  // CLIO_GV_BENCH_WEIGHTS_LAUNCH_H_
