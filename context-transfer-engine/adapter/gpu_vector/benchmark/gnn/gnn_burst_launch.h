/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The one launch the burst benchmark needs, as a backend-neutral interface.
 *
 * Same seam and same reason as grayscott_launch.h: a CUDA kernel is a
 * __global__ at namespace scope so host code may sit beside it and be dropped
 * from the device pass, and gv::Vector is host-only. The workload itself is in
 * gnn_burst_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_GNN_BURST_LAUNCH_H_
#define CLIO_GV_BENCH_GNN_BURST_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::gnn_burst {

using ::clio::run::u32;
using ::clio::run::u64;
using DevF32 = ::clio::cte::gpu_vector::DeviceVector<float>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/** Stream one burst's page list across the grid. Yieldable. */
void LaunchBurst(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                 const u64 *d_pages, u64 npages, u64 epp, u32 nblocks,
                 double *d_out, u32 dim, View vw, StackView sv);

}  // namespace clio::gv_bench::gnn_burst

#endif  // CLIO_GV_BENCH_GNN_BURST_LAUNCH_H_
