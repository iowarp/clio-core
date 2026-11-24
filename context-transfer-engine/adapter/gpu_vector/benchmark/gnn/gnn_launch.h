/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * The launches the distributed GNN benchmark needs, as a backend-neutral
 * interface.
 *
 * THE SEAM, for the same reason as grayscott_launch.h: a CUDA kernel is a
 * __global__ function at namespace scope, so it can sit in its own TU beside
 * the device code, while the host driver includes gpu_vector.h's host-only
 * Vector. The workload itself lives in gnn_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_GNN_LAUNCH_H_
#define CLIO_GV_BENCH_GNN_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "gnn_kernels.h"

namespace clio::gv_bench::gnn {

using DevF32 = ::clio::cte::gpu_vector::DeviceVector<float>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/**
 * Seed every block's owned X pages and publish them. Yieldable.
 *
 * @param grid  launch grid (from the yield driver)
 * @param block launch block shape
 * @param info  GPU IPC info for CLIO_GPU_INIT
 * @param vec   the paged vector
 * @param p     launch parameters
 * @param vw    yield driver view
 * @param sv    yield stack view
 */
void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                const GnnParams &p, View vw, StackView sv);

/**
 * Run one GraphSAGE layer over every block's owned pages. Yieldable.
 *
 * @param grid  launch grid (from the yield driver)
 * @param block launch block shape
 * @param info  GPU IPC info for CLIO_GPU_INIT
 * @param vec   the paged vector
 * @param p     launch parameters
 * @param layer 1 (X -> H, published) or 2 (H -> logits)
 * @param vw    yield driver view
 * @param sv    yield stack view
 */
void LaunchLayer(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                 const GnnParams &p, u32 layer, View vw, StackView sv);

}  // namespace clio::gv_bench::gnn

#endif  // CLIO_GV_BENCH_GNN_LAUNCH_H_
