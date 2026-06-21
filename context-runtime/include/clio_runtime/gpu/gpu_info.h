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

#ifndef CLIO_RUNTIME_INCLUDE_GPU_INFO_H_
#define CLIO_RUNTIME_INCLUDE_GPU_INFO_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

/**
 * Per-device GPU IPC handle passed by value to GPU kernels.
 *
 * After the producer-only redesign there is exactly one queue direction:
 * GPU-to-CPU. Clients allocate task and data backends on the host and
 * register them with the runtime; kernels do not allocate. The kernel
 * needs only the gpu2cpu queue pointer and the device id to push tasks.
 */
struct IpcManagerGpuInfo {
  /** GPU->CPU queue (pinned host memory; CPU worker polls, GPU pushes). */
  GpuTaskQueue *gpu2cpu_queue = nullptr;

  /** Logical GPU device id. */
  u32 gpu_id = 0;

  /** Ring-buffer depth per lane (informational). */
  u32 gpu_queue_depth = 16;

  CTP_CROSS_FUN IpcManagerGpuInfo() = default;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_GPU_INFO_H_
