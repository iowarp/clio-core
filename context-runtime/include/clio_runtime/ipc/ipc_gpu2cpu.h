/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_H_
#define CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

namespace clio::run {

class IpcManager;
namespace gpu { class IpcManager; }

/**
 * IPC transport for GPU client → CPU runtime.
 *
 * Producer-only design: kernels do not allocate. The host pre-allocates
 * Task+FutureShm pairs in a registered device-memory backend before
 * launch. ClientSend just initializes the FutureShm flags and pushes
 * onto gpu2cpu_queue. The CPU worker (Worker::ProcessNewTaskGpu in
 * worker.cc) resolves both ShmPtrs via the per-device registered
 * backend map, copies the POD task into a CPU-side scratch slot, and
 * dispatches it on the local runtime. RuntimeSend writes the POD output
 * bytes back to the original device buffer and sets FUTURE_COMPLETE on
 * the gpu::FutureShm so the kernel poll-loop unblocks.
 */
struct IpcGpu2Cpu {
  /**
   * GPU-side: initialize the co-located gpu::FutureShm and push the task
   * onto gpu2cpu_queue.
   *
   * @param ipc gpu::IpcManager pointer (provides gpu2cpu_queue handle).
   * @param task_ptr Pre-allocated task (host built it; this kernel mutated
   *                 the POD input fields). The FutureShm lives at
   *                 task_ptr + sizeof(TaskT).
   * @return gpu::Future<TaskT> bound to the FutureShm.
   */
  template <typename TaskT>
  static CTP_GPU_FUN gpu::Future<TaskT> ClientSend(
      gpu::IpcManager *ipc, const ctp::ipc::FullPtr<TaskT> &task_ptr);

  /**
   * CPU-side: resolve the popped gpu::Future<Task> into a host-readable
   * task pointer, copying POD bytes from device memory if needed. Called
   * by Worker::ProcessNewTaskGpu before dispatch.
   *
   * Note: with the producer-only redesign this no longer touches a
   * lightbeam transport — the GPU never serializes through ZMQ.
   */
  static ctp::ipc::FullPtr<Task> RuntimeRecv(
      IpcManager *ipc, Future<Task> &future, Container *container,
      u32 method_id, ctp::lbm::Transport *recv_transport);

  /**
   * CPU-side: write POD output bytes back to the original device buffer
   * and signal FUTURE_COMPLETE on the gpu::FutureShm so the kernel
   * unblocks.
   */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container);
};

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#endif  // CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_H_
