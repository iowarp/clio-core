/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_

#include "clio_runtime/ipc/ipc_gpu2cpu.h"
#include "clio_ctp/util/gpu_intrinsics.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

namespace clio::run {

#if CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER
/**
 * GPU-side SendIn.
 *
 * Producer-only design: the host pre-allocated the (self-contained) Task in a
 * registered backend and passed `task_ptr` to the kernel, which already
 * mutated the POD input fields. We stamp the task's size + clear its
 * completion flag in the embedded FutureInfo (task->fut_), build a
 * gpu::Future<TaskT>, and push a base-Task handle onto gpu2cpu_queue with a
 * system fence. There is no separate gpu::FutureShm — the Task is its own
 * completion record.
 *
 * Threading:
 *   - CUDA/ROCm: only thread 0 of the block enqueues; other threads
 *     return an empty future (caller is expected to broadcast).
 *   - SYCL: kernels are single_task by convention, so the full WI runs.
 */
template <typename TaskT>
CTP_GPU_FUN gpu::Future<TaskT> IpcGpu2Cpu::SendIn(
    gpu::IpcManager *ipc, const ctp::ipc::FullPtr<TaskT> &task_ptr) {
  gpu::Future<TaskT> future;

#if CTP_IS_GPU_COMPILER
  if (threadIdx.x != 0) return future;
#endif

  if (task_ptr.IsNull() || !ipc->gpu_info_.gpu2cpu_queue) {
    return future;
  }

  // Self-contained Task: stamp its POD size + clear its completion flag in the
  // embedded FutureInfo (no co-located gpu::FutureShm).
  const u32 task_size = static_cast<u32>(sizeof(TaskT));
  task_ptr.ptr_->fut_.task_size_ = task_size;
  task_ptr.ptr_->fut_.is_complete_.store(0);

  future = gpu::Future<TaskT>(task_ptr, task_size);

  // Queue entry uses the base Task type with raw addressing so the CPU worker
  // can dereference the task pointer directly; task_size rides along so it does
  // not need to read the task first.
  ctp::ipc::FullPtr<Task> task_for_queue;
  task_for_queue.shm_.alloc_id_ = task_ptr.shm_.alloc_id_;
  task_for_queue.shm_.off_ = reinterpret_cast<size_t>(task_ptr.ptr_);
  task_for_queue.ptr_ = static_cast<Task *>(task_ptr.ptr_);
  gpu::Future<Task> task_future(task_for_queue, task_size);

  CTP_DEVICE_FENCE_SYSTEM();
  auto &qlane = ipc->gpu_info_.gpu2cpu_queue->GetLane(0, 0);
  qlane.Push(task_future);
  return future;
}
#endif  // CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER

#if CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER
/**
 * GPU-side Wait.
 *
 * Polls the task's embedded completion flag (task->fut_.is_complete_) via a
 * volatile read. Backend is pinned host or UVM, so the CPU GPU-worker's
 * system-scope write is visible to a device-side volatile read through PCIe
 * cache snooping. is_complete_ is a ctp::ipc::atomic<u32> whose storage is its
 * first member `.x`.
 */
template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void gpu::Future<TaskT, AllocT>::Wait() {
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
#if CTP_IS_GPU_COMPILER
  if (threadIdx.x != 0) return;
#endif
  if (task_ptr_.IsNull()) return;
  volatile unsigned int *fp = reinterpret_cast<volatile unsigned int *>(
      &task_ptr_.ptr_->fut_.is_complete_.x);
  while (((*fp) & 1u) == 0u) {}
  CTP_DEVICE_FENCE_SYSTEM();
#endif
}
#endif  // CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#endif  // CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_
