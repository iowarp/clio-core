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
 * GPU-side ClientSend.
 *
 * Producer-only design: the host pre-allocated Task+FutureShm in a
 * registered backend and passed `task_ptr` to the kernel. The kernel
 * already mutated POD input fields. We just clear the FutureShm flags,
 * build a gpu::Future<TaskT> carrying ShmPtrs to both the task and its
 * co-located FutureShm, then push onto gpu2cpu_queue with a system fence.
 *
 * Threading:
 *   - CUDA/ROCm: only thread 0 of the block enqueues; other threads
 *     return an empty future (caller is expected to broadcast).
 *   - SYCL: kernels are single_task by convention, so the full WI runs.
 */
template <typename TaskT>
CTP_GPU_FUN gpu::Future<TaskT> IpcGpu2Cpu::ClientSend(
    gpu::IpcManager *ipc, const ctp::ipc::FullPtr<TaskT> &task_ptr) {
  gpu::Future<TaskT> future;

#if CTP_IS_GPU_COMPILER
  if (threadIdx.x != 0) return future;
#endif

  if (task_ptr.IsNull() || !ipc->gpu_info_.gpu2cpu_queue) {
    return future;
  }

  // Co-located FutureShm sits immediately after the POD task struct.
  gpu::FutureShm *fshm = reinterpret_cast<gpu::FutureShm *>(
      reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
  fshm->Reset(static_cast<u32>(sizeof(TaskT)));

  // For kPinnedHost / kManagedUvm backends the host_view pointer is also
  // the device-accessible address — the kernel and the CPU worker can both
  // dereference it directly. We therefore stash the *raw address* in off_
  // and keep the AllocatorId for backend lookup. The worker pop path
  // dereferences off_ directly without per-backend resolution math. When
  // kDeviceMem is wired up the worker will branch on `kind` and issue a
  // cudaMemcpy of size sizeof(FutureShm) from device_ptr+task_off instead.
  ctp::ipc::ShmPtr<gpu::FutureShm> fshmptr;
  fshmptr.alloc_id_ = task_ptr.shm_.alloc_id_;
  fshmptr.off_ = reinterpret_cast<size_t>(fshm);

  // Mirror raw addressing for the task ShmPtr in the queue entry so the
  // CPU worker can dereference the task pointer the same way.
  ctp::ipc::FullPtr<Task> task_for_queue;
  task_for_queue.shm_.alloc_id_ = task_ptr.shm_.alloc_id_;
  task_for_queue.shm_.off_ = reinterpret_cast<size_t>(task_ptr.ptr_);
  task_for_queue.ptr_ = static_cast<Task *>(task_ptr.ptr_);

  future = gpu::Future<TaskT>(fshmptr, task_ptr);

  gpu::Future<Task> task_future(fshmptr, task_for_queue);
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
 * Polls FUTURE_COMPLETE via volatile read on the FutureShm. Backend is
 * pinned host or UVM, so the CPU's system-scope SetBitsSystem write is
 * visible to a device-side volatile read through PCIe cache snooping.
 */
template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void gpu::Future<TaskT, AllocT>::Wait() {
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
#if CTP_IS_GPU_COMPILER
  if (threadIdx.x != 0) return;
#endif
  if (future_shm_.IsNull()) return;
  gpu::FutureShm *fshm = GetFutureShmPtrRaw();
  if (!fshm) return;
  volatile unsigned int *fp =
      reinterpret_cast<volatile unsigned int *>(&fshm->flags_.bits_.x);
  while (!((*fp) & gpu::FutureShm::FUTURE_COMPLETE)) {}
  CTP_DEVICE_FENCE_SYSTEM();
#endif
}
#endif  // CTP_IS_GPU_COMPILER || CTP_IS_SYCL_COMPILER

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#endif  // CLIO_RUNTIME_INCLUDE_IPC_GPU2CPU_IMPL_H_
