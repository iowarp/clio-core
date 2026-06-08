/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_gpu2cpu.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

#include "clio_ctp/util/gpu_api.h"
#include "clio_runtime/gpu/future.h"
#include "clio_runtime/gpu/gpu_ipc_manager.h"
#include "clio_runtime/ipc_manager.h"

namespace clio::run {

/**
 * RuntimeRecv: producer-only — the GPU never serializes a task through
 * lightbeam. Worker::ProcessNewTaskGpu already wrapped the popped task
 * pointer in a chi::Future<Task> (for kDeviceMem the wrapped pointer
 * is a host scratch copy of the device POD). We just hand it back.
 */
ctp::ipc::FullPtr<Task> IpcGpu2Cpu::RuntimeRecv(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  (void)ipc; (void)container; (void)method_id; (void)recv_transport;
  return future.GetTaskPtr();
}

/**
 * RuntimeSend: writes the (mutated) POD task bytes back to the original
 * device address (when the kernel allocated in kDeviceMem) and sets
 * FUTURE_COMPLETE on the device-side gpu::FutureShm so the kernel
 * poll-loop unblocks.
 *
 * For kPinnedHost / kManagedUvm backends the host scratch copy IS the
 * authoritative storage (CPU and GPU share the same address) so no
 * writeback is needed; we just SetBits on the host-mapped flags. For
 * kDeviceMem we issue cudaMemcpy of the POD payload + a 4-byte cudaMemcpy
 * of the flag word to flip FUTURE_COMPLETE atomically — single-aligned-
 * 32-bit writes are observed atomically by the device's volatile read.
 */
void IpcGpu2Cpu::RuntimeSend(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container) {
  (void)container;
  auto future_shm = run_ctx->future_.GetFutureShm();
  HLOG(kDebug, "IpcGpu2Cpu::RuntimeSend: pool={} method={}",
       task_ptr->pool_id_, task_ptr->method_);

  // 1) Writeback the POD task bytes to device memory if the kernel
  //    allocated the task there. Worker::ProcessNewTaskGpu set
  //    gpu_task_device_ptr_ only when D2H-copy was needed.
  if (future_shm->gpu_task_device_ptr_ && future_shm->gpu_task_size_) {
    void *dst = reinterpret_cast<void *>(future_shm->gpu_task_device_ptr_);
    ctp::DeviceAwareMemcpy(dst, task_ptr.ptr_, future_shm->gpu_task_size_);
  }

  // 2) Signal FUTURE_COMPLETE on the device-side gpu::FutureShm. For
  //    pinned host / UVM the address is dereferenceable and we use a
  //    fenced atomic OR. For kDeviceMem we cudaMemcpy a 4-byte word.
  if (future_shm->gpu_fshm_device_ptr_) {
    auto *gpu_fshm = reinterpret_cast<gpu::FutureShm *>(
        future_shm->gpu_fshm_device_ptr_);
    bool fshm_on_device =
        ctp::IsDevicePointer(static_cast<void *>(gpu_fshm));
    if (fshm_on_device) {
      // GPU's volatile read of bits_.x sees the 4-byte write whole.
      // We OR-in the bit by reading then writing rather than racing
      // against the kernel — the kernel never writes flags_ while a
      // task is in-flight (it only reads), so a plain write of
      // FUTURE_COMPLETE is safe here.
      u32 new_flags = gpu::FutureShm::FUTURE_COMPLETE;
      ctp::DeviceAwareMemcpy(&gpu_fshm->flags_.bits_.x, &new_flags,
                             sizeof(u32));
    } else {
      // Atomic, system-scope OR of the completion bit. Use the bitfield's
      // portable helper rather than __sync_fetch_and_or (a GCC builtin that
      // MSVC does not provide).
      gpu_fshm->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
    }
  }

  // Mark the chi-side future complete for any host-side waiters.
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);

  // Producer-only model: the client owns the device-memory backend that
  // holds the task — the runtime does not free it.
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  (void)ipc;
}

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
