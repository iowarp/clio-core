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
#include "clio_runtime/singletons.h"
#include "clio_runtime/worker.h"

namespace clio::run {

/**
 * RecvIn (producer-only gpu2cpu pop): pop one gpu::Future<Task> off `gpu_lane`,
 * D2H-copy the gpu::FutureShm + POD task out of device memory when the kernel
 * allocated in kDeviceMem (the CPU cannot dereference device pointers), wrap the
 * host-resident task in a clio::run::Future<Task>, stash the original device
 * pointers + size on the chi FutureShm (so SendOut can H2D-copy the mutated POD
 * back and flip FUTURE_COMPLETE), then route it. Moved here from the worker so
 * the worker never deserializes tasks/futures. Runs on the worker thread.
 */
bool IpcGpu2Cpu::RecvIn(IpcManager *ipc, GpuTaskLane *gpu_lane, Worker *worker) {
  gpu::Future<Task> gpu_future;
  if (!gpu_lane->Pop(gpu_future)) {
    return false;
  }
  const u32 worker_id = worker->GetId();
  HLOG(kDebug, "IpcGpu2Cpu::RecvIn: worker {} popped task from gpu2cpu queue",
       worker_id);

  worker->SetCurrentRunContext(nullptr);

  ctp::ipc::ShmPtr<gpu::FutureShm> gpu_fshm_shmptr = gpu_future.GetFutureShmPtr();
  ctp::ipc::ShmPtr<Task> task_shmptr = gpu_future.GetTaskPtr().shm_;
  if (gpu_fshm_shmptr.IsNull() || task_shmptr.IsNull()) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null ShmPtr in queue entry",
         worker_id);
    return true;
  }

  void *gpu_fshm_raw = reinterpret_cast<void *>(gpu_fshm_shmptr.off_.load());
  void *gpu_task_raw = reinterpret_cast<void *>(task_shmptr.off_.load());
  if (!gpu_fshm_raw || !gpu_task_raw) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null off_ in queue entry",
         worker_id);
    return true;
  }

  // Detect whether the FutureShm / Task structs sit in pure device memory (host
  // cannot dereference them). ctp::IsDevicePointer returns false on host builds.
  bool fshm_on_device = ctp::IsDevicePointer(gpu_fshm_raw);
  bool task_on_device = ctp::IsDevicePointer(gpu_task_raw);

  // Pull gpu::FutureShm contents into a local copy (D2H if needed). task_size_
  // tells us how many bytes the Task POD occupies.
  alignas(8) char fshm_buf[sizeof(gpu::FutureShm)];
  if (fshm_on_device) {
    ctp::DeviceAwareMemcpy(fshm_buf, gpu_fshm_raw, sizeof(gpu::FutureShm));
  } else {
    std::memcpy(fshm_buf, gpu_fshm_raw, sizeof(gpu::FutureShm));
  }
  auto &fshm_copy = *reinterpret_cast<gpu::FutureShm *>(fshm_buf);
  u32 task_pod_size = fshm_copy.task_size_;
  if (task_pod_size == 0) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} gpu::FutureShm.task_size_=0 — kernel "
         "did not call Reset(sizeof(TaskT)) before Send",
         worker_id);
    return true;
  }

  // Per-thread scratch for the host-resident task copy. Sized to fit any
  // reasonable POD task (PutBlobTask is ~480 bytes today).
  static constexpr size_t kTaskScratchBytes = 4096;
  alignas(64) thread_local char task_scratch[kTaskScratchBytes];
  if (task_pod_size > kTaskScratchBytes) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} task_pod_size {} exceeds scratch "
         "capacity {}",
         worker_id, task_pod_size, kTaskScratchBytes);
    return true;
  }
  Task *task_raw = nullptr;
  if (task_on_device) {
    ctp::DeviceAwareMemcpy(task_scratch, gpu_task_raw, task_pod_size);
    task_raw = reinterpret_cast<Task *>(task_scratch);
  } else {
    task_raw = static_cast<Task *>(gpu_task_raw);
  }

  PoolId pool_id = task_raw->pool_id_;
  u32 method_id = task_raw->method_;

  ctp::ipc::FullPtr<Task> task_full_ptr(task_raw);

  Future<Task> future = ipc->MakePointerFuture(task_full_ptr);
  if (future.GetFutureShmPtr().IsNull()) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} MakePointerFuture failed (pool={}, "
         "method={})",
         worker_id, pool_id, method_id);
    if (!fshm_on_device) {
      static_cast<gpu::FutureShm *>(gpu_fshm_raw)
          ->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
    }
    return true;
  }

  auto chi_fshm = future.GetFutureShm();
  chi_fshm->pool_id_ = pool_id;
  chi_fshm->method_id_ = method_id;
  chi_fshm->origin_ = FutureShm::FUTURE_CLIENT_GPU2CPU;
  // Stash original device-side pointers + size so SendOut can H2D-copy the
  // mutated POD back and signal FUTURE_COMPLETE on the device-side gpu::FutureShm
  // (cudaMemcpy when in kDeviceMem).
  chi_fshm->gpu_fshm_device_ptr_ = reinterpret_cast<uintptr_t>(gpu_fshm_raw);
  chi_fshm->gpu_task_device_ptr_ =
      task_on_device ? reinterpret_cast<uintptr_t>(gpu_task_raw) : 0;
  chi_fshm->gpu_task_size_ = task_pod_size;

  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *container = pool_manager->GetStaticContainer(pool_id);
  if (!container) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} Container not found (pool={}, method={})",
         worker_id, pool_id, method_id);
    chi_fshm->flags_.SetBits(1 | FutureShm::FUTURE_COMPLETE);
    if (!fshm_on_device) {
      static_cast<gpu::FutureShm *>(gpu_fshm_raw)
          ->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
    } else {
      // Best-effort: still flip the device flag via cudaMemcpy.
      u32 v = gpu::FutureShm::FUTURE_COMPLETE;
      ctp::DeviceAwareMemcpy(
          &static_cast<gpu::FutureShm *>(gpu_fshm_raw)->flags_.bits_.x, &v,
          sizeof(u32));
    }
    return true;
  }

  // Fix up SSO/SVO `data_` pointers in the host-resident task copy if we
  // D2H-copied it. The chimod's container override dispatches by method id to
  // the per-task FixupAfterCopy(). Skip when the task never moved (kPinnedHost /
  // kManagedUvm path).
  if (task_on_device) {
    container->FixupAfterCopy(method_id, task_full_ptr);
  }

  if (!task_full_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
    ipc->BeginTask(future, container, worker->GetLane());
  } else {
    RunContext *run_ctx = task_full_ptr->GetRunCtx();
    if (run_ctx) {
      run_ctx->worker_id_ = worker_id;
      run_ctx->lane_ = worker->GetLane();
      run_ctx->event_queue_ = worker->GetEventQueue();
    }
  }

  RouteResult route_result = ipc->RouteTask(future, /*force_enqueue=*/true);
  HLOG(kDebug,
       "IpcGpu2Cpu::RecvIn: worker {} RouteTask returned {} pool={} method={}",
       worker_id, (int)route_result, pool_id, method_id);
  return true;
}

/**
 * RecvIn (legacy copy-space overload): producer-only — the GPU never serializes
 * a task through lightbeam, and the gpu2cpu-pop RecvIn above already wrapped the
 * popped task pointer in a clio::run::Future<Task>. We just hand it back.
 */
ctp::ipc::FullPtr<Task> IpcGpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  (void)ipc; (void)container; (void)method_id; (void)recv_transport;
  return future.GetTaskPtr();
}

/**
 * SendOut: writes the (mutated) POD task bytes back to the original
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
void IpcGpu2Cpu::SendOut(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container) {
  (void)container;
  auto future_shm = run_ctx->future_.GetFutureShm();
  HLOG(kDebug, "IpcGpu2Cpu::SendOut: pool={} method={}",
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
