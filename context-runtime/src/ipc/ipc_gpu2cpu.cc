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

  worker->SetCurrentTask(clio::run::shared_ptr<Task>());

  ctp::ipc::ShmPtr<Task> task_shmptr = gpu_future.GetTaskPtr().shm_;
  if (task_shmptr.IsNull()) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null task ShmPtr in queue entry",
         worker_id);
    return true;
  }

  void *gpu_task_raw = reinterpret_cast<void *>(task_shmptr.off_.load());
  if (!gpu_task_raw) {
    HLOG(kError, "IpcGpu2Cpu::RecvIn: worker {} null task off_ in queue entry",
         worker_id);
    return true;
  }

  // Detect whether the Task struct sits in pure device memory (host cannot
  // dereference it). ctp::IsDevicePointer returns false on host builds.
  bool task_on_device = ctp::IsDevicePointer(gpu_task_raw);

  // The self-contained Task carries its POD size in fut_.task_size_; the kernel
  // cached it in the queue entry so we know the copy size without reading the
  // task first.
  u32 task_pod_size = gpu_future.GetTaskSize();
  if (task_pod_size == 0) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} queue task_size_=0 — kernel did not "
         "stamp fut_.task_size_ before Send",
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

  // Signal completion directly on the (possibly device-resident) Task's
  // embedded flag so the kernel poll-loop unblocks on the error paths below.
  // is_complete_ is fut_'s first member (atomic<u32> whose storage is `.x`),
  // so its device address is gpu_task_raw + (offset of fut_.is_complete_.x).
  auto signal_task_complete = [&]() {
    if (task_on_device) {
      size_t off = reinterpret_cast<char *>(&task_raw->fut_.is_complete_.x) -
                   reinterpret_cast<char *>(task_raw);
      u32 one = 1;
      ctp::DeviceAwareMemcpy(reinterpret_cast<char *>(gpu_task_raw) + off, &one,
                             sizeof(u32));
    } else {
      task_raw->fut_.is_complete_.store(1);
    }
  };

  PoolId pool_id = task_raw->pool_id_;
  u32 method_id = task_raw->method_;

  // task_raw points into a reused worker scratch buffer (or device memory), not
  // a make_shared block — wrap it NON-OWNING so the Future frees nothing.
  Future<Task> future(pool_id, method_id,
                      clio::run::shared_ptr<Task>::WrapNonOwning(task_raw));
  if (future.GetFutureShmPtr().IsNull()) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} Future construction failed (pool={}, "
         "method={})",
         worker_id, pool_id, method_id);
    signal_task_complete();
    return true;
  }

  auto chi_fshm = future.GetFutureShm();
  chi_fshm->origin_ = ClientOrigin::kClientGpu2Cpu;
  // Stash the device-side task pointer + size so SendOut can H2D-copy the
  // mutated POD back and flip the task's completion flag (cudaMemcpy when in
  // kDeviceMem). The Task is its own completion record — no gpu::FutureShm.
  chi_fshm->gpu_task_device_ptr_ =
      task_on_device ? reinterpret_cast<uintptr_t>(gpu_task_raw) : 0;
  chi_fshm->gpu_task_size_ = task_pod_size;

  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(pool_id).get();
  if (!container) {
    HLOG(kError,
         "IpcGpu2Cpu::RecvIn: worker {} Container not found (pool={}, method={})",
         worker_id, pool_id, method_id);
    task_raw->SetReturnCode(1);
    signal_task_complete();
    return true;
  }

  // Fix up SSO/SVO `data_` pointers in the host-resident task copy if we
  // D2H-copied it. The chimod's container override dispatches by method id to
  // the per-task FixupAfterCopy(). Skip when the task never moved (kPinnedHost /
  // kManagedUvm path).
  if (task_on_device) {
    container->FixupAfterCopy(method_id, future.GetTaskPtr());
  }

  // Allocate the task's RunContext (and resolve its container) now that it is
  // deserialized, so RouteTask / the worker have an active RunContext.
  future.GetTaskPtr()->BeginRunContext();

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
clio::run::shared_ptr<Task> IpcGpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  (void)ipc; (void)method_id; (void)recv_transport;
  return future.GetTaskPtr();
}

/**
 * SendOut: writes the (mutated) POD task bytes back to the original device
 * address (when the kernel allocated in kDeviceMem) and flips the task's
 * embedded completion flag (task->fut_.is_complete_) so the kernel poll-loop
 * unblocks. There is no separate gpu::FutureShm — the Task is its own record.
 *
 * For kPinnedHost / kManagedUvm the host scratch copy IS the authoritative
 * storage (CPU and GPU share the address), so no writeback is needed and we
 * just mark complete in place. For kDeviceMem we cudaMemcpy the POD payload,
 * then a separate 4-byte cudaMemcpy of is_complete_=1 (ordered AFTER the POD,
 * so the kernel never sees completion before the outputs are written) — a
 * single aligned 32-bit write is observed whole by the device's volatile read.
 */
void IpcGpu2Cpu::SendOut(
    IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr) {
  auto future_shm = task_ptr->RunFuture().GetFutureShm();
  HLOG(kDebug, "IpcGpu2Cpu::SendOut: pool={} method={}",
       task_ptr->pool_id_, task_ptr->method_);
  Task *host_task = task_ptr.get();

  if (future_shm->gpu_task_device_ptr_ && future_shm->gpu_task_size_) {
    // kDeviceMem: writeback the mutated POD (is_complete_ still 0 here), then
    // flip the completion flag separately at its device address. is_complete_
    // is fut_'s first member (atomic<u32> whose storage is `.x`).
    ctp::DeviceAwareMemcpy(
        reinterpret_cast<void *>(future_shm->gpu_task_device_ptr_), host_task,
        future_shm->gpu_task_size_);
    size_t complete_off =
        reinterpret_cast<char *>(&host_task->fut_.is_complete_.x) -
        reinterpret_cast<char *>(host_task);
    u32 one = 1;
    ctp::DeviceAwareMemcpy(
        reinterpret_cast<char *>(future_shm->gpu_task_device_ptr_) +
            complete_off,
        &one, sizeof(u32));
  }

  // Mark complete: for kPinnedHost / kManagedUvm this storage is shared with
  // the device (the kernel's volatile poll sees it); also wakes host waiters.
  host_task->SetComplete();

  // Producer-only model: the client owns the device-memory backend that holds
  // the task — the runtime does not free it.
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  (void)ipc;
}

}  // namespace clio::run

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
