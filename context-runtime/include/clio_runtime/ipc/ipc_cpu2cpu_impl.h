/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_

#include "clio_runtime/ipc/ipc_cpu2cpu.h"

namespace clio::run {

template <typename TaskT>
Future<TaskT> IpcCpu2Cpu::ClientSend(IpcManager *ipc,
                                      const ctp::ipc::FullPtr<TaskT> &task_ptr) {
  if (task_ptr.IsNull()) return Future<TaskT>();

  // Allocate FutureShm with copy_space
  size_t copy_space_size = task_ptr->GetCopySpaceSize();
  if (copy_space_size == 0) copy_space_size = KILOBYTES(4);
  size_t alloc_size = sizeof(FutureShm) + copy_space_size;
  auto buffer = ipc->AllocateBuffer(alloc_size);
  if (buffer.IsNull()) return Future<TaskT>();

  FutureShm *future_shm = new (buffer.ptr_) FutureShm();
  future_shm->pool_id_ = task_ptr->pool_id_;
  future_shm->method_id_ = task_ptr->method_;
  future_shm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  future_shm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(task_ptr.ptr_);
  future_shm->input_.copy_space_size_ = copy_space_size;
  future_shm->output_.copy_space_size_ = copy_space_size;
  future_shm->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);
  // Register this thread as the waiter so RuntimeSend can wake it via
  // EventManager::Signal. GetTls() lazily creates this thread's EventManager
  // (registering its named (pid,tid) event) BEFORE the worker could complete,
  // closing the lost-wakeup window. Recorded before the lane Push below.
  ipc->GetTls();
  future_shm->waiter_pid_ = static_cast<u32>(ctp::SystemInfo::GetPid());
  future_shm->waiter_tid_ = static_cast<u32>(ctp::SystemInfo::GetTid());

  // Create Future
  auto future_shm_shmptr = buffer.shm_.template Cast<FutureShm>();
  Future<TaskT> future(future_shm_shmptr, task_ptr);

  // Build SHM context for transfer
  ctp::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->input_;

  // Enqueue BEFORE sending (worker must start RecvMetadata concurrently)
  LaneId lane_id =
      ipc->scheduler_->ClientMapTask(ipc, future.template Cast<Task>());
  auto &lane = ipc->worker_queues_->GetLane(lane_id, 0);
  lane.Push(future.template Cast<Task>());
  // Always signal — the prior `if (was_empty)` gate let producers skip
  // SIGUSR1 whenever any other producer's task was visible in the lane,
  // assuming the worker was already processing. Under FUSE-adapter load
  // (20+ ranks pushing nearly-simultaneous GetTagSize / GetBlob futures)
  // the assumption fails: the worker is in epoll_pwait2 past its own
  // recheck, the lane shows non-empty, and nobody sends the wakeup. The
  // syscall is cheap; correctness wins.
  ipc->AwakenWorker(&lane);

  SaveTaskArchive archive(MsgType::kSerializeIn,
                           ipc->shm_send_transport_.get());
  archive << (*task_ptr.ptr_);
  ipc->shm_send_transport_->Send(archive, ctx);

  return future;
}

template <typename TaskT>
bool IpcCpu2Cpu::ClientRecv(IpcManager *ipc,
                             Future<TaskT> &future, float max_sec) {
  auto future_shm = future.GetFutureShm();
  TaskT *task_ptr = future.get();

  // Normal SHM path: block in Recv until the worker's RuntimeSend streams the
  // full output. Recv sleeps on this thread's EventManager (woken by SendOut's
  // start-of-transfer Signal) instead of busy-polling FUTURE_COMPLETE. The
  // bounded ctx.timeout_ms is the safety net: Recv returns EAGAIN if no output
  // appears in time, which we treat as a possible server death.
  ctp::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->output_;
  ctx.event_manager_ = &ipc->GetTls()->event_manager_;
  // Re-check liveness within ~1s even when the caller asked to wait forever, so
  // a dead server is detected and routed to the ZMQ reconnect path below.
  ctx.timeout_ms = (max_sec > 0) ? static_cast<int>(max_sec * 1000) : 1000;

  LoadTaskArchive archive;
  auto info = ipc->shm_recv_transport_->Recv(archive, ctx);
  while (info.rc == EAGAIN) {
    if (!ipc->server_alive_.load()) {
      HLOG(kWarning, "Recv(SHM): Server died while waiting for response");
      // Fall through to ZMQ reconnect path
      return IpcCpu2CpuZmq::ClientRecv(ipc, future, max_sec);
    }
    if (max_sec > 0) {
      HLOG(kWarning, "Recv(SHM): Timeout waiting for response");
      return false;
    }
    // max_sec == 0 (wait forever): keep waiting through liveness re-checks.
    info = ipc->shm_recv_transport_->Recv(archive, ctx);
  }

  // Deserialize outputs (Recv guarantees the full task data is present).
  archive.ResetBulkIndex();
  archive.msg_type_ = MsgType::kSerializeOut;
  archive >> (*task_ptr);
  return true;
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
