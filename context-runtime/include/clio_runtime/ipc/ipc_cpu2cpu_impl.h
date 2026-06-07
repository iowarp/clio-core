/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_IMPL_H_

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

  // Normal SHM path: server is alive, use ring buffer recv
  ctp::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->output_;

  LoadTaskArchive archive;
  ipc->shm_recv_transport_->Recv(archive, ctx);

  // Wait for FUTURE_COMPLETE, but bail if the server dies or times out
  ctp::abitfield32_t &flags = future_shm->flags_;
  auto shm_start = std::chrono::steady_clock::now();
  while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
    CTP_THREAD_MODEL->Yield();
    if (!ipc->server_alive_.load()) {
      HLOG(kWarning, "Recv(SHM): Server died while waiting for response");
      // Fall through to ZMQ reconnect path
      return IpcCpu2CpuZmq::ClientRecv(ipc, future, max_sec);
    }
    if (max_sec > 0) {
      float elapsed = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - shm_start)
                          .count();
      if (elapsed >= max_sec) {
        HLOG(kWarning, "Recv(SHM): Timeout after {:.1f}s", elapsed);
        return false;
      }
    }
  }

  // Deserialize outputs
  archive.ResetBulkIndex();
  archive.msg_type_ = MsgType::kSerializeOut;
  archive >> (*task_ptr);
  return true;
}

}  // namespace clio::run

#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_IMPL_H_
