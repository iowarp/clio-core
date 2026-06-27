/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_cpu2cpu.h"
#include "clio_runtime/ipc_manager.h"

namespace clio::run {

ctp::ipc::FullPtr<Task> IpcCpu2Cpu::RuntimeRecv(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  auto future_shm = future.GetFutureShm();
  FullPtr<Task> task_full_ptr = future.GetTaskPtr();

  // Only deserialize if task was copied from client and not yet processed
  if (!future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) ||
      future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED)) {
    return task_full_ptr;
  }

  // Build SHM context for transfer
  ctp::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->input_;

  // Detect legacy GPU->CPU tasks by client_task_vaddr_ == 0
  bool is_gpu_task = (future_shm->client_task_vaddr_ == 0);
  if (is_gpu_task) {
    clio::run::priv::vector<char> recv_buf(CLIO_PRIV_ALLOC);
    recv_buf.reserve(256);
    DefaultLoadArchive local_archive(recv_buf);
    recv_transport->Recv(local_archive, ctx);
    task_full_ptr = container->LocalAllocLoadTask(method_id, local_archive);
  } else {
    // Normal CPU->CPU SHM path: cereal serialization
    LoadTaskArchive archive;
    recv_transport->Recv(archive, ctx);
    task_full_ptr = container->AllocLoadTask(method_id, archive);
  }

  // This task came straight from an external user client (the
  // FUTURE_COPY_FROM_CLIENT guard above), so tag it for per-RPC access control.
  // The flag is serialized in SerializeIn, so it survives if the task is later
  // forwarded to a remote container owner.
  if (!task_full_ptr.IsNull()) {
    task_full_ptr->SetFlags(TASK_EXTERNAL_CLIENT);
  }

  // Update the Future's task pointer and mark as copied
  future.GetTaskPtr() = task_full_ptr;
  future_shm->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);
  return task_full_ptr;
}

void IpcCpu2Cpu::RuntimeSend(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container,
    ctp::lbm::Transport *send_transport) {
  auto future_shm = run_ctx->future_.GetFutureShm();

  // Serialize outputs into SHM ring buffer
  future_shm->output_.copy_space_size_ =
      future_shm->input_.copy_space_size_;
  ctp::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->output_;
  // Wake the client thread blocked in Recv at the start of the transfer so it
  // drains the output ring as we fill it (and returns promptly) instead of
  // busy-polling FUTURE_COMPLETE. The transport's Send issues the static
  // EventManager::Signal to this (pid, tid); 0 means no registered waiter.
  ctx.signal_pid_ = static_cast<int>(future_shm->waiter_pid_);
  ctx.signal_tid_ = static_cast<int>(future_shm->waiter_tid_);
  SaveTaskArchive archive(MsgType::kSerializeOut, send_transport);
  container->SaveTask(task_ptr->method_, archive, task_ptr);
  send_transport->Send(archive, ctx);

  // Signal completion and clean up
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  container->DelTask(task_ptr->method_, task_ptr);
}

}  // namespace clio::run
