/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_cpu2cpu.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"

namespace clio::run {

bool IpcCpu2Cpu::RecvIn(IpcManager *ipc, TaskLane *lane) {
  // #642: drain this worker's MPSC SHM server (DONTWAIT) and enqueue any
  // received external-client task onto the normal dispatch path. The client
  // already addressed this worker, so the deserialize happens here (the work
  // that used to funnel through one worker); routing/execution then follow the
  // standard ProcessNewTask flow. Keeping this off the worker means the worker
  // never touches serialized task/future bytes.
  IpcManagerTls *tls = ipc->GetTls();
  if (!tls->shm_server_ok_) {
    return false;
  }
  LoadTaskArchive archive;
  ctp::lbm::ClientInfo info =
      tls->shm_server_.Recv(archive, ctp::lbm::SHM_MPSC_DONTWAIT);
  if (info.rc != 0) {
    return false;
  }
  const auto &tis = archive.GetTaskInfos();
  if (tis.empty()) {
    return false;
  }
  const auto &ti = tis[0];
  Container *container = CLIO_POOL_MANAGER->GetStaticContainer(ti.pool_id_);
  if (container == nullptr) {
    return false;
  }
  ctp::ipc::FullPtr<Task> tp = container->AllocLoadTask(ti.method_id_, archive);
  if (tp.IsNull()) {
    return false;
  }
  tp->SetFlags(TASK_EXTERNAL_CLIENT);
  // The Future owns the FutureShm via shared_ptr; pushing it onto the lane
  // copies the Future, so the FutureShm stays alive until the worker (and its
  // RunContext copy) is done with it.
  Future<Task> f(ti.pool_id_, ti.method_id_, tp);
  auto fs = f.GetFutureShm();
  fs->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  fs->client_task_vaddr_ = ti.task_id_.net_key_;
  fs->client_pid_ = ti.task_id_.pid_;
  // The SHM client blocks on its own MPSC server clio-<pid>-<tid>; SendOut
  // routes the result back there using these (the OS tid stamped by SendIn).
  fs->waiter_pid_ = ti.task_id_.pid_;
  fs->waiter_tid_ = ti.task_id_.tid_;
  fs->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);
  // Allocate the task's RunContext now that it is deserialized, before it is
  // visible on the lane (the worker no longer does this).
  ipc->BeginTask(f, lane);
  lane->Push(f);
  return true;
}

ctp::ipc::FullPtr<Task> IpcCpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  // The inbound MPSC SHM drain (the RecvIn(ipc, lane) overload above) already
  // deserialized the task from the client's serialized bytes and stamped the
  // Future's task pointer before enqueuing it. There is no inline copy_space to
  // deserialize from here, so just return the already-resolved task pointer.
  (void)ipc;
  (void)container;
  (void)method_id;
  (void)recv_transport;
  return future.GetTaskPtr();
}

void IpcCpu2Cpu::SendOut(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container,
    ctp::lbm::Transport *send_transport) {
  auto future_shm = run_ctx->future_.GetFutureShm();

  // #642: serialize the result and high-level Send it to the originating client
  // thread's MPSC server ("clio-<client_pid>-<client_tid>"). send_transport is
  // used only to Expose bulk descriptors while building the archive; conn->Send
  // performs the actual MPSC transfer (metadata + data).
  std::string name = "clio-" + std::to_string(future_shm->waiter_pid_) + "-" +
                     std::to_string(future_shm->waiter_tid_);
  ctp::lbm::ShmMpscTransport *conn = ipc->GetOrCreateShmConn(name);
  if (conn != nullptr) {
    SaveTaskArchive archive(MsgType::kSerializeOut, send_transport);
    container->SaveTask(task_ptr->method_, archive, task_ptr);
    conn->Send(archive);
  } else {
    HLOG(kError, "IpcCpu2Cpu::SendOut: no client server '{}'", name);
  }

  // Signal completion and clean up
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  container->DelTask(task_ptr->method_, task_ptr);
}

}  // namespace clio::run
