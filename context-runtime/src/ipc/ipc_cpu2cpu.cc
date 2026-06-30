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
  auto container = CLIO_POOL_MANAGER->GetStaticContainer(ti.pool_id_).get();
  if (container == nullptr) {
    return false;
  }
  clio::run::shared_ptr<clio::run::Task> tp =
      container->AllocLoadTask(ti.method_id_, archive);
  if (tp.IsNull()) {
    return false;
  }
  tp->SetFlags(TASK_EXTERNAL_CLIENT);
  // The Future owns the FutureShm via shared_ptr; pushing it onto the lane
  // copies the Future, so the FutureShm stays alive until the worker (and its
  // RunContext copy) is done with it.
  // The Future ctor ensures tp's RunContext exists (its embedded route_ holds
  // the routing state); BeginRunContext below reuses it (idempotent).
  Future<Task> f(ti.pool_id_, ti.method_id_, tp);
  auto fs = f.GetFutureShm();
  fs->origin_ = ClientOrigin::kClientShm;
  fs->client_pid_ = ti.task_id_.pid_;
  // The SHM client blocks on its own MPSC server clio-<pid>-<tid>; SendOut routes
  // the result back there using the waiter (the OS pid/tid stamped by SendIn,
  // carried in task_id_). net_key (task_id_.net_key_) is already on the task.
  tp->SetWaiter(ti.task_id_.pid_, ti.task_id_.tid_);
  // Allocate the task's RunContext (and resolve its container) now that it is
  // deserialized, so RouteTask / the worker have an active RunContext.
  f.GetTaskPtr()->BeginRunContext();
  lane->Push(f);
  return true;
}

clio::run::shared_ptr<clio::run::Task> IpcCpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  // The inbound MPSC SHM drain (the RecvIn(ipc, lane) overload above) already
  // deserialized the task from the client's serialized bytes and stamped the
  // Future's task pointer before enqueuing it. There is no inline copy_space to
  // deserialize from here, so just return the already-resolved task pointer.
  (void)ipc;
  (void)method_id;
  (void)recv_transport;
  return future.GetTaskPtr();
}

void IpcCpu2Cpu::SendOut(
    IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr,
    ctp::lbm::Transport *send_transport) {
  clio::run::ContainerHold container =
      CLIO_POOL_MANAGER->GetStaticContainer(task_ptr->pool_id_).get();
  auto future_shm = task_ptr->RunFuture().GetFutureShm();

  // #642: serialize the result and high-level Send it to the originating client
  // thread's MPSC server ("clio-<client_pid>-<client_tid>"). send_transport is
  // used only to Expose bulk descriptors while building the archive; conn->Send
  // performs the actual MPSC transfer (metadata + data).
  std::string name = "clio-" + std::to_string(task_ptr->WaiterPid()) + "-" +
                     std::to_string(task_ptr->WaiterTid());
  ctp::lbm::ShmMpscTransport *conn = ipc->GetOrCreateShmConn(name);
  if (conn != nullptr) {
    SaveTaskArchive archive(MsgType::kSerializeOut, send_transport);
    // SaveTask takes a non-const shared_ptr&; copy the (const) handle.
    clio::run::shared_ptr<clio::run::Task> save_handle = task_ptr;
    container->SaveTask(save_handle->method_, archive, save_handle);
    conn->Send(archive);
  } else {
    HLOG(kError, "IpcCpu2Cpu::SendOut: no client server '{}'", name);
  }

  // Signal completion (per-process: the client's own task is woken via the MPSC
  // response above). The task frees via RAII when the owning shared_ptr (held by
  // the worker's RunContext/Future) drops — no explicit DelTask.
  task_ptr->SetComplete();
  task_ptr->ClearFlags(TASK_DATA_OWNER);
}

}  // namespace clio::run
