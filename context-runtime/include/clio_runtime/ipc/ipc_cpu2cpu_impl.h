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
Future<TaskT> IpcCpu2Cpu::SendIn(IpcManager *ipc,
                                      const clio::run::shared_ptr<TaskT> &task_ptr) {
#if !CTP_IS_HOST
  // Host-only SHM client path (MPSC server, SystemInfo, mutex). Never invoked
  // from device kernels; provide an inert device definition so the template
  // compiles in the GPU device pass.
  (void)ipc;
  (void)task_ptr;
  return Future<TaskT>();
#else
  if (task_ptr.IsNull()) return Future<TaskT>();

  // #642: the task's virtual address is the response key the worker echoes back
  // so this client thread can match the result to the right Future.
  size_t net_key = reinterpret_cast<size_t>(task_ptr.get());
  task_ptr->task_id_.net_key_ = net_key;

  // The worker routes the result to "clio-<task_id_.pid_>-<task_id_.tid_>", which
  // MUST equal this client thread's MPSC server name. IpcManagerTls names that
  // server with ctp::SystemInfo::GetPid()/GetTid() (the OS tid), but CreateTaskId
  // stamps task_id_.tid_ from the thread model's *logical* id (PthreadModel hands
  // out a TLS counter, not the OS tid) — so without this the response is addressed
  // to a non-existent segment and the client hangs forever. Stamp the routing
  // identity from the same SystemInfo source the server is named with.
  task_ptr->task_id_.pid_ = static_cast<u32>(ctp::SystemInfo::GetPid());
  task_ptr->task_id_.tid_ = static_cast<u32>(ctp::SystemInfo::GetTid());

  // FutureShm now lives in PRIVATE memory owned by the Future's shared_ptr: the
  // worker never touches it; the result returns over this client thread's MPSC
  // server (clio-<pid>-<tid>).
  Future<TaskT> future(task_ptr->pool_id_, task_ptr->method_, task_ptr);
  RunContext *future_shm = future.GetFutureShm().ptr_;
  future_shm->origin_ = ClientOrigin::kClientShm;
  // Ensure this thread's MPSC receive server exists before the response lands.
  ipc->GetTls();
  // The waiter (this client thread) lives on the task's FutureInfo; the response
  // routes by task_id_.net_key_ (set above).
  task_ptr->SetWaiter(static_cast<u32>(ctp::SystemInfo::GetPid()),
                      static_cast<u32>(ctp::SystemInfo::GetTid()));

  // Register for response matching. The raw pointer stays valid as long as the
  // returned Future (or a copy) is alive — the client holds it until Recv.
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = {task_ptr.get()};
  }

  // Pick a worker and high-level Send the task to its server. The worker tid
  // comes from ClientConnect; the runtime pid keys the segment name.
  ctp::lbm::ShmMpscTransport *conn = nullptr;
  if (!ipc->worker_tids_.empty()) {
    u32 wtid = ipc->worker_tids_[net_key % ipc->worker_tids_.size()];
    conn = ipc->GetOrCreateShmConn(
        "clio-" + std::to_string(ipc->runtime_pid_) + "-" +
        std::to_string(wtid));
  }
  if (conn == nullptr) {
    HLOG(kError, "IpcCpu2Cpu::SendIn: no MPSC worker server available");
    task_ptr->SetComplete();  // unblock the waiter on the error path
    return future;
  }
  // shm_send_transport_ is only used to Expose bulk descriptors while building
  // the archive; conn->Send performs the actual MPSC transfer (metadata+data).
  SaveTaskArchive archive(MsgType::kSerializeIn,
                           ipc->shm_send_transport_.get());
  archive << (*task_ptr);
  conn->Send(archive);
  return future;
#endif  // CTP_IS_HOST
}

template <typename TaskT>
bool IpcCpu2Cpu::RecvOut(IpcManager *ipc,
                             Future<TaskT> &future, float max_sec) {
#if !CTP_IS_HOST
  (void)ipc;
  (void)future;
  (void)max_sec;
  return false;
#else
  TaskT *task_ptr = future.get();
  auto *tls = ipc->GetTls();

  // This thread's MPSC server only receives results for tasks this thread sent
  // (the worker routes responses to clio-<this_pid>-<this_tid>). For the common
  // one-outstanding-per-thread case the next result IS ours; deserialize it.
  // (Per-net_key demux for concurrent async sends is a later refinement.)
  ctp::Timepoint start;
  start.Now();
  while (true) {
    LoadTaskArchive archive;
    ctp::lbm::ClientInfo info =
        tls->shm_server_.Recv(archive, ctp::lbm::SHM_MPSC_DONTWAIT);
    if (info.rc == 0) {
      archive.ResetBulkIndex();
      archive.msg_type_ = MsgType::kSerializeOut;
      archive >> (*task_ptr);
      return true;
    }
    if (max_sec > 0) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= static_cast<double>(max_sec) * 1e6) {
        return false;
      }
    }
    CTP_THREAD_MODEL->Yield();
  }
#endif  // CTP_IS_HOST
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
