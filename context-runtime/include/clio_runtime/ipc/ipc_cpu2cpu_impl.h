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

  // #642: the task's virtual address is the response key the worker echoes back
  // so this client thread can match the result to the right Future.
  size_t net_key = reinterpret_cast<size_t>(task_ptr.ptr_);
  task_ptr->task_id_.net_key_ = net_key;

  // FutureShm now lives in PRIVATE memory: the worker never touches it; the
  // result returns over this client thread's MPSC server (clio-<pid>-<tid>).
  ctp::ipc::FullPtr<char> buffer =
      CTP_MALLOC->AllocateObjs<char>(sizeof(FutureShm));
  if (buffer.IsNull()) return Future<TaskT>();
  FutureShm *future_shm = new (buffer.ptr_) FutureShm();
  future_shm->pool_id_ = task_ptr->pool_id_;
  future_shm->method_id_ = task_ptr->method_;
  future_shm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  future_shm->client_task_vaddr_ = net_key;
  // Ensure this thread's MPSC receive server exists before the response lands.
  ipc->GetTls();
  future_shm->waiter_pid_ = static_cast<u32>(ctp::SystemInfo::GetPid());
  future_shm->waiter_tid_ = static_cast<u32>(ctp::SystemInfo::GetTid());

  // Register for response matching.
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = future_shm;
  }

  auto future_shm_shmptr = buffer.shm_.template Cast<FutureShm>();
  Future<TaskT> future(future_shm_shmptr, task_ptr);

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
    HLOG(kError, "IpcCpu2Cpu::ClientSend: no MPSC worker server available");
    future_shm->flags_.SetBits(1 | FutureShm::FUTURE_COMPLETE);
    return future;
  }
  // shm_send_transport_ is only used to Expose bulk descriptors while building
  // the archive; conn->Send performs the actual MPSC transfer (metadata+data).
  SaveTaskArchive archive(MsgType::kSerializeIn,
                           ipc->shm_send_transport_.get());
  archive << (*task_ptr.ptr_);
  conn->Send(archive);
  return future;
}

template <typename TaskT>
bool IpcCpu2Cpu::ClientRecv(IpcManager *ipc,
                             Future<TaskT> &future, float max_sec) {
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
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
