/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_IMPL_H_

#include "clio_runtime/ipc/ipc_cpu2cpu_zmq.h"

namespace clio::run {

template <typename TaskT>
Future<TaskT> IpcCpu2CpuZmq::ClientSend(IpcManager *ipc,
                                          const ctp::ipc::FullPtr<TaskT> &task_ptr,
                                          IpcMode mode) {
  if (task_ptr.IsNull()) return Future<TaskT>();

  // Set net_key for response routing
  size_t net_key = reinterpret_cast<size_t>(task_ptr.ptr_);
  task_ptr->task_id_.net_key_ = net_key;

  // Serialize task inputs
  SaveTaskArchive archive(MsgType::kSerializeIn, ipc->zmq_transport_.get());
  // Advertise the ephemeral response-listener port (TCP mode) so the runtime
  // opens a dedicated dial-back connection for the response. 0 in IPC mode,
  // where the response returns over the same unix socket.
  archive.client_port_ = ipc->GetClientResponsePort();
  archive << (*task_ptr.ptr_);

  // Allocate FutureShm via CTP_MALLOC (no copy_space needed)
  size_t alloc_size = sizeof(FutureShm);
  ctp::ipc::FullPtr<char> buffer = CTP_MALLOC->AllocateObjs<char>(alloc_size);
  if (buffer.IsNull()) {
    HLOG(kError, "SendZmq: Failed to allocate FutureShm ({} bytes)",
         alloc_size);
    return Future<TaskT>();
  }
  FutureShm *future_shm = new (buffer.ptr_) FutureShm();
  future_shm->pool_id_ = task_ptr->pool_id_;
  future_shm->method_id_ = task_ptr->method_;
  future_shm->origin_ = (mode == IpcMode::kTcp)
                            ? FutureShm::FUTURE_CLIENT_TCP
                            : FutureShm::FUTURE_CLIENT_IPC;
  future_shm->client_task_vaddr_ = net_key;
  // Register this client thread as the waiter so the async recv thread can wake
  // it via EventManager::Signal when the response lands, instead of the client
  // busy-polling FUTURE_COMPLETE. GetTls creates this thread's EventManager
  // (its named (pid,tid) event) before the response can arrive.
  ipc->GetTls();
  future_shm->waiter_pid_ = static_cast<u32>(ctp::SystemInfo::GetPid());
  future_shm->waiter_tid_ = static_cast<u32>(ctp::SystemInfo::GetTid());

  // Register in pending futures map
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = future_shm;
  }

  // Send via lightbeam PUSH client
  {
    std::lock_guard<std::mutex> lock(ipc->zmq_client_send_mutex_);
    ipc->zmq_transport_->Send(archive, ctp::lbm::LbmContext());
  }

  ctp::ipc::ShmPtr<FutureShm> future_shm_shmptr =
      buffer.shm_.template Cast<FutureShm>();
  return Future<TaskT>(future_shm_shmptr, task_ptr);
}

template <typename TaskT>
bool IpcCpu2CpuZmq::ClientRecv(IpcManager *ipc,
                                 Future<TaskT> &future, float max_sec) {
  auto future_shm = future.GetFutureShm();
  TaskT *task_ptr = future.get();
  u32 origin = future_shm->origin_;

  // If origin was SHM but server is dead, reconnect and resend via ZMQ
  if (origin == FutureShm::FUTURE_CLIENT_SHM) {
    if (ipc->client_retry_timeout_ == 0 && ipc->client_try_new_servers_ <= 0) {
      HLOG(kError,
           "Recv(SHM): Server dead, no retry/failover configured, failing");
      return false;
    }
    HLOG(kWarning, "Recv(SHM): Server dead, attempting reconnect...");
    auto start = std::chrono::steady_clock::now();
    if (!ipc->WaitForServerAndReconnect(start)) return false;
    ResendTask(ipc, future);
    future_shm = future.GetFutureShm();
  }

  // ZMQ wait loop: sleep on this thread's EventManager until the async recv
  // thread sets FUTURE_COMPLETE and signals us. The bounded Wait re-checks
  // FUTURE_COMPLETE / server liveness / timeout if a signal is missed, and the
  // named auto-reset event latches a signal that races the Wait.
  ctp::lbm::EventManager *em = &ipc->GetTls()->event_manager_;
  auto start = std::chrono::steady_clock::now();
  while (!future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE)) {
    em->Wait(100);  // 100us bounded re-check; woken immediately by Signal
    float elapsed =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
            .count();
    if (max_sec > 0 && elapsed >= max_sec) return false;
    if (!ipc->server_alive_.load() && !ipc->reconnecting_.load()) {
      if (ipc->client_retry_timeout_ == 0 &&
          ipc->client_try_new_servers_ <= 0) {
        HLOG(kError, "Recv: Server dead, no failover configured, failing");
        return false;
      }
      HLOG(kWarning, "Recv: Server unreachable, reconnecting...");
      if (!ipc->WaitForServerAndReconnect(start)) return false;
      ResendTask(ipc, future);
      future_shm = future.GetFutureShm();
      start = std::chrono::steady_clock::now();
      continue;
    }
  }

  // Memory fence + deserialize from pending_response_archives_
  std::atomic_thread_fence(std::memory_order_acquire);
  size_t net_key = future_shm->client_task_vaddr_;
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    auto it = ipc->pending_response_archives_.find(net_key);
    if (it != ipc->pending_response_archives_.end()) {
      LoadTaskArchive *archive = it->second.get();
      archive->ResetBulkIndex();
      archive->msg_type_ = MsgType::kSerializeOut;
      *archive >> (*task_ptr);
    }
  }
  return true;
}

template <typename TaskT>
void IpcCpu2CpuZmq::ResendTask(IpcManager *ipc, Future<TaskT> &future) {
  auto future_shm = future.GetFutureShm();
  TaskT *task_ptr = future.get();
  size_t old_net_key = future_shm->client_task_vaddr_;

  // Remove old pending entries
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_.erase(old_net_key);
    auto it = ipc->pending_response_archives_.find(old_net_key);
    if (it != ipc->pending_response_archives_.end()) {
      ipc->zmq_transport_->ClearRecvHandles(*(it->second));
      ipc->pending_response_archives_.erase(it);
    }
  }

  size_t net_key = reinterpret_cast<size_t>(task_ptr);
  task_ptr->task_id_.net_key_ = net_key;

  SaveTaskArchive archive(MsgType::kSerializeIn, ipc->zmq_transport_.get());
  archive.client_port_ = ipc->GetClientResponsePort();
  archive << (*task_ptr);

  future_shm->flags_.UnsetBits(FutureShm::FUTURE_COMPLETE);
  future_shm->origin_ = (ipc->ipc_mode_ == IpcMode::kIpc)
                            ? FutureShm::FUTURE_CLIENT_IPC
                            : FutureShm::FUTURE_CLIENT_TCP;
  future_shm->client_task_vaddr_ = net_key;

  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = future_shm.ptr_;
  }
  {
    std::lock_guard<std::mutex> lock(ipc->zmq_client_send_mutex_);
    ipc->zmq_transport_->Send(archive, ctp::lbm::LbmContext());
  }
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_IMPL_H_
