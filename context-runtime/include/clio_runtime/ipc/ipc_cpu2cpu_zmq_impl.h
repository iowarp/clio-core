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
Future<TaskT> IpcCpu2CpuZmq::SendIn(IpcManager *ipc,
                                          const clio::run::shared_ptr<TaskT> &task_ptr,
                                          IpcMode mode) {
#if !CTP_IS_HOST
  // Host-only ZMQ client path; inert in the device pass (kernels never ZMQ-send).
  (void)ipc;
  (void)task_ptr;
  (void)mode;
  return Future<TaskT>();
#else
  if (task_ptr.IsNull()) return Future<TaskT>();

  // Set net_key for response routing
  size_t net_key = reinterpret_cast<size_t>(task_ptr.get());
  task_ptr->task_id_.net_key_ = net_key;

  // Serialize task inputs
  SaveTaskArchive archive(MsgType::kSerializeIn, ipc->zmq_transport_.get());
  // Advertise the ephemeral response-listener port (TCP mode) so the runtime
  // opens a dedicated dial-back connection for the response. 0 in IPC mode,
  // where the response returns over the same unix socket.
  archive.client_port_ = ipc->GetClientResponsePort();
  archive << (*task_ptr);

  // The Future owns a fresh FutureShm via shared_ptr (private memory).
  Future<TaskT> future(task_ptr->pool_id_, task_ptr->method_, task_ptr);
  RunContext *future_shm = future.GetFutureShm().ptr_;
  future_shm->origin_ = (mode == IpcMode::kTcp)
                            ? ClientOrigin::kClientTcp
                            : ClientOrigin::kClientIpc;
  // Register this client thread as the waiter so the async recv thread can wake
  // it via EventManager::Signal when the response lands, instead of the client
  // busy-polling. GetTls creates this thread's EventManager (its named (pid,tid)
  // event) before the response can arrive. The waiter lives on the task's
  // FutureInfo; the response routes by task_id_.net_key_ (set above).
  ipc->GetTls();
  task_ptr->SetWaiter(static_cast<u32>(ctp::SystemInfo::GetPid()),
                      static_cast<u32>(ctp::SystemInfo::GetTid()));

  // Register in pending futures map
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = {task_ptr.get()};
  }

  // Send via lightbeam PUSH client
  {
    std::lock_guard<std::mutex> lock(ipc->zmq_client_send_mutex_);
    ipc->zmq_transport_->Send(archive, ctp::lbm::LbmContext());
  }

  return future;
#endif  // CTP_IS_HOST
}

template <typename TaskT>
bool IpcCpu2CpuZmq::RecvOut(IpcManager *ipc,
                                 Future<TaskT> &future, float max_sec) {
#if !CTP_IS_HOST
  (void)ipc;
  (void)future;
  (void)max_sec;
  return false;
#else
  auto future_shm = future.GetFutureShm();
  TaskT *task_ptr = future.get();
  ClientOrigin origin = future_shm->origin_;

  // If origin was SHM but server is dead, reconnect and resend via ZMQ
  if (origin == ClientOrigin::kClientShm) {
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
  while (!task_ptr->IsComplete()) {
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
  size_t net_key = task_ptr->task_id_.net_key_;
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
#endif  // CTP_IS_HOST
}

template <typename TaskT>
void IpcCpu2CpuZmq::ResendTask(IpcManager *ipc, Future<TaskT> &future) {
#if !CTP_IS_HOST
  (void)ipc;
  (void)future;
  return;
#else
  auto future_shm = future.GetFutureShm();
  TaskT *task_ptr = future.get();
  size_t old_net_key = task_ptr->task_id_.net_key_;

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

  task_ptr->UnsetComplete();
  future_shm->origin_ = (ipc->ipc_mode_ == IpcMode::kIpc)
                            ? ClientOrigin::kClientIpc
                            : ClientOrigin::kClientTcp;

  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = {task_ptr};
  }
  {
    std::lock_guard<std::mutex> lock(ipc->zmq_client_send_mutex_);
    ipc->zmq_transport_->Send(archive, ctp::lbm::LbmContext());
  }
#endif  // CTP_IS_HOST
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_IMPL_H_
