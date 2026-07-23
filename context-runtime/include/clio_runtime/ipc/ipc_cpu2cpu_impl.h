/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_

#include "clio_runtime/ipc/ipc_cpu2cpu.h"

#include <unordered_map>

namespace clio::run {

// NOTE (SHM refactor): the thread-local ShmOutResponseStash that used to demux
// sibling responses off a per-thread ring is gone. Responses now arrive on the
// single per-process ring and are demuxed centrally by RecvShmClientThread,
// which parks each archive in IpcManager::pending_response_archives_ keyed by
// net_key and wakes the owning waiter — exactly like the ZMQ recv thread. Each
// RecvOut simply blocks on its EventManager and claims its own archive.

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

  // issue #807: shard across the runtime's S inbound rings. Key by this client
  // thread's tid so a given thread always targets the same ring — preserves
  // per-thread request order and keeps the ring's producer set stable (locality).
  // Each shard ring has its own dedicated drain thread on the runtime, so this
  // spreads both the MPSC-tail contention and the deserialize+route work.
  u32 shards = ipc->shm_in_shards_ >= 1 ? ipc->shm_in_shards_ : 1;
  u32 shard = static_cast<u32>(ctp::SystemInfo::GetTid()) % shards;
  ctp::lbm::ShmMpscTransport *conn = ipc->GetOrCreateShmConn(
      "clio-" + std::to_string(ipc->runtime_pid_) + "-shm-in-" +
      std::to_string(shard));
  if (conn == nullptr) {
    HLOG(kError, "IpcCpu2Cpu::SendIn: inbound SHM ring unavailable");
    task_ptr->SetComplete();  // unblock the waiter on the error path
    return future;
  }
  // shm_send_transport_ is only used to Expose bulk descriptors while building
  // the archive; conn->Send performs the actual MPSC transfer (metadata+data).
  SaveTaskArchive archive(MsgType::kSerializeIn,
                           ipc->shm_send_transport_.get());
  archive << (*task_ptr);
  int send_rc = conn->Send(archive);
  if (send_rc != 0) {
    // A submit that never reached the daemon must FAIL the future, not hang
    // it (issue #774: every silent drop on this path turns into a client
    // parked forever in Future::Wait).
    HLOG(kError, "IpcCpu2Cpu::SendIn: MPSC send failed rc={} for task {}",
         send_rc, task_ptr->task_id_);
    task_ptr->SetReturnCode(static_cast<clio::run::u32>(-send_rc));
    task_ptr->SetComplete();
  }
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
  const size_t want_key = task_ptr->task_id_.net_key_;

  // Block on this thread's EventManager until RecvShmClientThread marks this
  // task complete and signals us (SHM analogue of IpcCpu2CpuZmq::RecvOut). The
  // dedicated recv thread — not this thread — drains the single response ring,
  // so app threads no longer poll a per-thread ring or demux siblings here. The
  // 100us bounded Wait is a missed-signal / timeout safety re-check; the named
  // auto-reset event latches a Signal that races the Wait.
  ctp::lbm::EventManager *em = &ipc->GetTls()->event_manager_;
  ctp::Timepoint start;
  start.Now();

  // issue #807/#784: spin-poll for the response before parking. On SHM the
  // round-trip is microseconds; a brief spin catches the completion (set by
  // RecvShmClientThread when it demuxes our archive) without the park/wake +
  // signal syscall that otherwise dominates low-latency round-trips. Bounded by
  // config so a slow/absent response still falls through to the parked Wait.
  const double spin_us =
      static_cast<double>(CLIO_CONFIG_MANAGER->GetShmClientSpinUs());
  if (spin_us > 0.0 && !task_ptr->IsComplete()) {
    ctp::Timepoint spin_now;
    do {
      // issue #807: DRAIN INLINE while spinning instead of waiting for the
      // dedicated RecvShmClientThread to demux our response and signal us. The
      // try-lock inside makes us the sole consumer when we win it (no concurrent
      // Recv with the fallback thread); if another drainer holds it we just poll
      // our own completion below. Draining here also demuxes other waiters'
      // responses, so one active waiter serves the whole process — and it cuts
      // the entire RecvShmClientThread wake+demux hop off the response path,
      // which the split-timing showed dominates the round-trip.
      ipc->DrainShmResponses();
      if (task_ptr->IsComplete()) break;
      spin_now.Now();
    } while (start.GetUsecFromStart(spin_now) < spin_us);
  }

  while (!task_ptr->IsComplete()) {
    em->Wait(100);
    if (max_sec > 0) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= static_cast<double>(max_sec) * 1e6) {
        return false;
      }
    }
  }

  // Claim our parked response archive and deserialize into the task. Moving it
  // out and letting it destruct here matches the old stack-archive freeing:
  // output buffers are adopted into the task (TASK_DATA_OWNER) and the archive
  // frees only the wire bytes. Erasing the entry means Future::~Future's
  // CleanupResponseArchive is a no-op on this (consumed) path.
  std::atomic_thread_fence(std::memory_order_acquire);
  std::unique_ptr<LoadTaskArchive> archive;
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    auto it = ipc->pending_response_archives_.find(want_key);
    if (it != ipc->pending_response_archives_.end()) {
      archive = std::move(it->second);
      ipc->pending_response_archives_.erase(it);
    }
  }
  if (archive) {
    archive->ResetBulkIndex();
    archive->msg_type_ = MsgType::kSerializeOut;
    *archive >> (*task_ptr);
  }
  return true;
#endif  // CTP_IS_HOST
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
