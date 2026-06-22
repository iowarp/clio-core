/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_run2fallback.h"

#include <chrono>
#include <memory>
#include <thread>

#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/scheduler/scheduler.h"
#include "clio_runtime/task_archives.h"

namespace clio::run {

bool IpcRun2Fallback::SendIn(IpcManager *ipc, Future<Task> &future) {
  if (ipc == nullptr) {
    return false;
  }
  // The fallback connection is a nested IpcManager acting as an SHM client of
  // the main runtime. Null = standalone runtime, nothing to punt to.
  IpcManager *fb = ipc->fallback_.get();
  if (fb == nullptr) {
    return false;
  }

  auto future_shm = future.GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  // Loop guard: never re-punt. If the main runtime also lacks the pool it must
  // fail the task locally rather than bounce it back.
  if (future_shm->flags_.Any(FutureShm::FUTURE_PUNTED)) {
    return false;
  }
  future_shm->flags_.SetBits(FutureShm::FUTURE_PUNTED);

  // Enqueue the SAME Future onto the MAIN runtime's worker lane:
  //  - fb->scheduler_ maps it to one of the main runtime's lanes,
  //  - fb->worker_queues_ is the main runtime's queue (attached during the
  //    fallback client's ClientInit),
  //  - fb->AwakenWorker signals the main runtime's worker (fb->runtime_pid_ is
  //    the main runtime's pid), exactly as a normal SHM client would.
  // The FutureShm + serialized task live in the client's shared data segment,
  // which the main runtime registered via the dual RegisterMemory path, so the
  // main runtime deserializes, runs, and completes the FutureShm in place. The
  // client polling that FutureShm observes the result directly.
  LaneId lane_id = fb->scheduler_->ClientMapTask(fb, future);
  auto &lane = fb->worker_queues_->GetLane(lane_id, 0);
  lane.Push(future);
  fb->AwakenWorker(&lane);
  return true;
}

bool IpcRun2Fallback::RelayToFallback(IpcManager *ipc, Container *container,
                                      const ctp::ipc::FullPtr<Task> &task,
                                      Future<Task> &future) {
  if (ipc == nullptr || container == nullptr || task.IsNull()) {
    HLOG(kError, "RelayToFallback: null ipc/container/task");
    return false;
  }
  IpcManager *fb = ipc->fallback_.get();
  if (fb == nullptr) {
    HLOG(kError, "RelayToFallback: no fallback_ on this runtime");
    return false;
  }
  auto future_shm = future.GetFutureShm();
  if (future_shm.IsNull()) {
    HLOG(kError, "RelayToFallback: null FutureShm");
    return false;
  }
  if (future_shm->flags_.Any(FutureShm::FUTURE_PUNTED)) {
    HLOG(kError, "RelayToFallback: already punted");
    return false;
  }
  future_shm->flags_.SetBits(FutureShm::FUTURE_PUNTED);

  const u32 method = task->method_;
  // net_key correlates the reply with this task on our DEALER (same scheme as
  // the normal ZMQ client path).
  const size_t net_key = reinterpret_cast<size_t>(task.ptr_);
  task->task_id_.net_key_ = net_key;

  // Serialize the task (inputs + bulk data) against the local external-stub
  // container, send it to main as a client call over the fallback DEALER, then
  // synchronously poll for the serialized outputs and load them back into the
  // task. No async recv thread (see FallbackClientInit). Retry the send every
  // ~2s within the overall budget to cover the ZMTP handshake window.
  constexpr float kTotalTimeout = 30.0f;
  auto start = std::chrono::steady_clock::now();
  bool got = false;
  while (std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
             .count() < kTotalTimeout) {
    {
      std::lock_guard<std::mutex> lock(fb->zmq_client_send_mutex_);
      SaveTaskArchive in_ar(MsgType::kSerializeIn, fb->zmq_transport_.get());
      container->SaveTask(method, in_ar, task);
      fb->zmq_transport_->Send(in_ar, ctp::lbm::LbmContext());
      HLOG(kDebug, "RelayToFallback: sent pool={} method={} net_key={} to main",
           task->pool_id_, method, net_key);

      auto attempt_start = std::chrono::steady_clock::now();
      while (std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                          attempt_start)
                 .count() < 2.0f) {
        auto out_ar = std::make_unique<LoadTaskArchive>();
        auto info = fb->zmq_transport_->Recv(*out_ar);
        if (info.rc == EAGAIN) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        if (info.rc != 0) {
          fb->zmq_transport_->ClearRecvHandles(*out_ar);
          break;  // re-send
        }
        size_t got_key = out_ar->task_infos_.empty()
                             ? 0
                             : out_ar->task_infos_[0].task_id_.net_key_;
        HLOG(kDebug, "RelayToFallback: recv reply net_key={} (want {})", got_key,
             net_key);
        if (out_ar->task_infos_.empty() || got_key != net_key) {
          fb->zmq_transport_->ClearRecvHandles(*out_ar);
          continue;  // not our reply
        }
        out_ar->msg_type_ = MsgType::kSerializeOut;
        container->LoadTask(method, *out_ar, task);
        fb->zmq_transport_->ClearRecvHandles(*out_ar);
        got = true;
        break;
      }
    }
    if (got) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!got) {
    HLOG(kError,
         "RelayToFallback: no reply from main for pool={} method={} net_key={} "
         "within {}s",
         task->pool_id_, method, net_key, kTotalTimeout);
    return false;
  }

  // Complete the original (local, private-heap) FutureShm in place. The fence
  // (SetBitsSystem) publishes the deserialized outputs before completion is
  // observed by the local waiter (e.g. the CTE handler's co_await).
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  return true;
}

}  // namespace clio::run
