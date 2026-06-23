/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_run2fallback.h"

#include <atomic>
#include <mutex>

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

bool IpcRun2Fallback::PuntCopyIn(IpcManager *ipc, Container *container,
                                 const ctp::ipc::FullPtr<Task> &task,
                                 Future<Task> &orig_future, PendingPunt &out) {
  if (ipc == nullptr || container == nullptr || task.IsNull()) {
    HLOG(kError, "PuntCopyIn: null ipc/container/task");
    return false;
  }
  IpcManager *fb = ipc->fallback_.get();
  if (fb == nullptr) {
    HLOG(kError, "PuntCopyIn: no fallback_ on this runtime");
    return false;
  }
  auto orig_shm = orig_future.GetFutureShm();
  if (orig_shm.IsNull()) {
    HLOG(kError, "PuntCopyIn: null original FutureShm");
    return false;
  }
  if (orig_shm->flags_.Any(FutureShm::FUTURE_PUNTED)) {
    HLOG(kError, "PuntCopyIn: already punted");
    return false;
  }
  // Mark the ORIGINAL FutureShm punted so the worker never re-punts it while
  // the copy is in flight.
  orig_shm->flags_.SetBits(FutureShm::FUTURE_PUNTED);

  const u32 method = task->method_;

  // Allocate a fresh SHARED FutureShm + copy_space. AllocateBuffer now draws
  // from the runtime's per-process MultiProcessAllocator segments (Stage 1),
  // which are registered with main via RegisterMemoryWithFallback — so main can
  // resolve this FutureShm and complete it in place.
  size_t copy_space_size = task->GetCopySpaceSize();
  if (copy_space_size == 0) {
    copy_space_size = KILOBYTES(4);
  }
  size_t alloc_size = sizeof(FutureShm) + copy_space_size;
  ctp::ipc::FullPtr<char> buffer = ipc->AllocateBuffer(alloc_size);
  if (buffer.IsNull()) {
    HLOG(kError, "PuntCopyIn: AllocateBuffer({}) failed", alloc_size);
    orig_shm->flags_.UnsetBits(FutureShm::FUTURE_PUNTED);
    return false;
  }
  FutureShm *copy_shm = new (buffer.ptr_) FutureShm();
  copy_shm->pool_id_ = task->pool_id_;
  copy_shm->method_id_ = method;
  // Present to main as an SHM client call so EndTask completes copy_space in
  // place (IpcCpu2Cpu::RuntimeSend), not a run2run SendOut. Non-zero vaddr keeps
  // main off the legacy GPU path (client_task_vaddr_ == 0).
  copy_shm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  copy_shm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(task.ptr_);
  copy_shm->input_.copy_space_size_ = copy_space_size;
  copy_shm->output_.copy_space_size_ = copy_space_size;
  copy_shm->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

  // The subtask must look like a fresh LOCAL client call to main: clear the
  // process-local RunContext flags (main creates its own via BeginTask) and
  // present a Local pool_query (else main's RouteTask re-derives TASK_REMOTE
  // from the return-node pool_query). Same flag dance the external-client punt
  // (SendIn) relies on.
  task->ClearFlags(TASK_REMOTE | TASK_RUN_CTX_EXISTS | TASK_STARTED);
  task->pool_query_ = PoolQuery::Local();

  // Serialize the task (inputs + bulk) into copy_space via the fallback's kShm
  // transport. The transport is segment-agnostic (operates on ctx.copy_space);
  // guard it against concurrent punts from other workers.
  {
    std::lock_guard<std::mutex> lock(fb->zmq_client_send_mutex_);
    ctp::lbm::LbmContext ctx;
    ctx.copy_space = copy_shm->copy_space;
    ctx.shm_info_ = &copy_shm->input_;
    SaveTaskArchive in_ar(MsgType::kSerializeIn, fb->shm_send_transport_.get());
    container->SaveTask(method, in_ar, task);
    fb->shm_send_transport_->Send(in_ar, ctx);
  }

  // Build the copy Future and enqueue it onto main's worker lane — the same
  // in-place SendIn mechanism the external-client punt uses. Mark FUTURE_PUNTED
  // so main fails rather than bounces it if it also lacks the pool.
  auto copy_shm_shmptr = buffer.shm_.Cast<FutureShm>();
  Future<Task> copy_future(copy_shm_shmptr, task);
  copy_shm->flags_.SetBits(FutureShm::FUTURE_PUNTED);
  LaneId lane_id = fb->scheduler_->ClientMapTask(fb, copy_future);
  auto &lane = fb->worker_queues_->GetLane(lane_id, 0);
  lane.Push(copy_future);
  fb->AwakenWorker(&lane);

  out.copy_buffer_ = buffer;
  out.copy_future_ = copy_future;
  out.orig_future_ = orig_future;
  out.container_ = container;
  out.method_ = method;
  return true;
}

bool IpcRun2Fallback::CompletePunt(IpcManager *ipc, PendingPunt &p) {
  auto copy_shm = p.copy_future_.GetFutureShm();
  if (copy_shm.IsNull()) {
    return true;  // nothing to wait on — drop the entry
  }
  if (!copy_shm->flags_.Any(FutureShm::FUTURE_COMPLETE)) {
    return false;  // still in flight on main
  }
  // Publish-before-read: see the outputs main wrote before reading copy_space.
  std::atomic_thread_fence(std::memory_order_acquire);

  IpcManager *fb = ipc->fallback_.get();
  ctp::ipc::FullPtr<Task> orig_task = p.orig_future_.GetTaskPtr();
  if (!orig_task.IsNull() && p.container_ != nullptr && fb != nullptr) {
    // Deserialize the outputs main serialized into copy_space back into the
    // ORIGINAL task the parent coroutine holds. Mirrors IpcCpu2Cpu::ClientRecv,
    // dispatched to the concrete type via the stub container's LoadTask.
    std::lock_guard<std::mutex> lock(fb->zmq_client_send_mutex_);
    ctp::lbm::LbmContext ctx;
    ctx.copy_space = copy_shm->copy_space;
    ctx.shm_info_ = &copy_shm->output_;
    LoadTaskArchive out_ar;
    fb->shm_recv_transport_->Recv(out_ar, ctx);
    out_ar.ResetBulkIndex();
    out_ar.msg_type_ = MsgType::kSerializeOut;
    p.container_->LoadTask(p.method_, out_ar, orig_task);
  }

  // Resume the parent coroutine. Mirror IpcCpu2Self::RuntimeSend: a subtask with
  // a parent is resumed by enqueueing its Future to the parent worker's event
  // queue (ProcessEventQueue sets FUTURE_COMPLETE on the parent's thread). Only
  // a top-level task sets FUTURE_COMPLETE on its own FutureShm directly.
  RunContext *parent_task = p.orig_future_.GetParentTask();
  if (parent_task != nullptr && parent_task->event_queue_ != nullptr) {
    auto *parent_event_queue = reinterpret_cast<ctp::ipc::mpsc_ring_buffer<
        Future<Task, CLIO_QUEUE_ALLOC_T>, ctp::ipc::MallocAllocator> *>(
        parent_task->event_queue_);
    parent_event_queue->Emplace(p.orig_future_);
    if (parent_task->lane_ != nullptr) {
      ipc->AwakenWorker(parent_task->lane_);
    }
  } else {
    auto orig_shm = p.orig_future_.GetFutureShm();
    if (!orig_shm.IsNull()) {
      orig_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }
  }

  // The copy FutureShm has served its purpose; release it.
  ipc->FreeBuffer(p.copy_buffer_);
  return true;
}

}  // namespace clio::run
