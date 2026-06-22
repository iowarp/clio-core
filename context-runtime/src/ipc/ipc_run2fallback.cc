/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_run2fallback.h"

#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/scheduler/scheduler.h"

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

}  // namespace clio::run
