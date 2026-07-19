/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

class IpcManager;

/**
 * IPC transport for CPU client → CPU runtime via shared memory (lightbeam).
 */
struct IpcCpu2Cpu {
  /** Serialize task into SHM ring buffer and enqueue to worker (inbound). */
  template <typename TaskT>
  static Future<TaskT> SendIn(IpcManager *ipc,
                              const clio::run::shared_ptr<TaskT> &task_ptr);

  /**
   * Drain this worker thread's named MPSC SHM receive server (clio-<pid>-<tid>,
   * DONTWAIT). If a client task is waiting, deserialize it, build a server-side
   * FutureShm carrying the client's response identity, and push the resulting
   * Future onto `lane` for the normal dispatch path. Keeps all task/future
   * deserialization off the worker. Must run on the worker's own thread (the
   * per-thread server is keyed by tid).
   * @return true if a task was received and enqueued (the caller did work).
   */
  static bool RecvIn(IpcManager *ipc, TaskLane *lane);

  /** Deserialize task from SHM ring buffer on runtime side (inbound). */
  static clio::run::shared_ptr<clio::run::Task> RecvIn(
      IpcManager *ipc, Future<Task> &future,
      u32 method_id, ctp::lbm::Transport *recv_transport);

  /** Serialize outputs and set FUTURE_COMPLETE (outbound). */
  static void SendOut(
      IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr,
      ctp::lbm::Transport *send_transport);

  /** Wait for COMPLETE, then deserialize outputs (outbound). */
  template <typename TaskT>
  static bool RecvOut(IpcManager *ipc,
                      Future<TaskT> &future, float max_sec);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_H_
