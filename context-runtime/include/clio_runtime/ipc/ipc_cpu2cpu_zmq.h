/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

class IpcManager;

/**
 * IPC transport for CPU client → CPU runtime via ZeroMQ (TCP or IPC).
 *
 * Two-phase SendOut/RecvIn:
 *   Phase 1 (worker inline): EnqueueSendOut enqueues to net_queue_
 *   Phase 2 (net worker periodic): SendOut/RecvIn do actual I/O
 */
struct IpcCpu2CpuZmq {
  /** Serialize and send via ZMQ (inbound). */
  template <typename TaskT>
  static Future<TaskT> SendIn(IpcManager *ipc,
                              const clio::run::shared_ptr<TaskT> &task_ptr,
                              IpcMode mode);

  /**
   * Net-worker RecvIn: poll ZMQ transports for incoming tasks.
   * Called by Admin::ClientRecv periodic coroutine.
   * Deserializes tasks, creates FutureShm, enqueues to worker lanes.
   * @param ipc IpcManager
   * @param tasks_received Output: number of tasks received
   * @return true if any work was done
   */
  static bool RecvIn(IpcManager *ipc, u32 &tasks_received);

  /**
   * Worker-inline SendOut: enqueue completed task to net_queue_.
   * The actual ZMQ send happens in the net-worker SendOut phase.
   */
  static void EnqueueSendOut(IpcManager *ipc,
                             const clio::run::shared_ptr<Task> &task,
                             ClientOrigin origin);

  /**
   * Net-worker SendOut: serialize outputs and send via ZMQ.
   * Called by Admin::ClientSend periodic coroutine.
   * Pops from net_queue_, serializes, sends response to client.
   * @param ipc IpcManager
   * @param tasks_sent Output: number of tasks sent
   * @param deferred_deletes Tasks to delete on next invocation (zero-copy safety)
   * @return true if any work was done
   */
  static bool SendOut(IpcManager *ipc, u32 &tasks_sent,
                      std::vector<clio::run::shared_ptr<Task>> &deferred_deletes);

  /** Wait for COMPLETE, deserialize from pending archives (outbound). */
  template <typename TaskT>
  static bool RecvOut(IpcManager *ipc,
                      Future<TaskT> &future, float max_sec);

  /** Re-send a task via ZMQ after server restart. */
  template <typename TaskT>
  static void ResendTask(IpcManager *ipc, Future<TaskT> &future);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_H_
