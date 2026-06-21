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
 * Two-phase RuntimeSend/RuntimeRecv:
 *   Phase 1 (worker inline): EnqueueRuntimeSend enqueues to net_queue_
 *   Phase 2 (net worker periodic): RuntimeSend/RuntimeRecv do actual I/O
 */
struct IpcCpu2CpuZmq {
  /** Serialize and send via ZMQ. */
  template <typename TaskT>
  static Future<TaskT> ClientSend(IpcManager *ipc,
                                   const ctp::ipc::FullPtr<TaskT> &task_ptr,
                                   IpcMode mode);

  /**
   * Net-worker RuntimeRecv: poll ZMQ transports for incoming tasks.
   * Called by Admin::ClientRecv periodic coroutine.
   * Deserializes tasks, creates FutureShm, enqueues to worker lanes.
   * @param ipc IpcManager
   * @param tasks_received Output: number of tasks received
   * @return true if any work was done
   */
  static bool RuntimeRecv(IpcManager *ipc, u32 &tasks_received);

  /**
   * Worker-inline RuntimeSend: enqueue completed task to net_queue_.
   * The actual ZMQ send happens in RuntimeSendOut (net worker phase).
   */
  static void EnqueueRuntimeSend(IpcManager *ipc, RunContext *run_ctx,
                                  u32 origin);

  /**
   * Net-worker RuntimeSend: serialize outputs and send via ZMQ.
   * Called by Admin::ClientSend periodic coroutine.
   * Pops from net_queue_, serializes, sends response to client.
   * @param ipc IpcManager
   * @param tasks_sent Output: number of tasks sent
   * @param deferred_deletes Tasks to delete on next invocation (zero-copy safety)
   * @return true if any work was done
   */
  static bool RuntimeSend(IpcManager *ipc, u32 &tasks_sent,
                           std::vector<ctp::ipc::FullPtr<Task>> &deferred_deletes);

  /** Wait for COMPLETE, deserialize from pending archives. */
  template <typename TaskT>
  static bool ClientRecv(IpcManager *ipc,
                          Future<TaskT> &future, float max_sec);

  /** Re-send a task via ZMQ after server restart. */
  template <typename TaskT>
  static void ResendTask(IpcManager *ipc, Future<TaskT> &future);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_ZMQ_H_
