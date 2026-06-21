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
  /** Serialize task into SHM ring buffer and enqueue to worker. */
  template <typename TaskT>
  static Future<TaskT> ClientSend(IpcManager *ipc,
                                   const ctp::ipc::FullPtr<TaskT> &task_ptr);

  /** Deserialize task from SHM ring buffer on runtime side. */
  static ctp::ipc::FullPtr<Task> RuntimeRecv(
      IpcManager *ipc, Future<Task> &future, Container *container,
      u32 method_id, ctp::lbm::Transport *recv_transport);

  /** Serialize outputs and set FUTURE_COMPLETE. */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container,
      ctp::lbm::Transport *send_transport);

  /** Wait for COMPLETE, then deserialize outputs. */
  template <typename TaskT>
  static bool ClientRecv(IpcManager *ipc,
                          Future<TaskT> &future, float max_sec);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_H_
