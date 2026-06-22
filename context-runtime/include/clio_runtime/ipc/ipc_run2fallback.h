/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_RUN2FALLBACK_H_
#define CLIO_RUNTIME_INCLUDE_IPC_RUN2FALLBACK_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

class IpcManager;
class Container;

/**
 * IPC transport for forwarding ("punting") a task from this runtime to the
 * fallback ("main") runtime when the task's pool is not owned locally.
 *
 * It is SHM-based and reuses the cpu2cpu worker queues: the punted task's
 * FutureShm already lives in the client's shared data segment (which the main
 * runtime has registered via the dual RegisterMemory path), so the main
 * runtime deserializes, runs, and completes that FutureShm IN PLACE. The
 * original client — already polling that FutureShm — sees the result directly,
 * without the response being relayed back through this runtime.
 *
 * Only SendIn is implemented: the response path is the main runtime's normal
 * in-place completion of the shared FutureShm, so this transport has no
 * RuntimeRecv/RuntimeSend of its own. The "original communication method"
 * (origin_, client_task_vaddr_, response routing) is already carried in the
 * FutureShm and rides along unchanged.
 */
struct IpcRun2Fallback {
  /**
   * Punt an already-deserialized-from-client (still in copy_space) task to the
   * fallback runtime by enqueueing the SAME Future onto the main runtime's
   * worker lane. Marks the FutureShm FUTURE_PUNTED so it is never re-punted.
   *
   * @param ipc This runtime's IpcManager (its fallback_ is the main-runtime
   *   client connection).
   * @param future The client's Future, whose FutureShm + serialized task live
   *   in shared memory the main runtime can resolve.
   * @return true if the task was punted; false if no fallback is configured or
   *   the task was already punted (caller should then fail it locally).
   */
  static bool SendIn(IpcManager *ipc, Future<Task> &future);

  /**
   * Relay a task to the fallback ("main") runtime via a SYNCHRONOUS round-trip
   * over the fallback DEALER, then complete the original (local) FutureShm.
   *
   * Used for runtime-internal subtasks targeting an external pool (e.g. CTE on
   * the user runtime calling a bdev hosted on main): their task + data live in
   * the runtime's PRIVATE heap, so they cannot be completed in place by main.
   * Instead we serialize the task (inputs + bulk data) against the local
   * external-stub @p container, send it to main as an ordinary client call,
   * receive the serialized outputs, deserialize them back into the task, and
   * mark the local FutureShm complete so the local waiter resumes. Works for
   * any task whose pool resolves to an external stub.
   *
   * @param ipc This runtime's IpcManager (its fallback_ is the main client).
   * @param container The local external-stub container (provides SaveTask /
   *   LoadTask for this task's concrete type).
   * @param task The task to relay (generic Task pointer).
   * @param future The task's Future, whose FutureShm is completed on success.
   * @return true if the task was relayed and completed; false if no fallback,
   *   already punted, or the main runtime did not answer.
   */
  static bool RelayToFallback(IpcManager *ipc, Container *container,
                              const ctp::ipc::FullPtr<Task> &task,
                              Future<Task> &future);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_RUN2FALLBACK_H_
