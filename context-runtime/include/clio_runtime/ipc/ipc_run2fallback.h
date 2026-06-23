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
#include "clio_runtime/future.h"

namespace clio::run {

class IpcManager;
class Container;

/**
 * A runtime-internal subtask that has been punted to the main runtime via
 * IpcRun2Fallback::PuntCopyIn and is awaiting in-place completion. The punting
 * worker keeps one of these per outstanding punt and polls it each loop
 * iteration (CompletePunt); no worker thread is blocked waiting for main.
 */
struct PendingPunt {
  /** AllocateBuffer'd FutureShm+copy_space sent to main (freed on completion). */
  ctp::ipc::FullPtr<char> copy_buffer_;
  /** The copy Future: poll its FutureShm for FUTURE_COMPLETE + read copy_space. */
  Future<Task> copy_future_;
  /** The original subtask Future: carries the parent RunContext + the original
   *  task pointer that the parent coroutine will read outputs from. */
  Future<Task> orig_future_;
  /** Local external-stub container providing LoadTask for the concrete type. */
  Container *container_ = nullptr;
  /** Method id for SaveTask/LoadTask dispatch. */
  u32 method_ = 0;
};

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
   * Non-blocking punt of a runtime-internal subtask targeting an external pool
   * (e.g. CTE on the user runtime calling a bdev hosted on main).
   *
   * Unlike RelayToFallback (a synchronous ZMQ round-trip that blocks the worker
   * thread for the subtask's whole duration), this serializes the task into a
   * fresh SHARED FutureShm copy_space and enqueues it onto main's worker lane
   * (SendIn-style). Main resolves the shared FutureShm, deserializes the task,
   * runs it, writes the outputs back into copy_space, and sets FUTURE_COMPLETE
   * in place — all without this worker blocking. The worker registers the
   * returned PendingPunt and later calls CompletePunt to finish.
   *
   * @param ipc This runtime's IpcManager (its fallback_ is the main client).
   * @param container The local external-stub container (SaveTask for the type).
   * @param task The task to punt (generic Task pointer).
   * @param orig_future The subtask's original Future (carries the parent
   *   RunContext that must be resumed once main completes the task).
   * @param out On success, populated with the state CompletePunt needs.
   * @return true if the task was serialized + enqueued to main; false if no
   *   fallback, already punted, or serialization/allocation failed.
   */
  static bool PuntCopyIn(IpcManager *ipc, Container *container,
                         const ctp::ipc::FullPtr<Task> &task,
                         Future<Task> &orig_future, PendingPunt &out);

  /**
   * Poll a PendingPunt for completion. If main has set FUTURE_COMPLETE on the
   * shared copy FutureShm, deserialize the outputs from copy_space back into the
   * original task, resume the parent coroutine (enqueue the original Future to
   * the parent worker's event queue + AwakenWorker), free the copy FutureShm,
   * and return true. If still pending, return false (caller polls again).
   *
   * @param ipc This runtime's IpcManager.
   * @param p The pending punt produced by PuntCopyIn.
   * @return true if completed (p may be discarded); false if still in flight.
   */
  static bool CompletePunt(IpcManager *ipc, PendingPunt &p);
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_RUN2FALLBACK_H_
