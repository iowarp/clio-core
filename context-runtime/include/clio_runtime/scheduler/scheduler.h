/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Copyright 2024 IOWarp contributors
#ifndef CLIO_RUNTIME_INCLUDE_SCHEDULER_SCHEDULER_H_
#define CLIO_RUNTIME_INCLUDE_SCHEDULER_SCHEDULER_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

// Forward declarations
class IpcManager;
class WorkOrchestrator;
class Worker;
struct RunContext;

/**
 * Base class for task scheduling strategies.
 * Implementations decide how to map tasks to workers and balance load.
 */
class Scheduler {
 public:
  /**
   * Virtual destructor
   */
  virtual ~Scheduler() = default;

  /**
   * Decides how to pin workers to cores and create worker groups.
   * Called after all workers have been spawned in the WorkOrchestrator.
   *
   * @param work_orch Pointer to the work orchestrator
   */
  virtual void DivideWorkers(WorkOrchestrator *work_orch) = 0;

  /**
   * Determines which worker to initially map a task to from clients.
   * First few workers are always the scheduling workers.
   * Analogous to the old MapTaskToLane function.
   *
   * @param ipc_manager Pointer to the IPC manager
   * @param task The task to be scheduled
   * @return Worker lane ID to assign the task to
   */
  virtual u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) = 0;

  /**
   * Determines which worker to initially map a task to from runtime.
   * Called in RouteTask via IpcManager::EnqueueRuntime.
   *
   * @param worker The worker that called this method
   * @param task The task to be scheduled
   * @param container The resolved execution container for the task (may be
   *   nullptr when called from IpcManager::Send without a resolved container).
   *   Used for task-group affinity lookups and updates.
   * @return Worker ID to assign the task to
   */
  virtual u32 RuntimeMapTask(Worker *worker, const Future<Task> &task,
                             ContainerHold container) = 0;

  /**
   * Either steal or delegate tasks on a worker to balance load.
   * Should be called after every ProcessNewTasks loop before SuspendMe.
   *
   * @param worker Pointer to the worker to rebalance
   */
  virtual void RebalanceWorker(Worker *worker) = 0;

  /**
   * Adjust polling interval for periodic tasks based on work done.
   * Implements adaptive polling to reduce CPU utilization when tasks
   * aren't performing I/O or computation.
   *
   * @param task The periodic task whose RunContext polling interval to adjust
   */
  virtual void AdjustPolling(const clio::run::shared_ptr<Task> &task) = 0;

  /**
   * Get the designated GPU worker (polls GPU queues).
   * @return Pointer to GPU worker, or nullptr if none assigned
   */
  virtual Worker *GetGpuWorker() const { return nullptr; }

  /**
   * Get the designated network worker. With the recv/send split this aliases
   * the recv worker (so callers that register sockets with an EventManager
   * still talk to the thread that polls them); use the more specific
   * GetNetSendWorker() / GetNetRecvWorker() when routing periodic tasks.
   * @return Pointer to net (recv) worker, or nullptr if none assigned
   */
  virtual Worker *GetNetWorker() const { return nullptr; }

  /**
   * Drains net_queue_ (SendIn / SendOut) and writes via per-peer DEALERs;
   * also runs the ClientSend periodic that writes to client return sockets.
   * @return Pointer to net send worker, or nullptr if none assigned
   */
  virtual Worker *GetNetSendWorker() const { return GetNetWorker(); }

  /**
   * Polls the inbound ROUTER (kRecv) and the client-facing server (kClientRecv).
   * Owns the EventManager that the transport FDs are registered with so it
   * wakes on socket readability instead of busy-polling.
   * @return Pointer to net recv worker, or nullptr if none assigned
   */
  virtual Worker *GetNetRecvWorker() const { return GetNetWorker(); }
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_SCHEDULER_SCHEDULER_H_
