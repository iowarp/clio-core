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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_
#define CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_

#include <atomic>
#include <memory>
#include <vector>

#include "chimaera/task.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "chimaera/scheduler/scheduler.h"

namespace chi {

// Forward declarations
class Worker;


/**
 * Work Orchestrator singleton for managing worker threads and lane scheduling
 *
 * Spawns configurable worker threads of different types using HSHM thread
 * model, maps queue lanes to workers using round-robin scheduling, and
 * coordinates task distribution. Uses hipc::multi_mpsc_ring_buffer for both container
 * queues and process queues.
 */
class WorkOrchestrator {
 public:
  /**
   * Initialize work orchestrator
   * @return true if initialization successful, false otherwise
   */
  bool Init();

  /**
   * Finalize and cleanup orchestrator resources
   */
  void Finalize();

  /**
   * Start all worker threads
   * @return true if all workers started successfully, false otherwise
   */
  bool StartWorkers();

  /**
   * Stop all worker threads
   */
  void StopWorkers();

  /**
   * Get worker by ID
   * @param worker_id Worker identifier
   * @return Pointer to worker or nullptr if not found
   */
  Worker* GetWorker(u32 worker_id) const;

  /**
   * Get total number of workers
   * @return Total count of all workers
   */
  size_t GetWorkerCount() const;


  /**
   * Initialize process queues with multiple lanes (for runtime)
   * @param num_lanes Number of lanes per priority level
   * @return true if initialization successful, false otherwise
   */
  bool ServerInitQueues(u32 num_lanes);

  /**
   * Get next available worker using round-robin scheduling
   * @return Worker ID of next available worker
   */
  WorkerId GetNextAvailableWorker();

  /**
   * Check if orchestrator is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Check if workers are running
   * @return true if workers are active, false otherwise
   */
  bool AreWorkersRunning() const;

  /**
   * Map a lane to a specific worker by setting the worker ID in the lane's header
   * @param lane Raw pointer to the TaskLane
   * @param worker_id Worker ID to assign to this lane
   */
  void MapLaneToWorker(TaskLane* lane, WorkerId worker_id);

  /**
   * Check if there is any work remaining across all containers in the system
   * @param total_work_remaining Reference to store the total work count
   * @return true if work is remaining, false if all work is complete
   */
  bool HasWorkRemaining(u64& total_work_remaining) const;

  /**
   * Get the total worker count (all types)
   * Used by scheduler to determine how to partition workers
   * @return Total number of workers
   */
  u32 GetTotalWorkerCount() const { return static_cast<u32>(all_workers_.size()); }

 private:
  /**
   * Spawn worker threads using HSHM thread model
   * @return true if spawning successful, false otherwise
   */
  bool SpawnWorkerThreads();

  /**
   * Create workers
   * @param count Number of workers to create
   * @return true if creation successful, false otherwise
   */
  bool CreateWorkers(u32 count);

  /**
   * Create a single worker
   * @return true if creation successful, false otherwise
   */
  bool CreateWorker();

  /**
   * Initialize queue lane mappings
   * @return true if mapping successful, false otherwise
   */
  bool InitializeQueueMappings();

  bool is_initialized_ = false;
  bool workers_running_ = false;

  // Worker ownership container (owns all worker unique_ptrs)
  std::vector<std::unique_ptr<Worker>> workers_;

  // All workers for easy access (raw pointers to owned workers)
  std::vector<Worker*> all_workers_;

  // Active lanes pointer to IPC Manager worker queues
  void* active_lanes_;

  // Round-robin scheduling state
  std::atomic<u32> next_worker_index_for_scheduling_;

  // HSHM threads (will be filled during initialization)
  std::vector<hshm::thread::Thread> worker_threads_;
  hshm::thread::ThreadGroup thread_group_;

  // Scheduler pointer (owned by IpcManager, not WorkOrchestrator)
  Scheduler *scheduler_;

};

}  // namespace chi

// Global pointer variable declaration for Work Orchestrator singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::WorkOrchestrator, g_work_orchestrator);

// Macro for accessing the Work Orchestrator singleton using global pointer variable
#define CHI_WORK_ORCHESTRATOR HSHM_GET_GLOBAL_PTR_VAR(::chi::WorkOrchestrator, g_work_orchestrator)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_