// Copyright 2024 IOWarp contributors
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_

#include <atomic>
#include <vector>

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Default scheduler implementation.
 * Uses PID+TID hash-based lane mapping and provides no rebalancing.
 * Manages its own worker partitioning into scheduler, slow, and network groups.
 */
class DefaultScheduler : public Scheduler {
 public:
  /**
   * Constructor
   */
  DefaultScheduler() : net_worker_(nullptr) {}

  /**
   * Destructor
   */
  ~DefaultScheduler() override = default;

  /**
   * Partition workers into scheduler, slow, and network worker groups.
   * Reads worker counts from ConfigManager and assigns workers to:
   * - scheduler_workers_: First N workers for fast tasks
   * - slow_workers_: Next M workers for long-running tasks
   * - net_worker_: Last worker for network operations
   * @param work_orch Pointer to the work orchestrator
   */
  void DivideWorkers(WorkOrchestrator *work_orch) override;

  /**
   * Map task to lane using PID+TID hash.
   */
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;

  /**
   * Return current worker (no migration).
   * @param worker The worker that called this method
   * @param task The task to be scheduled
   * @return Worker ID to assign the task to
   */
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) override;

  /**
   * No rebalancing in default scheduler.
   */
  void RebalanceWorker(Worker *worker) override;

  /**
   * Adjust polling interval for periodic tasks based on work done.
   * Implements exponential backoff when tasks aren't doing work.
   */
  void AdjustPolling(RunContext *run_ctx) override;

  /**
   * Assign a task to a worker of the specified type using round-robin.
   * @param thread_type Type of worker to assign to (kSchedWorker or kSlow)
   * @param future Future containing the task to assign
   */
  void AssignToWorkerType(ThreadType thread_type, Future<Task> &future);

  /**
   * Get the network worker
   * @return Pointer to the network worker, or nullptr if not assigned
   */
  Worker *GetNetWorker() const { return net_worker_; }

 private:
  /**
   * Map task to lane by PID+TID hash
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  u32 MapByPidTid(u32 num_lanes);

  // Worker partitioning - specific to default scheduler
  std::vector<Worker *> scheduler_workers_;  ///< Workers for fast tasks
  std::vector<Worker *> slow_workers_;       ///< Workers for long-running tasks
  Worker *net_worker_;                        ///< Network worker

  // Round-robin assignment counters
  std::atomic<size_t> scheduler_idx_{0};
  std::atomic<size_t> slow_idx_{0};
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
