#ifndef CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_
#define CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_

#include <atomic>
#include <memory>
#include <vector>

#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"

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
   * Get workers by thread type
   * @param thread_type Type of worker threads
   * @return Vector of worker pointers of specified type
   */
  std::vector<Worker*> GetWorkersByType(ThreadType thread_type) const;

  /**
   * Get total number of workers
   * @return Total count of all workers
   */
  size_t GetWorkerCount() const;

  /**
   * Get worker count by type
   * @param thread_type Type of worker threads
   * @return Count of workers of specified type
   */
  u32 GetWorkerCountByType(ThreadType thread_type) const;


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
   * Get stack growth direction (detected during initialization)
   * @return true if stack grows downward, false if upward
   */
  bool IsStackDownward() const;

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
   * Assign a task to a specific worker type using round-robin scheduling
   * @param thread_type Worker type to assign the task to (kSchedWorker or kSlow)
   * @param task_ptr Full pointer to task to assign to the worker type
   */
  void AssignToWorkerType(ThreadType thread_type, const FullPtr<Task>& task_ptr);

  /**
   * Get the dedicated network worker
   * @return Pointer to the network worker
   */
  Worker* GetNetWorker() const { return net_worker_; }

 private:
  /**
   * Spawn worker threads using HSHM thread model
   * @return true if spawning successful, false otherwise
   */
  bool SpawnWorkerThreads();

  /**
   * Create workers of specified type
   * @param thread_type Type of workers to create
   * @param count Number of workers to create
   * @return true if creation successful, false otherwise
   */
  bool CreateWorkers(ThreadType thread_type, u32 count);

  /**
   * Create a single worker with specified type
   * @param thread_type Type of worker to create
   * @return true if creation successful, false otherwise
   */
  bool CreateWorker(ThreadType thread_type);

  /**
   * Initialize queue lane mappings
   * @return true if mapping successful, false otherwise
   */
  bool InitializeQueueMappings();

  bool is_initialized_ = false;
  bool workers_running_ = false;
  bool stack_is_downward_ = true; // Stack growth direction (detected at initialization)

  // Worker containers organized by type
  std::vector<std::unique_ptr<Worker>> sched_workers_;

  // All workers for easy access
  std::vector<Worker*> all_workers_;

  // Worker groups for task routing based on execution characteristics
  std::vector<Worker*> scheduler_workers_; ///< Fast task workers (EstCpuTime < 50us)
  std::vector<Worker*> slow_workers_;      ///< Slow task workers (EstCpuTime >= 50us)
  Worker* net_worker_;                     ///< Dedicated network worker for Send/Recv tasks

  // Active lanes pointer to IPC Manager worker queues
  void* active_lanes_;

  // Round-robin scheduling state
  std::atomic<u32> next_worker_index_for_scheduling_;

  // HSHM threads (will be filled during initialization)
  std::vector<hshm::thread::Thread> worker_threads_;
  hshm::thread::ThreadGroup thread_group_;

};

}  // namespace chi

// Global pointer variable declaration for Work Orchestrator singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::WorkOrchestrator, g_work_orchestrator);

// Macro for accessing the Work Orchestrator singleton using global pointer variable
#define CHI_WORK_ORCHESTRATOR HSHM_GET_GLOBAL_PTR_VAR(::chi::WorkOrchestrator, g_work_orchestrator)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORK_ORCHESTRATOR_H_