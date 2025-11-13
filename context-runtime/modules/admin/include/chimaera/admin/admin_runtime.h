#ifndef ADMIN_RUNTIME_H_
#define ADMIN_RUNTIME_H_

#include "admin_client.h"
#include "admin_tasks.h"
#include <chimaera/chimaera.h>
#include <chimaera/container.h>
#include <chimaera/pool_manager.h>
#include <chimaera/comutex.h>
#include <chimaera/unordered_map_ll.h>
#include <vector>

namespace chimaera::admin {

// Admin local queue indices
enum AdminQueueIndex {
  kMetadataQueue = 0,          // Queue for metadata operations
  kClientSendTaskInQueue = 1,  // Queue for client task input processing
  kServerRecvTaskInQueue = 2,  // Queue for server task input reception
  kServerSendTaskOutQueue = 3, // Queue for server task output sending
  kClientRecvTaskOutQueue = 4  // Queue for client task output reception
};

// Forward declarations
// Note: CreateTask and GetOrCreatePoolTask are using aliases defined in
// admin_tasks.h We cannot forward declare using aliases, so we rely on the
// include

/**
 * Runtime implementation for Admin container
 *
 * Critical ChiMod responsible for managing ChiPools and runtime lifecycle.
 * Must always be found by the runtime or a fatal error occurs.
 */
class Runtime : public chi::Container {
public:
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = chimaera::admin::CreateParams;

private:
  // Container-specific state
  chi::u32 create_count_ = 0;
  chi::u32 pools_created_ = 0;
  chi::u32 pools_destroyed_ = 0;

  // Runtime state
  bool is_shutdown_requested_ = false;

  // Client for making calls to this ChiMod
  Client client_;

  // Network task tracking maps (keyed by net_key for efficient lookup)
  // Using lock-free unordered_map_ll with 1024 buckets for high concurrency
  static constexpr size_t kNumMapBuckets = 1024;
  chi::unordered_map_ll<size_t, hipc::FullPtr<chi::Task>> send_map_{kNumMapBuckets};  // Tasks sent to remote nodes
  chi::unordered_map_ll<size_t, hipc::FullPtr<chi::Task>> recv_map_{kNumMapBuckets};  // Tasks received from remote nodes

  // CoMutex vector for synchronizing map access (one per worker thread)
  // Mutable to allow locking in const methods like GetWorkRemaining()
  mutable std::vector<chi::CoMutex> send_map_locks_;
  mutable std::vector<chi::CoMutex> recv_map_locks_;

public:
  /**
   * Constructor
   */
  Runtime() = default;

  /**
   * Destructor
   */
  virtual ~Runtime() = default;

  /**
   * Initialize container with pool information
   */
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
           chi::RunContext &rctx) override;

  /**
   * Delete/cleanup a task
   */
  void Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task - Initialize the Admin container (IS_ADMIN=true)
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext &rctx);

  /**
   * Handle GetOrCreatePool task - Pool get-or-create operation (IS_ADMIN=false)
   */
  void GetOrCreatePool(
      hipc::FullPtr<
          chimaera::admin::GetOrCreatePoolTask<chimaera::admin::CreateParams>>
          task,
      chi::RunContext &rctx);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   */
  void Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &rctx);

  /**
   * Handle DestroyPool task - Destroy an existing ChiPool
   */
  void DestroyPool(hipc::FullPtr<DestroyPoolTask> task, chi::RunContext &rctx);

  /**
   * Handle StopRuntime task - Stop the entire runtime
   */
  void StopRuntime(hipc::FullPtr<StopRuntimeTask> task, chi::RunContext &rctx);

  /**
   * Handle Flush task - Flush administrative operations
   */
  void Flush(hipc::FullPtr<FlushTask> task, chi::RunContext &rctx);

  //===========================================================================
  // Distributed Task Scheduling Methods
  //===========================================================================

  /**
   * Handle Send - Send task inputs or outputs over network
   */
  void Send(hipc::FullPtr<SendTask> task, chi::RunContext &rctx);

  /**
   * Helper: Send task inputs to remote node
   */
  void SendIn(hipc::FullPtr<SendTask> task, chi::RunContext &rctx);

  /**
   * Helper: Send task outputs back to origin node
   */
  void SendOut(hipc::FullPtr<SendTask> task);

  /**
   * Handle Recv - Receive task inputs or outputs from network
   */
  void Recv(hipc::FullPtr<RecvTask> task, chi::RunContext &rctx);

  /**
   * Helper: Receive task inputs from remote node
   */
  void RecvIn(hipc::FullPtr<RecvTask> task, chi::LoadTaskArchive& archive, hshm::lbm::Server* lbm_server);

  /**
   * Helper: Receive task outputs from remote node
   */
  void RecvOut(hipc::FullPtr<RecvTask> task, chi::LoadTaskArchive& archive, hshm::lbm::Server* lbm_server);

  /**
   * Get remaining work count for this admin container
   * Admin container typically has no pending work, returns 0
   */
  chi::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods
  //===========================================================================

  /**
   * Serialize task parameters (IN or OUT based on archive mode)
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive &archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters (IN or OUT based on archive mode)
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                hipc::FullPtr<chi::Task>& task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  void NewCopy(chi::u32 method, const hipc::FullPtr<chi::Task> &orig_task,
               hipc::FullPtr<chi::Task> &dup_task, bool deep) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 hipc::FullPtr<chi::Task> origin_task,
                 hipc::FullPtr<chi::Task> replica_task) override;

private:
  /**
   * Initiate runtime shutdown sequence
   */
  void InitiateShutdown(chi::u32 grace_period_ms);
};

} // namespace chimaera::admin

#endif // ADMIN_RUNTIME_H_