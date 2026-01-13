#ifndef MOD_NAME_RUNTIME_H_
#define MOD_NAME_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <chimaera/container.h>
#include <chimaera/comutex.h>
#include <chimaera/corwlock.h>
#include "MOD_NAME_tasks.h"
#include "autogen/MOD_NAME_methods.h"
#include "MOD_NAME_client.h"

namespace chimaera::MOD_NAME {

// Forward declarations (CustomTask only, CreateTask is a using alias in MOD_NAME_tasks.h)
struct CustomTask;
struct CoMutexTestTask;
struct CoRwLockTestTask;
struct WaitTestTask;

/**
 * Runtime implementation for MOD_NAME container
 */
class Runtime : public chi::Container {
public:
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = chimaera::MOD_NAME::CreateParams;

private:
  // Container-specific state
  chi::u32 create_count_ = 0;
  chi::u32 custom_count_ = 0;

  // Client for making calls to this ChiMod
  Client client_;

  // Static synchronization objects for testing
  static chi::CoMutex test_comutex_;
  static chi::CoRwLock test_corwlock_;

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
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  chi::TaskResume Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) override;

  /**
   * Delete/cleanup a task
   */
  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task
   * Returns TaskResume for consistency with Run method
   */
  chi::TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);

  /**
   * Handle Custom task
   * Returns TaskResume for consistency with Run method
   */
  chi::TaskResume Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& rctx);

  /**
   * Handle CoMutexTest task
   * Returns TaskResume for consistency with Run method
   */
  chi::TaskResume CoMutexTest(hipc::FullPtr<CoMutexTestTask> task, chi::RunContext& rctx);

  /**
   * Handle CoRwLockTest task
   * Returns TaskResume for consistency with Run method
   */
  chi::TaskResume CoRwLockTest(hipc::FullPtr<CoRwLockTestTask> task, chi::RunContext& rctx);

  /**
   * Handle WaitTest task
   * Returns TaskResume for coroutine-based async operations
   */
  chi::TaskResume WaitTest(hipc::FullPtr<WaitTestTask> task, chi::RunContext& rctx);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   * Returns TaskResume for consistency with Run method
   */
  chi::TaskResume Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& rctx);

  /**
   * Get remaining work count for this container
   * Template implementation returns 0 (no work tracking)
   */
  chi::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods
  //===========================================================================

  /**
   * Serialize task parameters for network transfer (unified method)
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive, hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters into an existing task from network transfer
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  hipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  hipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method,
                                        hipc::FullPtr<chi::Task> orig_task_ptr, bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 hipc::FullPtr<chi::Task> origin_task_ptr,
                 hipc::FullPtr<chi::Task> replica_task_ptr) override;
};

} // namespace chimaera::MOD_NAME

#endif // MOD_NAME_RUNTIME_H_