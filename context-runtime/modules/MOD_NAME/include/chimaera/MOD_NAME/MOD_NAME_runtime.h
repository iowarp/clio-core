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
  void Run(chi::u32 method, chi::Future<chi::Task>& task_future, chi::RunContext& rctx) override;

  /**
   * Delete/cleanup a task
   */
  void Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);

  /**
   * Handle Custom task
   */
  void Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& rctx);

  /**
   * Handle CoMutexTest task
   */
  void CoMutexTest(hipc::FullPtr<CoMutexTestTask> task, chi::RunContext& rctx);

  /**
   * Handle CoRwLockTest task
   */
  void CoRwLockTest(hipc::FullPtr<CoRwLockTestTask> task, chi::RunContext& rctx);

  /**
   * Handle WaitTest task
   */
  void WaitTest(hipc::FullPtr<WaitTestTask> task, chi::RunContext& rctx);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   */
  void Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& rctx);

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
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive, chi::Future<chi::Task>& task_future) override;

  /**
   * Deserialize task parameters from network transfer (unified method)
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive, chi::Future<chi::Task>& task_future) override;

  /**
   * Deserialize task input parameters using LocalSerialize (for local transfers)
   */
  void LocalLoadIn(chi::u32 method, chi::LocalLoadTaskArchive& archive,
                   chi::Future<chi::Task>& task_future) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveOut(chi::u32 method, chi::LocalSaveTaskArchive& archive,
                    chi::Future<chi::Task>& task_future) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  void NewCopy(chi::u32 method,
               chi::Future<chi::Task>& orig_future,
               chi::Future<chi::Task>& dup_future, bool deep) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 chi::Future<chi::Task>& origin_future,
                 chi::Future<chi::Task>& replica_future) override;
};

} // namespace chimaera::MOD_NAME

#endif // MOD_NAME_RUNTIME_H_