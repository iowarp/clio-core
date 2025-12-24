#ifndef SIMPLE_MOD_RUNTIME_H_
#define SIMPLE_MOD_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <chimaera/container.h>
#include "simple_mod_tasks.h"
#include "simple_mod_client.h"

namespace external_test::simple_mod {

// Simple mod local queue indices
enum SimpleModQueueIndex {
  kMetadataQueue = 0,  // Queue for metadata operations
};

/**
 * Runtime implementation for Simple Mod container
 * 
 * Minimal ChiMod for testing external development patterns.
 * Demonstrates basic runtime structure for external ChiMod development.
 */
class Runtime : public chi::Container {
public:
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = external_test::simple_mod::CreateParams;

private:
  // Container-specific state
  chi::u32 create_count_ = 0;
  
  // Client for making calls to this ChiMod
  Client client_;

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
   * @param pool_id The unique ID of this pool
   * @param pool_name The semantic name of this pool (user-provided)
   * @param container_id The container ID
   */
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) override;

  /**
   * Delete/cleanup a task
   */
  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task - Initialize the Simple Mod container
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);

  /**
   * Handle Destroy task - Destroy the Simple Mod container
   */
  void Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& rctx);

  /**
   * Handle Flush task - Flush simple mod operations
   */
  void Flush(hipc::FullPtr<FlushTask> task, chi::RunContext& rctx);

  /**
   * Get remaining work count for this simple mod container
   */
  chi::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Container Virtual Methods (automatically generated in autogen/)
  //===========================================================================

  /**
   * Serialize task parameters for network transfer (auto-generated)
   * @param method The method ID
   * @param archive SaveTaskArchive for serialization
   * @param task_ptr The task to serialize
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters from network transfer (auto-generated)
   * @param method The method ID
   * @param archive LoadTaskArchive for deserialization
   * @return The deserialized task
   */
  hipc::FullPtr<chi::Task> LoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;

  /**
   * Deserialize task for local transfer (auto-generated)
   */
  hipc::FullPtr<chi::Task> LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive) override;

  /**
   * Serialize task for local transfer (auto-generated)
   */
  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task for distributed execution (auto-generated)
   */
  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method,
                                        hipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type (auto-generated)
   */
  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;

  /**
   * Aggregate a replica task into the origin task (auto-generated)
   */
  void Aggregate(chi::u32 method, hipc::FullPtr<chi::Task> origin_task_ptr,
                 hipc::FullPtr<chi::Task> replica_task_ptr) override;
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_RUNTIME_H_