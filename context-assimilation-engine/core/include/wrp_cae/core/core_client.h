#ifndef WRP_CAE_CORE_CLIENT_H_
#define WRP_CAE_CORE_CLIENT_H_

#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_tasks.h>

namespace wrp_cae::core {

class Client : public chi::ContainerClient {
 public:
  Client() = default;
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Synchronous Create - waits for completion
   */
  void Create(const hipc::MemContext& mctx,
              const chi::PoolQuery& pool_query,
              const std::string& pool_name,
              const chi::PoolId& custom_pool_id,
              const CreateParams& params = CreateParams()) {
    auto task = AsyncCreate(mctx, pool_query, pool_name, custom_pool_id, params);
    task->Wait();

    // CRITICAL: Update client pool_id_ with the actual pool ID from the task
    pool_id_ = task->new_pool_id_;

    CHI_IPC->DelTask(task);
  }

  /**
   * Asynchronous Create - returns immediately
   */
  hipc::FullPtr<CreateTask> AsyncCreate(
      const hipc::MemContext& mctx,
      const chi::PoolQuery& pool_query,
      const std::string& pool_name,
      const chi::PoolId& custom_pool_id,
      const CreateParams& params = CreateParams()) {
    auto* ipc_manager = CHI_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,                       // Pool name
        custom_pool_id,                  // Target pool ID
        params);                         // CreateParams with configuration

    // Submit to runtime
    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Synchronous ParseOmni - Parse OMNI YAML file and schedule assimilation tasks
   * Accepts vector of AssimilationCtx and serializes it transparently
   */
  chi::u32 ParseOmni(const hipc::MemContext& mctx,
                     const std::vector<AssimilationCtx>& contexts,
                     chi::u32& num_tasks_scheduled) {
    auto task = AsyncParseOmni(mctx, contexts);
    task->Wait();

    num_tasks_scheduled = task->num_tasks_scheduled_;
    chi::u32 result = task->result_code_;

    CHI_IPC->DelTask(task);
    return result;
  }

  /**
   * Asynchronous ParseOmni - returns immediately
   * Accepts vector of AssimilationCtx and serializes it transparently in the task constructor
   */
  hipc::FullPtr<ParseOmniTask> AsyncParseOmni(
      const hipc::MemContext& mctx,
      const std::vector<AssimilationCtx>& contexts) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ParseOmniTask>(
        chi::CreateTaskId(),
        pool_id_,
        chi::PoolQuery::Local(),
        contexts);

    ipc_manager->Enqueue(task);
    return task;
  }

};

}  // namespace wrp_cae::core

// Global pointer-based singleton for CAE client with lazy initialization
HSHM_DEFINE_GLOBAL_PTR_VAR_H(wrp_cae::core::Client, g_cae_client);

/**
 * Initialize CAE client singleton
 * Calls WRP_CTE_CLIENT_INIT internally to ensure CTE is initialized
 * Creates and initializes a global CAE client singleton
 *
 * @param config_path Path to configuration file (optional)
 * @param pool_query Pool query for CAE pool creation (default: Dynamic)
 * @return true on success, false on failure
 */
bool WRP_CAE_CLIENT_INIT(const std::string &config_path = "",
                         const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

/**
 * Global CAE client singleton accessor macro
 * Returns pointer to the global CAE client instance
 */
#define WRP_CAE_CLIENT                                                         \
  (&(*HSHM_GET_GLOBAL_PTR_VAR(wrp_cae::core::Client,                         \
                              g_cae_client)))

#endif  // WRP_CAE_CORE_CLIENT_H_
