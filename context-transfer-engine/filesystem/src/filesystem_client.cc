/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>

namespace clio::cte::filesystem {

// Process-wide filesystem client singleton (defined inside the namespace so it
// is clio::cte::filesystem::g_fs_client, matching the CLIO_CFS_CLIENT macro).
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_CC(clio::cte::filesystem::Client, g_fs_client);

/**
 * Create-or-bind the default filesystem pool over the default CTE core pool
 * and publish the process-wide client. Mirrors
 * clio::cte::core::CLIO_CTE_CLIENT_INIT / ContentTransferEngine::ClientInit:
 * GetOrCreatePool is idempotent, so this both creates the pool on first call
 * and binds to the existing one if a launcher already composed it.
 */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path,
                          const clio::run::PoolQuery &pool_query) {
  static bool s_initialized = false;
  if (s_initialized) {
    return true;
  }
  (void)config_path;  // configuration now flows through clio compose

  // The filesystem chimod sits over the default CTE core pool, so make sure
  // that exists first (also brings up the runtime client / IPC).
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "CFS ClientInit: failed to initialize the CTE core pool");
    return false;
  }

  // CLIO_CFS_CLIENT lazily allocates the global Client on first access.
  auto *fs_client = CLIO_CFS_CLIENT;
  if (fs_client == nullptr) {
    return false;
  }
  fs_client->Init(kCfsPoolId);

  FilesystemConfig params;
  params.next_pool_id_ = clio::cte::core::kCtePoolId;
  auto create_task =
      fs_client->AsyncCreate(pool_query, kCfsPoolName, kCfsPoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    HLOG(kError, "CFS ClientInit: failed to create filesystem pool '{}' (rc={})",
         kCfsPoolName, create_task->GetReturnCode());
    return false;
  }
  fs_client->pool_id_ = create_task->new_pool_id_;

  s_initialized = true;
  return true;
}

}  // namespace clio::cte::filesystem
