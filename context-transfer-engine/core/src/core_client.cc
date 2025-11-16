#include <chimaera/chimaera.h>
#include <wrp_cte/core/content_transfer_engine.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>

namespace wrp_cte::core {

// Define global pointer variable for CTE client in source file
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(wrp_cte::core::Client, g_cte_client);

bool WRP_CTE_CLIENT_INIT(const std::string &config_path,
                         const chi::PoolQuery &pool_query) {
  // Static guard to prevent double initialization
  static bool s_initialized = false;
  if (s_initialized) {
    return true;  // Already initialized, return success
  }

  // Allocate the global client object if not already allocated
  if (g_cte_client == nullptr) {
    g_cte_client = new wrp_cte::core::Client();
  }

  // config_path is no longer used - configuration now provided via chimaera_compose
  (void)config_path; // Suppress unused parameter warning
  auto *cte_manager = CTE_MANAGER;
  bool result = cte_manager->ClientInit(pool_query);

  // Mark as initialized on success
  if (result) {
    s_initialized = true;
  }

  return result;
}

} // namespace wrp_cte::core