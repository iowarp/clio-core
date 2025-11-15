#include <wrp_cae/core/constants.h>
#include <wrp_cte/core/core_client.h>  // For WRP_CTE_CLIENT_INIT

// Must include wrp_cae core_client.h last to avoid circular dependency
#include <wrp_cae/core/core_client.h>

// Global CAE client singleton definition
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(wrp_cae::core::Client, g_cae_client);

/**
 * Initialize CAE client singleton
 * Calls WRP_CTE_CLIENT_INIT internally to ensure CTE is initialized
 */
bool WRP_CAE_CLIENT_INIT(const std::string &config_path,
                         const chi::PoolQuery &pool_query) {
  // Note: Default parameters are defined in the header

  // Static guard to prevent multiple initializations
  static bool is_initialized = false;
  if (is_initialized) {
    return true;
  }

  // Suppress unused parameter warning
  (void)config_path;

  // First, ensure CTE client is initialized (CAE depends on CTE)
  if (!wrp_cte::core::WRP_CTE_CLIENT_INIT(config_path, pool_query)) {
    return false;
  }

  // Get or create the CAE client singleton
  auto *cae_client = HSHM_GET_GLOBAL_PTR_VAR(wrp_cae::core::Client, g_cae_client);
  if (!cae_client) {
    return false;
  }

  // Create the CAE pool
  wrp_cae::core::CreateParams params;
  cae_client->Create(
      hipc::MemContext(),
      pool_query,
      "cae_client_pool",
      wrp_cae::core::kCaePoolId,
      params);

  // Mark as initialized
  is_initialized = true;

  return true;
}
