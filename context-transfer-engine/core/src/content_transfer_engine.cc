#include <chimaera/chimaera.h>
#include <wrp_cte/core/content_transfer_engine.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <string>

// Define global pointer variable in source file (outside namespace)
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(wrp_cte::core::ContentTransferEngine,
                              g_cte_manager);

namespace wrp_cte::core {

bool ContentTransferEngine::ClientInit(const chi::PoolQuery &pool_query) {
  // Check for race conditions - if already initialized or initializing
  if (is_initialized_) {
    return true;
  }
  if (is_initializing_) {
    return true;
  }
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager->IsInitializing()) {
    return true;
  }

  // Set initializing flag
  is_initializing_ = true;

  // Initialize Chimaera client
  if (!chi::CHIMAERA_CLIENT_INIT()) {
    is_initializing_ = false;
    return false;
  }

  // Initialize CTE core client
  auto *cte_client = WRP_CTE_CLIENT;

  // Create CreateParams without config - configuration is now provided via chimaera_compose
  CreateParams params;

  // Create CTE Core container using constants from core_tasks.h and specified pool_query
  cte_client->Create(hipc::MemContext(), pool_query,
                     wrp_cte::core::kCtePoolName, wrp_cte::core::kCtePoolId, params);

  // Suppress unused variable warnings
  (void)cte_client;

  // Mark as successfully initialized
  is_initialized_ = true;
  is_initializing_ = false;

  return true;
}

std::vector<std::string> ContentTransferEngine::TagQuery(
    const std::string &tag_re,
    chi::u32 max_tags,
    const chi::PoolQuery &pool_query) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->TagQuery(hipc::MemContext(), tag_re, max_tags, pool_query);
}

std::vector<std::pair<std::string, std::string>> ContentTransferEngine::BlobQuery(
    const std::string &tag_re,
    const std::string &blob_re,
    chi::u32 max_blobs,
    const chi::PoolQuery &pool_query) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->BlobQuery(hipc::MemContext(), tag_re, blob_re, max_blobs, pool_query);
}

} // namespace wrp_cte::core