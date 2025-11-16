/**
 * Chimaera runtime startup utility
 */

#include <chrono>
#include <iostream>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"

namespace {
volatile bool g_keep_running = true;

/**
 * Find and initialize the admin ChiMod
 * Creates a ChiPool for the admin module using PoolManager
 * @return true if successful, false on failure
 */
bool InitializeAdminChiMod() {
  HILOG(kDebug, "Initializing admin ChiMod...");

  // Get the module manager to find the admin chimod
  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HELOG(kError, "Module manager not available");
    return false;
  }

  // Check if admin chimod is available
  auto* admin_chimod = module_manager->GetChiMod("chimaera_admin");
  if (!admin_chimod) {
    HELOG(kError, "CRITICAL: Admin ChiMod not found! This is a required system component.");
    return false;
  }

  // Get the pool manager to register the admin pool
  auto* pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager) {
    HELOG(kError, "Pool manager not available");
    return false;
  }

  try {
    // Use PoolManager to create the admin pool
    // This functionality is now handled by PoolManager::ServerInit()
    // which calls CreatePool internally with proper task and RunContext
    // No need to manually create admin pool here anymore
    HILOG(kDebug, "Admin pool creation handled by PoolManager::ServerInit()");

    // Verify the pool was created successfully
    if (!pool_manager->HasPool(chi::kAdminPoolId)) {
      HELOG(kError, "Admin pool creation reported success but pool is not found");
      return false;
    }

    HILOG(kDebug, "Admin ChiPool created successfully (ID: {})", chi::kAdminPoolId);
    return true;

  } catch (const std::exception& e) {
    HELOG(kError, "Exception during admin ChiMod initialization: {}", e.what());
    return false;
  }
}

/**
 * Shutdown the admin ChiMod properly
 */
void ShutdownAdminChiMod() {
  HILOG(kDebug, "Shutting down admin ChiMod...");

  try {
    // Get the pool manager to destroy the admin pool
    auto* pool_manager = CHI_POOL_MANAGER;
    if (pool_manager && pool_manager->HasPool(chi::kAdminPoolId)) {
      // Use PoolManager to destroy the admin pool locally
      if (pool_manager->DestroyLocalPool(chi::kAdminPoolId)) {
        HILOG(kDebug, "Admin pool destroyed successfully");
      } else {
        HELOG(kError, "Failed to destroy admin pool");
      }
    }

  } catch (const std::exception& e) {
    HELOG(kError, "Exception during admin ChiMod shutdown: {}", e.what());
  }

  HILOG(kDebug, "Admin ChiMod shutdown complete");
}

}  // namespace

int main(int argc, char* argv[]) {
  HILOG(kDebug, "Starting Chimaera runtime...");

  // Initialize Chimaera runtime
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kRuntime, true)) {
    HELOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  HILOG(kDebug, "Chimaera runtime started successfully");

  // Find and initialize admin ChiMod
  if (!InitializeAdminChiMod()) {
    HELOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HILOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", chi::kAdminPoolId);

  // Main runtime loop
  while (g_keep_running) {
    // Sleep for a short period
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  HILOG(kDebug, "Shutting down Chimaera runtime...");

  // Shutdown admin pool first
  ShutdownAdminChiMod();

  HILOG(kDebug, "Chimaera runtime stopped (finalization will happen automatically)");
  return 0;
}