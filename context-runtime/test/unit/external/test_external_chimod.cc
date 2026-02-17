#include <iostream>
#include <hermes_shm/util/logging.h>
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>

/**
 * Simple external ChiMod test that verifies:
 * 1. Chimaera can be found and initialized as a client
 * 2. Admin ChiMod client can be created
 * 3. Basic functionality works through the installed packages
 */
int main() {
  HIPRINT("Testing external ChiMod integration...");

  try {
    // Initialize Chimaera client
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
      HLOG(kError, "Failed to initialize Chimaera client");
      return 1;
    }
    HIPRINT("Chimaera client initialized successfully");

    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);
    HIPRINT("Admin client created successfully");

    // Test that we can call basic methods (without actually creating containers)
    // This tests that the linking and symbol resolution is working
    auto pool_query = chi::PoolQuery::Local();
    HIPRINT("Pool query created successfully");

    HIPRINT("All external ChiMod integration tests passed!");
    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error during external ChiMod test: {}", e.what());
    return 1;
  } catch (...) {
    HLOG(kError, "Unknown error during external ChiMod test");
    return 1;
  }
}