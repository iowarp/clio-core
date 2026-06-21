#include <iostream>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>

/**
 * Simple external ChiMod test that verifies:
 * 1. CLIO Runtime can be found and initialized as a client
 * 2. Admin ChiMod client can be created
 * 3. Basic functionality works through the installed packages
 */
int main() {
  HIPRINT("Testing external ChiMod integration...");

  try {
    // Initialize CLIO Runtime client
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
      HLOG(kError, "Failed to initialize Clio client");
      return 1;
    }
    HIPRINT("Clio client initialized successfully");

    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);
    HIPRINT("Admin client created successfully");

    // Test that we can call basic methods (without actually creating containers)
    // This tests that the linking and symbol resolution is working
    auto pool_query = clio::run::PoolQuery::Local();
    HIPRINT("Pool query created successfully");

    HIPRINT("All external ChiMod integration tests passed!");
    clio::run::CLIO_RUNTIME_FINALIZE();
    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error during external ChiMod test: {}", e.what());
    return 1;
  } catch (...) {
    HLOG(kError, "Unknown error during external ChiMod test");
    return 1;
  }
}