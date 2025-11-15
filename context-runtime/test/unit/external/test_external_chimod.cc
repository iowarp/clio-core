#include <iostream>
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>

/**
 * Simple external ChiMod test that verifies:
 * 1. Chimaera can be found and initialized as a client
 * 2. Admin ChiMod client can be created
 * 3. Basic functionality works through the installed packages
 */
int main() {
  std::cout << "Testing external ChiMod integration..." << std::endl;
  
  try {
    // Initialize Chimaera client
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
      std::cerr << "Failed to initialize Chimaera client" << std::endl;
      return 1;
    }
    std::cout << "✓ Chimaera client initialized successfully" << std::endl;

    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);
    std::cout << "✓ Admin client created successfully" << std::endl;
    
    // Test that we can call basic methods (without actually creating containers)
    // This tests that the linking and symbol resolution is working
    auto pool_query = chi::PoolQuery::Local();
    std::cout << "✓ Pool query created successfully" << std::endl;
    
    std::cout << "All external ChiMod integration tests passed!" << std::endl;
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "Error during external ChiMod test: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error during external ChiMod test" << std::endl;
    return 1;
  }
}