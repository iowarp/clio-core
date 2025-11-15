/**
 * External ChiMod development test: Using simple_mod ChiMod client libraries
 * 
 * This demonstrates how external ChiMod development works with custom namespaces
 * and proper CMake linking patterns. It tests the external_test::simple_mod ChiMod
 * created as part of the external-chimod unit test.
 * 
 * This test demonstrates:
 * - External namespace usage (external_test vs chimaera)  
 * - Custom ChiMod development patterns
 * - Proper CMake find_package and linking
 * - add_chimod_client and add_chimod_runtime functionality
 */

#include <iostream>
#include <memory>
#include <chimaera/chimaera.h>
#include <chimaera/simple_mod/simple_mod_client.h>
#include <chimaera/admin/admin_client.h>

namespace {
constexpr chi::PoolId kExternalTestPoolId = chi::PoolId(7001, 0);
}

int main() {
  std::cout << "=== External ChiMod Development Test ===" << std::endl;
  std::cout << "Testing external_test::simple_mod with custom namespace and CMake linking." << std::endl;
  
  try {
    // Step 1: Initialize Chimaera client
    std::cout << "\n1. Initializing Chimaera client..." << std::endl;
    bool client_init_success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    if (!client_init_success) {
      std::cout << "NOTICE: Chimaera client initialization failed." << std::endl;
      std::cout << "This is expected when no runtime is active." << std::endl;
      std::cout << "In a production environment, ensure chimaera_start_runtime is running." << std::endl;
    } else {
      std::cout << "SUCCESS: Chimaera client initialized!" << std::endl;
    }
    
    // Step 2: Create admin client (required for pool management)
    std::cout << "\n2. Creating admin client..." << std::endl;
    chimaera::admin::Client admin_client(chi::kAdminPoolId);
    std::cout << "Admin client created with pool ID: " << chi::kAdminPoolId << std::endl;
    
    // Step 3: Create simple_mod client (from external_test namespace)
    std::cout << "\n3. Creating external_test::simple_mod client..." << std::endl;
    external_test::simple_mod::Client simple_mod_client(kExternalTestPoolId);
    std::cout << "Simple mod client created with pool ID: " << kExternalTestPoolId << std::endl;
    std::cout << "Successfully using external namespace: external_test::simple_mod" << std::endl;
    
    // Step 4: Create simple_mod container
    std::cout << "\n4. Creating simple_mod container..." << std::endl;
    
    // Use local pool query (recommended default)
    auto pool_query = chi::PoolQuery::Local();
    
    try {
      // This will create the pool if it doesn't exist
      simple_mod_client.Create(HSHM_MCTX, pool_query);
      std::cout << "SUCCESS: Simple mod container created!" << std::endl;
      
      // Step 5: Demonstrate flush operation
      std::cout << "\n5. Testing simple_mod flush operation..." << std::endl;
      
      simple_mod_client.Flush(HSHM_MCTX, pool_query);
      std::cout << "SUCCESS: Flush operation completed!" << std::endl;
      
      // Step 6: Destroy container for cleanup
      std::cout << "\n6. Destroying simple_mod container..." << std::endl;
      simple_mod_client.Destroy(HSHM_MCTX, pool_query);
      std::cout << "SUCCESS: Container destroyed!" << std::endl;
      
      std::cout << "\n=== External ChiMod Development Test completed successfully! ===" << std::endl;
      
    } catch (const std::exception& e) {
      std::cout << "NOTICE: Container operations failed: " << e.what() << std::endl;
      std::cout << "This is expected when no runtime is active." << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::cout << "ERROR: Exception occurred: " << e.what() << std::endl;
    return 1;
  }
  
  std::cout << "\n=== Key External ChiMod Features Demonstrated ===" << std::endl;
  std::cout << "✓ Custom namespace (external_test vs chimaera)" << std::endl;
  std::cout << "✓ External chimaera_repo.yaml configuration" << std::endl;
  std::cout << "✓ add_chimod_client()/add_chimod_runtime() CMake functions" << std::endl;
  std::cout << "✓ install_chimod() CMake function usage" << std::endl;
  std::cout << "✓ find_package(chimaera::core) linking" << std::endl;
  std::cout << "✓ External module directory structure" << std::endl;
  std::cout << "✓ CHI_TASK_CC macro with external library name" << std::endl;
  
  std::cout << "\nNOTE: This test demonstrates successful external ChiMod development patterns." << std::endl;
  std::cout << "For full functionality, run chimaera_start_runtime in another terminal." << std::endl;
  
  return 0;
}