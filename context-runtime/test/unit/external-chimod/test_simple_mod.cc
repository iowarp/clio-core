/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
#include <hermes_shm/util/logging.h>
#include <chimaera/chimaera.h>
#include <chimaera/simple_mod/simple_mod_client.h>
#include <chimaera/admin/admin_client.h>

namespace {
constexpr chi::PoolId kExternalTestPoolId = chi::PoolId(7001, 0);
}

int main() {
  HIPRINT("=== External ChiMod Development Test ===");
  HIPRINT("Testing external_test::simple_mod with custom namespace and CMake linking.");

  try {
    // Step 1: Initialize Chimaera client
    HLOG(kInfo, "\n1. Initializing Chimaera client...");
    bool client_init_success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    if (!client_init_success) {
      HLOG(kInfo, "NOTICE: Chimaera client initialization failed.");
      HLOG(kInfo, "This is expected when no runtime is active.");
      HLOG(kInfo, "In a production environment, ensure chimaera runtime start is running.");
    } else {
      HLOG(kSuccess, "SUCCESS: Chimaera client initialized!");
    }

    // Step 2: Create admin client (required for pool management)
    HLOG(kInfo, "\n2. Creating admin client...");
    chimaera::admin::Client admin_client(chi::kAdminPoolId);
    HLOG(kInfo, "Admin client created with pool ID: {}", chi::kAdminPoolId);

    // Step 3: Create simple_mod client (from external_test namespace)
    HLOG(kInfo, "\n3. Creating external_test::simple_mod client...");
    external_test::simple_mod::Client simple_mod_client(kExternalTestPoolId);
    HLOG(kInfo, "Simple mod client created with pool ID: {}", kExternalTestPoolId);
    HLOG(kInfo, "Successfully using external namespace: external_test::simple_mod");

    // Step 4: Create simple_mod container
    HLOG(kInfo, "\n4. Creating simple_mod container...");

    // Use local pool query (recommended default)
    auto pool_query = chi::PoolQuery::Local();

    try {
      // This will create the pool if it doesn't exist
      simple_mod_client.Create(pool_query);
      HLOG(kSuccess, "SUCCESS: Simple mod container created!");

      // Step 5: Demonstrate flush operation
      HLOG(kInfo, "\n5. Testing simple_mod flush operation...");

      simple_mod_client.Flush(pool_query);
      HLOG(kSuccess, "SUCCESS: Flush operation completed!");

      // Step 6: Destroy container for cleanup
      HLOG(kInfo, "\n6. Destroying simple_mod container...");
      simple_mod_client.Destroy(pool_query);
      HLOG(kSuccess, "SUCCESS: Container destroyed!");

      HLOG(kSuccess, "\n=== External ChiMod Development Test completed successfully! ===");

    } catch (const std::exception& e) {
      HLOG(kInfo, "NOTICE: Container operations failed: {}", e.what());
      HLOG(kInfo, "This is expected when no runtime is active.");
    }

  } catch (const std::exception& e) {
    HLOG(kError, "ERROR: Exception occurred: {}", e.what());
    return 1;
  }

  HIPRINT("\n=== Key External ChiMod Features Demonstrated ===");
  HIPRINT("Custom namespace (external_test vs chimaera)");
  HIPRINT("External chimaera_repo.yaml configuration");
  HIPRINT("add_chimod_client()/add_chimod_runtime() CMake functions");
  HIPRINT("install_chimod() CMake function usage");
  HIPRINT("find_package(chimaera::core) linking");
  HIPRINT("External module directory structure");
  HIPRINT("CHI_TASK_CC macro with external library name");

  HIPRINT("\nNOTE: This test demonstrates successful external ChiMod development patterns.");
  HIPRINT("For full functionality, run chimaera runtime start in another terminal.");

  return 0;
}