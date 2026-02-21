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
 * Unit tests for Compose feature
 */

#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <fstream>
#include <hermes_shm/util/logging.h>

#include "simple_test.h"

/**
 * Create a temporary compose configuration file
 */
std::string CreateComposeConfig() {
    std::string config_path = "/tmp/test_compose_config.yaml";
    std::ofstream config_file(config_path);

    config_file << "# Test compose configuration\n";
    config_file << "runtime:\n";
    config_file << "  num_threads: 4\n";
    config_file << "\n";
    config_file << "networking:\n";
    config_file << "  port: 9413\n";
    config_file << "\n";
    config_file << "compose:\n";
    config_file << "- mod_name: chimaera_bdev\n";
    config_file << "  pool_name: /tmp/test_bdev.dat\n";
    config_file << "  pool_query: dynamic\n";
    config_file << "  pool_id: 200.0\n";
    config_file << "  capacity: 10MB\n";
    config_file << "  bdev_type: file\n";
    config_file << "  io_depth: 16\n";
    config_file << "  alignment: 4096\n";

    config_file.close();
    return config_path;
}

/**
 * Test PoolId::FromString parsing
 */
TEST_CASE("PoolId::FromString parsing", "[compose]") {
  // Test valid pool ID formats
  chi::PoolId pool_id1 = chi::PoolId::FromString("200.0");
  REQUIRE(pool_id1.major_ == 200);
  REQUIRE(pool_id1.minor_ == 0);

  chi::PoolId pool_id2 = chi::PoolId::FromString("123.456");
  REQUIRE(pool_id2.major_ == 123);
  REQUIRE(pool_id2.minor_ == 456);

  HIPRINT("PoolId::FromString tests passed");
}

/**
 * Test PoolQuery::FromString parsing
 */
TEST_CASE("PoolQuery::FromString parsing", "[compose]") {
  // Test "local"
  chi::PoolQuery query1 = chi::PoolQuery::FromString("local");
  REQUIRE(query1.IsLocalMode());

  // Test "dynamic"
  chi::PoolQuery query2 = chi::PoolQuery::FromString("dynamic");
  REQUIRE(query2.IsDynamicMode());

  // Test case insensitive
  chi::PoolQuery query3 = chi::PoolQuery::FromString("LOCAL");
  REQUIRE(query3.IsLocalMode());

  chi::PoolQuery query4 = chi::PoolQuery::FromString("Dynamic");
  REQUIRE(query4.IsDynamicMode());

  HIPRINT("PoolQuery::FromString tests passed");
}

/**
 * Test basic compose configuration parsing
 */
TEST_CASE("Parse compose configuration", "[compose]") {
  // Create config file
  std::string config_path = CreateComposeConfig();

  // Load configuration
  auto* config_manager = CHI_CONFIG_MANAGER;
  REQUIRE(config_manager != nullptr);

  REQUIRE(config_manager->LoadYaml(config_path));

  // Get compose config
  const auto& compose_config = config_manager->GetComposeConfig();

  // Verify compose section was parsed - at least 1 pool should exist
  // (there may be more from server initialization)
  REQUIRE(compose_config.pools_.size() >= 1);

  // Find our test pool configuration (chimaera_bdev with pool_name /tmp/test_bdev.dat)
  bool found_test_pool = false;
  for (const auto& pool_config : compose_config.pools_) {
    if (pool_config.mod_name_ == "chimaera_bdev" &&
        pool_config.pool_name_ == "/tmp/test_bdev.dat") {
      // Verify pool configuration
      REQUIRE(pool_config.pool_id_.major_ == 200);
      REQUIRE(pool_config.pool_id_.minor_ == 0);
      REQUIRE(pool_config.pool_query_.IsDynamicMode());
      found_test_pool = true;
      break;
    }
  }
  REQUIRE(found_test_pool);

  HIPRINT("Parse compose config test passed");
}

/**
 * Test admin client Compose method
 */
TEST_CASE("Admin client Compose method", "[compose]") {
  // Create config file
  std::string config_path = CreateComposeConfig();

  // Load configuration
  auto* config_manager = CHI_CONFIG_MANAGER;
  REQUIRE(config_manager != nullptr);

  REQUIRE(config_manager->LoadYaml(config_path));

  // Get admin client
  auto* admin_client = CHI_ADMIN;
  REQUIRE(admin_client != nullptr);

  // Get compose config
  const auto& compose_config = config_manager->GetComposeConfig();
  REQUIRE(!compose_config.pools_.empty());

  // Call AsyncCompose for each pool - tasks auto-freed when Future goes out of scope
  for (const auto& pool_config : compose_config.pools_) {
    auto compose_task = admin_client->AsyncCompose(pool_config);
    compose_task.Wait();
    REQUIRE(compose_task->GetReturnCode() == 0);
    // Task automatically freed when compose_task goes out of scope
  }

  // Verify pool was created by checking if we can access it
  chi::PoolId bdev_pool_id(200, 0);
  chimaera::bdev::Client bdev_client(bdev_pool_id);

  // Try to allocate blocks to verify the pool exists and is functional
  auto alloc_task = bdev_client.AsyncAllocateBlocks(chi::PoolQuery::Local(), 1024);
  alloc_task.Wait();
  std::vector<chimaera::bdev::Block> blocks;
  for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
    blocks.push_back(alloc_task->blocks_[i]);
  }
  REQUIRE(alloc_task->GetReturnCode() == 0);
  // Task automatically freed when alloc_task goes out of scope

  REQUIRE(blocks.size() > 0);

  HIPRINT("Admin client Compose test passed");
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // Initialize runtime
  HIPRINT("Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  // Run tests using simple_test framework
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  return SimpleTest::run_all_tests(filter);
}
