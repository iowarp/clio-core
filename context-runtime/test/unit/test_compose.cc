/**
 * Unit tests for Compose feature
 */

#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <fstream>

#include "../simple_test.h"

/**
 * Create a temporary compose configuration file
 */
std::string CreateComposeConfig() {
    std::string config_path = "/tmp/test_compose_config.yaml";
    std::ofstream config_file(config_path);

    config_file << "# Test compose configuration\n";
    config_file << "workers:\n";
    config_file << "  sched_threads: 2\n";
    config_file << "  slow_threads: 2\n";
    config_file << "\n";
    config_file << "memory:\n";
    config_file << "  main_segment_size: 1GB\n";
    config_file << "  client_data_segment_size: 256MB\n";
    config_file << "  runtime_data_segment_size: 256MB\n";
    config_file << "\n";
    config_file << "networking:\n";
    config_file << "  port: 5555\n";
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

  std::cout << "PoolId::FromString tests passed\n";
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

  std::cout << "PoolQuery::FromString tests passed\n";
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

  // Verify compose section was parsed
  REQUIRE(compose_config.pools_.size() == 1);

  // Verify pool configuration
  const auto& pool_config = compose_config.pools_[0];
  REQUIRE(pool_config.mod_name_ == "chimaera_bdev");
  REQUIRE(pool_config.pool_name_ == "/tmp/test_bdev.dat");
  REQUIRE(pool_config.pool_id_.major_ == 200);
  REQUIRE(pool_config.pool_id_.minor_ == 0);
  REQUIRE(pool_config.pool_query_.IsDynamicMode());

  std::cout << "Parse compose config test passed\n";
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

  // Call Compose
  REQUIRE(admin_client->Compose(compose_config));

  // Verify pool was created by checking if we can access it
  chi::PoolId bdev_pool_id(200, 0);
  chimaera::bdev::Client bdev_client(bdev_pool_id);

  // Try to allocate blocks to verify the pool exists and is functional
  auto blocks = bdev_client.AllocateBlocks(HSHM_MCTX, chi::PoolQuery::Local(), 1024);

  REQUIRE(bdev_client.GetReturnCode() == 0);
  REQUIRE(blocks.size() > 0);

  std::cout << "Admin client Compose test passed\n";
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // Initialize runtime
  std::cout << "Initializing Chimaera runtime...\n";
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    std::cerr << "Failed to initialize Chimaera runtime\n";
    return 1;
  }

  // Run tests using simple_test framework
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  return SimpleTest::run_all_tests(filter);
}
