/**
 * Chimaera Compose Utility
 *
 * Loads and processes a compose configuration to create pools
 * Assumes runtime is already initialized
 */

#include <iostream>
#include <string>
#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>

void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " <compose_config.yaml>\n";
  std::cout << "  Loads compose configuration and creates specified pools\n";
  std::cout << "  Requires runtime to be already initialized\n";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string config_path = argv[1];

  // Initialize Chimaera client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "Failed to initialize Chimaera client\n";
    return 1;
  }

  // Load configuration
  auto* config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->LoadYaml(config_path)) {
    std::cerr << "Failed to load configuration from " << config_path << "\n";
    return 1;
  }

  // Get compose configuration
  const auto& compose_config = config_manager->GetComposeConfig();
  if (compose_config.pools_.empty()) {
    std::cerr << "No compose section found in configuration\n";
    return 1;
  }

  std::cout << "Found " << compose_config.pools_.size() << " pools to create\n";

  // Get admin client
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    std::cerr << "Failed to get admin client\n";
    return 1;
  }

  // Process compose
  if (!admin_client->Compose(compose_config)) {
    std::cerr << "Compose processing failed\n";
    return 1;
  }

  std::cout << "Compose processing completed successfully\n";
  return 0;
}
