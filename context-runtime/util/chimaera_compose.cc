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
 * Chimaera Compose Utility
 *
 * Loads and processes a compose configuration to create pools
 * Assumes runtime is already initialized
 */

#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>

void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [--unregister] <compose_config.yaml>\n";
  std::cout << "  Loads compose configuration and creates/destroys specified pools\n";
  std::cout << "  --unregister: Destroy pools instead of creating them\n";
  std::cout << "  Requires runtime to be already initialized\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  bool unregister = false;
  std::string config_path;

  // Parse arguments
  int i = 1;
  while (i < argc) {
    std::string arg(argv[i]);
    if (arg == "--unregister") {
      unregister = true;
      ++i;
    } else {
      config_path = arg;
      ++i;
    }
  }

  if (config_path.empty()) {
    std::cerr << "Missing compose config path\n";
    PrintUsage(argv[0]);
    return 1;
  }

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

  std::cout << "Found " << compose_config.pools_.size() << " pools to "
            << (unregister ? "destroy" : "create") << "\n";

  // Get admin client
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    std::cerr << "Failed to get admin client\n";
    return 1;
  }

  if (unregister) {
    // Unregister mode: destroy pools
    for (const auto& pool_config : compose_config.pools_) {
      std::cout << "Destroying pool " << pool_config.pool_name_
                << " (module: " << pool_config.mod_name_ << ")\n";

      auto task = admin_client->AsyncDestroyPool(
          chi::PoolQuery::Dynamic(), pool_config.pool_id_);
      task.Wait();

      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        std::cerr << "Failed to destroy pool " << pool_config.pool_name_
                  << ", return code: " << return_code << "\n";
        // Continue destroying other pools
      } else {
        std::cout << "Successfully destroyed pool " << pool_config.pool_name_ << "\n";
      }

      // Remove restart file if it exists
      namespace fs = std::filesystem;
      std::string restart_file = config_manager->GetConfDir() + "/restart/"
                                 + pool_config.pool_name_ + ".yaml";
      if (fs::exists(restart_file)) {
        fs::remove(restart_file);
        std::cout << "Removed restart file: " << restart_file << "\n";
      }
    }

    std::cout << "Unregister completed for "
              << compose_config.pools_.size() << " pools\n";
  } else {
    // Register mode: create pools
    for (const auto& pool_config : compose_config.pools_) {
      std::cout << "Creating pool " << pool_config.pool_name_
                << " (module: " << pool_config.mod_name_ << ")\n";

      // Create pool asynchronously and wait
      auto task = admin_client->AsyncCompose(pool_config);
      task.Wait();

      // Check return code
      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        std::cerr << "Failed to create pool " << pool_config.pool_name_
                  << " (module: " << pool_config.mod_name_
                  << "), return code: " << return_code << "\n";
        return 1;
      }

      std::cout << "Successfully created pool " << pool_config.pool_name_ << "\n";

      // Save restart config if restart_ flag is set
      if (pool_config.restart_) {
        namespace fs = std::filesystem;
        std::string restart_dir = config_manager->GetConfDir() + "/restart";
        fs::create_directories(restart_dir);
        std::string restart_file = restart_dir + "/" + pool_config.pool_name_ + ".yaml";

        // Write pool config wrapped in compose section so RestartContainers
        // can load it via ConfigManager::LoadYaml (which expects compose: [...])
        std::ofstream ofs(restart_file);
        if (ofs.is_open()) {
          // Indent the pool config under compose: list entry
          std::string indented;
          std::istringstream stream(pool_config.config_);
          std::string line;
          bool first = true;
          while (std::getline(stream, line)) {
            if (first) {
              indented += "  - " + line + "\n";
              first = false;
            } else {
              indented += "    " + line + "\n";
            }
          }
          ofs << "compose:\n" << indented;
          ofs.close();
          std::cout << "Saved restart config: " << restart_file << "\n";
        } else {
          std::cerr << "Warning: Failed to save restart config: " << restart_file << "\n";
        }
      }
    }

    std::cout << "Compose processing completed successfully - all "
              << compose_config.pools_.size() << " pools created\n";
  }
  return 0;
}
