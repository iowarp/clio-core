#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <chimaera/admin/admin_client.h>
#include "chimaera_commands.h"

namespace {
void PrintComposeUsage() {
  HIPRINT("Usage: chimaera compose [--unregister] <compose_config.yaml>");
  HIPRINT("  Loads compose configuration and creates/destroys specified pools");
  HIPRINT("  --unregister: Destroy pools instead of creating them");
  HIPRINT("  Requires runtime to be already initialized");
}
}  // namespace

int Compose(int argc, char** argv) {
  if (argc < 1) {
    PrintComposeUsage();
    return 1;
  }

  bool unregister = false;
  std::string config_path;

  int i = 0;
  while (i < argc) {
    std::string arg(argv[i]);
    if (arg == "--unregister") {
      unregister = true;
      ++i;
    } else if (arg == "--help" || arg == "-h") {
      PrintComposeUsage();
      return 0;
    } else {
      config_path = arg;
      ++i;
    }
  }

  if (config_path.empty()) {
    HLOG(kError, "Missing compose config path");
    PrintComposeUsage();
    return 1;
  }

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera client");
    return 1;
  }

  // RAII guard: call ClientFinalize() on every return path so the background
  // ZMQ receive thread is joined and the DEALER socket is closed before the
  // ZMQ shared-context static destructor runs.  Without this, zmq_ctx_destroy
  // blocks forever because the singleton Chimaera object is heap-allocated
  // (via GetGlobalPtrVar) and its destructor is never invoked by the runtime.
  struct ClientFinalizeGuard {
    ~ClientFinalizeGuard() {
      auto* mgr = CHI_CHIMAERA_MANAGER;
      if (mgr) {
        mgr->ClientFinalize();
      }
    }
  } finalize_guard;

  auto* config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->LoadYaml(config_path)) {
    HLOG(kError, "Failed to load configuration from {}", config_path);
    return 1;
  }

  const auto& compose_config = config_manager->GetComposeConfig();
  if (compose_config.pools_.empty()) {
    HLOG(kError, "No compose section found in configuration");
    return 1;
  }

  HLOG(kInfo, "Found {} pools to {}",
       compose_config.pools_.size(), (unregister ? "destroy" : "create"));

  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  if (unregister) {
    for (const auto& pool_config : compose_config.pools_) {
      HLOG(kInfo, "Destroying pool {} (module: {})",
           pool_config.pool_name_, pool_config.mod_name_);

      auto task = admin_client->AsyncDestroyPool(
          chi::PoolQuery::Dynamic(), pool_config.pool_id_);
      task.Wait();

      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HLOG(kError, "Failed to destroy pool {}, return code: {}",
             pool_config.pool_name_, return_code);
      } else {
        HLOG(kSuccess, "Successfully destroyed pool {}", pool_config.pool_name_);
      }

      namespace fs = std::filesystem;
      std::string restart_file = config_manager->GetConfDir() + "/restart/"
                                 + pool_config.pool_name_ + ".yaml";
      if (fs::exists(restart_file)) {
        fs::remove(restart_file);
        HLOG(kInfo, "Removed restart file: {}", restart_file);
      }
    }

    HLOG(kSuccess, "Unregister completed for {} pools",
         compose_config.pools_.size());
  } else {
    for (const auto& pool_config : compose_config.pools_) {
      HLOG(kInfo, "Creating pool {} (module: {})",
           pool_config.pool_name_, pool_config.mod_name_);

      auto task = admin_client->AsyncCompose(pool_config);
      task.Wait();

      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HLOG(kError, "Failed to create pool {} (module: {}), return code: {}",
             pool_config.pool_name_, pool_config.mod_name_, return_code);
        return 1;
      }

      HLOG(kSuccess, "Successfully created pool {}", pool_config.pool_name_);

      if (pool_config.restart_) {
        namespace fs = std::filesystem;
        std::string restart_dir = config_manager->GetConfDir() + "/restart";
        fs::create_directories(restart_dir);
        std::string restart_file = restart_dir + "/" + pool_config.pool_name_ + ".yaml";

        std::ofstream ofs(restart_file);
        if (ofs.is_open()) {
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
          HLOG(kInfo, "Saved restart config: {}", restart_file);
        } else {
          HLOG(kWarning, "Failed to save restart config: {}", restart_file);
        }
      }
    }

    HLOG(kSuccess, "Compose processing completed successfully - all {} pools created",
         compose_config.pools_.size());
  }
  return 0;
}
