#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/config_manager.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"
#include "chimaera_commands.h"

namespace {
volatile sig_atomic_t g_keep_running = 1;

void SignalHandler(int /*sig*/) {
  g_keep_running = 0;
}

bool InitializeAdminChiMod() {
  HLOG(kDebug, "Initializing admin ChiMod...");

  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "Module manager not available");
    return false;
  }

  auto* admin_chimod = module_manager->GetChiMod("chimaera_admin");
  if (!admin_chimod) {
    HLOG(kError, "CRITICAL: Admin ChiMod not found! This is a required system component.");
    return false;
  }

  auto* pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager) {
    HLOG(kError, "Pool manager not available");
    return false;
  }

  try {
    HLOG(kDebug, "Admin pool creation handled by PoolManager::ServerInit()");

    if (!pool_manager->HasPool(chi::kAdminPoolId)) {
      HLOG(kError, "Admin pool creation reported success but pool is not found");
      return false;
    }

    HLOG(kDebug, "Admin ChiPool created successfully (ID: {})", chi::kAdminPoolId);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod initialization: {}", e.what());
    return false;
  }
}

void ShutdownAdminChiMod() {
  HLOG(kDebug, "Shutting down admin ChiMod...");

  try {
    auto* pool_manager = CHI_POOL_MANAGER;
    if (pool_manager && pool_manager->HasPool(chi::kAdminPoolId)) {
      if (pool_manager->DestroyLocalPool(chi::kAdminPoolId)) {
        HLOG(kDebug, "Admin pool destroyed successfully");
      } else {
        HLOG(kError, "Failed to destroy admin pool");
      }
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod shutdown: {}", e.what());
  }

  HLOG(kDebug, "Admin ChiMod shutdown complete");
}

bool InductNode() {
  auto* ipc_manager = CHI_IPC;
  auto* config = CHI_CONFIG_MANAGER;
  auto* admin_client = CHI_ADMIN;

  std::string my_ip = ipc_manager->GetCurrentHostname();
  chi::u32 my_port = config->GetPort();

  HLOG(kInfo, "Inducting this node ({}:{}) into the cluster...", my_ip, my_port);

  auto task = admin_client->AsyncAddNode(
      chi::PoolQuery::Broadcast(), my_ip, my_port);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    HLOG(kError, "Failed to induct node: {}", task->error_message_.str());
    return false;
  }

  HLOG(kInfo, "Node inducted successfully as node_id={}", task->new_node_id_);
  return true;
}

void PrintRuntimeStartUsage() {
  HIPRINT("Usage: chimaera runtime start [--induct]");
  HIPRINT("  Starts the Chimaera runtime server");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
}

void PrintRuntimeRestartUsage() {
  HIPRINT("Usage: chimaera runtime restart [--induct]");
  HIPRINT("  Restarts the Chimaera runtime, replaying WAL to recover address table");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
}

}  // namespace

int RuntimeStart(int argc, char* argv[]) {
  bool induct = false;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--induct") == 0) {
      induct = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      PrintRuntimeStartUsage();
      return 0;
    } else {
      HLOG(kError, "Unknown argument: {}", argv[i]);
      PrintRuntimeStartUsage();
      return 1;
    }
  }

  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);

  HLOG(kDebug, "Starting Chimaera runtime...");

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kRuntime, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  HLOG(kDebug, "Chimaera runtime started successfully");

  if (!InitializeAdminChiMod()) {
    HLOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HLOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", chi::kAdminPoolId);

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  while (g_keep_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  HLOG(kDebug, "Shutting down Chimaera runtime...");
  ShutdownAdminChiMod();
  HLOG(kDebug, "Chimaera runtime stopped (finalization will happen automatically)");
  return 0;
}

int RuntimeRestart(int argc, char* argv[]) {
  bool induct = false;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--induct") == 0) {
      induct = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      PrintRuntimeRestartUsage();
      return 0;
    } else {
      HLOG(kError, "Unknown argument: {}", argv[i]);
      PrintRuntimeRestartUsage();
      return 1;
    }
  }

  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);

  HLOG(kInfo, "Restarting Chimaera runtime (WAL replay enabled)...");

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kRuntime, true,
                           /*is_restart=*/true)) {
    HLOG(kError, "Failed to restart Chimaera runtime");
    return 1;
  }

  HLOG(kInfo, "Chimaera runtime restarted successfully");

  if (!InitializeAdminChiMod()) {
    HLOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HLOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", chi::kAdminPoolId);

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  while (g_keep_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  HLOG(kDebug, "Shutting down Chimaera runtime...");
  ShutdownAdminChiMod();
  HLOG(kDebug, "Chimaera runtime stopped (finalization will happen automatically)");
  return 0;
}
