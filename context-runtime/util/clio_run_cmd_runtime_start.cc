#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <clio_ctp/thread/thread_model_manager.h>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/types.h"
#include "clio_run_commands.h"

namespace {
volatile sig_atomic_t g_keep_running = 1;

void SignalHandler(int /*sig*/) {
  g_keep_running = 0;
}

bool InitializeAdminChiMod() {
  HLOG(kDebug, "Initializing admin ChiMod...");

  auto* module_manager = CLIO_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "Module manager not available");
    return false;
  }

  auto* admin_chimod = module_manager->GetChiMod("clio_admin");
  if (!admin_chimod) {
    HLOG(kError, "CRITICAL: Admin ChiMod not found! This is a required system component.");
    return false;
  }

  auto* pool_manager = CLIO_POOL_MANAGER;
  if (!pool_manager) {
    HLOG(kError, "Pool manager not available");
    return false;
  }

  try {
    HLOG(kDebug, "Admin pool creation handled by PoolManager::ServerInit()");

    if (!pool_manager->HasPool(clio::run::kAdminPoolId)) {
      HLOG(kError, "Admin pool creation reported success but pool is not found");
      return false;
    }

    HLOG(kDebug, "Admin ChiPool created successfully (ID: {})", clio::run::kAdminPoolId);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod initialization: {}", e.what());
    return false;
  }
}

void ShutdownAdminChiMod() {
  HLOG(kDebug, "Shutting down admin ChiMod...");

  try {
    auto* pool_manager = CLIO_POOL_MANAGER;
    if (pool_manager && pool_manager->HasPool(clio::run::kAdminPoolId)) {
      if (pool_manager->DestroyLocalPool(clio::run::kAdminPoolId)) {
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
  auto* ipc_manager = CLIO_IPC;
  auto* config = CLIO_CONFIG_MANAGER;
  auto* admin_client = CLIO_ADMIN;

  std::string my_ip = ipc_manager->GetCurrentHostname();
  clio::run::u32 my_port = config->GetPort();

  HLOG(kInfo, "Inducting this node ({}:{}) into the cluster...", my_ip, my_port);

  auto task = admin_client->AsyncAddNode(
      clio::run::PoolQuery::Broadcast(), my_ip, my_port);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    HLOG(kError, "Failed to induct node: {}", task->error_message_.str());
    return false;
  }

  HLOG(kInfo, "Node inducted successfully as node_id={}", task->new_node_id_);
  return true;
}

void PrintRuntimeStartUsage() {
  HIPRINT("Usage: clio runtime start [--induct] [--ephemeral]");
  HIPRINT("  Starts the Clio runtime server");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
  HIPRINT("  --ephemeral: Skip the default compose; start bare (admin only)");
}

void PrintRuntimeRestartUsage() {
  HIPRINT("Usage: clio runtime restart [--induct]");
  HIPRINT("  Restarts the Clio runtime, replaying WAL to recover address table");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
}

}  // namespace

int RuntimeStart(int argc, char* argv[]) {
  bool induct = false;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--induct") == 0) {
      induct = true;
    } else if (std::strcmp(argv[i], "--ephemeral") == 0) {
      // Skip the default compose: start bare (admin only), to be composed
      // explicitly. Communicated to ConfigManager via CLIO_EPHEMERAL, read
      // during the CLIO_INIT below.
#ifdef _WIN32
      _putenv_s("CLIO_EPHEMERAL", "1");
#else
      setenv("CLIO_EPHEMERAL", "1", 1);
#endif
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

  HLOG(kDebug, "Starting Clio runtime...");

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kRuntime, true)) {
    HLOG(kError, "Failed to initialize Clio runtime");
    return 1;
  }

  HLOG(kDebug, "Clio runtime started successfully");

  if (!InitializeAdminChiMod()) {
    HLOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HLOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", clio::run::kAdminPoolId);

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  while (g_keep_running) {
    CTP_THREAD_MODEL->SleepForUs(100000);
  }

  HLOG(kDebug, "Shutting down Clio runtime...");
  ShutdownAdminChiMod();
  HLOG(kDebug, "Clio runtime stopped (finalization will happen automatically)");
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

  HLOG(kInfo, "Restarting Clio runtime (WAL replay enabled)...");

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kRuntime, true,
                           /*is_restart=*/true)) {
    HLOG(kError, "Failed to restart Clio runtime");
    return 1;
  }

  HLOG(kInfo, "Clio runtime restarted successfully");

  if (!InitializeAdminChiMod()) {
    HLOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HLOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", clio::run::kAdminPoolId);

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  while (g_keep_running) {
    CTP_THREAD_MODEL->SleepForUs(100000);
  }

  HLOG(kDebug, "Shutting down Clio runtime...");
  ShutdownAdminChiMod();
  HLOG(kDebug, "Clio runtime stopped (finalization will happen automatically)");
  return 0;
}
