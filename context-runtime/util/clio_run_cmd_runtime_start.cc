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
      // HasPool answers false for two very different states -- "the pool
      // manager is not initialized" and "the pool genuinely is not there" --
      // and the old message could not tell them apart. This check has fired
      // intermittently right after ServerInit reported the pool created,
      // killing the daemon at startup (issue #924, seen as a cr_detached_spawn
      // failure on macos-14). Say which state it is, so the next occurrence
      // answers the question instead of posing it.
      HLOG(kError,
           "Admin pool creation reported success but pool is not found "
           "(pool_manager initialized={}, pools known={}). {}",
           pool_manager->IsInitialized(), pool_manager->GetPoolCount(),
           pool_manager->IsInitialized()
               ? "The manager is up, so the admin pool's metadata is genuinely "
                 "absent -- it was inserted and lost, or never durably inserted."
               : "The manager reports NOT initialized, so this is an ordering "
                 "problem in startup, not a missing pool.");
      return false;
    }

    HLOG(kDebug, "Admin ChiPool created successfully (ID: {})", clio::run::kAdminPoolId);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod initialization: {}", e.what());
    return false;
  }
}

// NOTE: there is deliberately no ShutdownAdminChiMod() here any more.
//
// This used to call PoolManager::DestroyLocalPool(kAdminPoolId) right after
// RunUntilStopped() returned — i.e. while every worker was still running. The
// admin container owns the periodic service pumps (kSend=14, kClientSend=21,
// kHeartbeatProbe=28, kSystemMonitor=31); once its container is gone, each
// reschedule routes to a pool that no longer exists, RouteTask gets
// RouteResult::Dne, and Worker::AddToRetryQueue parks it. ProcessRetryQueue
// then re-routes it, gets Dne again, and re-parks it: an unbounded spin that
// DrainPendingTasks() can never drain, so the RequestStop watchdog force-exits
// the daemon with code 2 instead of a clean 0.
//
// The race was always here but the window was microseconds wide (destroy →
// return → atexit → StopWorkers), so nothing ever landed in it. Issue #990's
// dashboard put VizServer::Stop() at the top of ServerFinalize(), between the
// destroy and StopWorkers(); stopAll(true) drains in-flight HTTP requests and
// takes real time, holding the window open long enough for the pumps to fire.
// That is what broke cr_shutdown_bt_churn and
// cte_replication_persist_integration on 2026-08-19.
//
// Destroying the admin pool early was never necessary: ServerFinalize() calls
// PoolManager::DestroyAllContainers() after StopWorkers(), and that walks all
// of pool_metadata_ — admin included. Letting admin live exactly as long as
// every other pool restores the invariant that no container is torn down while
// a worker can still route to it.

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

/**
 * Block until the runtime should exit: either a SIGTERM/SIGINT flipped
 * g_keep_running, or a StopRuntimeTask requested a graceful stop via
 * RuntimeManager::RequestStop. Returning lets main() exit normally so the
 * atexit teardown (ServerFinalize) runs on this main thread.
 */
void RunUntilStopped() {
  auto* runtime_manager = CLIO_RUNTIME_MANAGER;
  while (g_keep_running &&
         !(runtime_manager && runtime_manager->IsStopRequested())) {
    CTP_THREAD_MODEL->SleepForUs(100000);
  }
  // Signal-triggered exit (SIGTERM/SIGINT): arm the same teardown watchdog
  // the task-driven stop uses, so a wedged ServerFinalize can never leave a
  // half-dead daemon behind (`docker stop` / Jarvis Kill rely on SIGTERM).
  // No-op if the stop was already requested through RequestStop.
  if (runtime_manager && !runtime_manager->IsStopRequested()) {
    runtime_manager->RequestStop(
        clio::run::RuntimeManager::StopMode::kGraceful, 0);
  }
}

void PrintVizUsage() {
  HIPRINT("  --viz / --no-viz: Serve (or don't serve) the web dashboard on this");
  HIPRINT("      node. Served by default; a `viz: enabled:` key in the config or");
  HIPRINT("      CLIO_VIZ_ENABLE in the environment wins over the default.");
  HIPRINT("  --viz-port <port>: Dashboard TCP port (default 8080, 0 = pick a");
  HIPRINT("      free one). Also CLIO_VIZ_PORT / `viz: port:`.");
  HIPRINT("  --viz-bind <addr>: Dashboard bind address (default 127.0.0.1).");
  HIPRINT("      Also CLIO_VIZ_BIND / `viz: bind:`.");
}

void PrintRuntimeStartUsage() {
  HIPRINT("Usage: clio runtime start [--induct] [--ephemeral] [viz options]");
  HIPRINT("  Starts the Clio runtime server");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
  HIPRINT("  --ephemeral: Skip the default compose; start bare (admin only)");
  PrintVizUsage();
}

void PrintRuntimeRestartUsage() {
  HIPRINT("Usage: clio runtime restart [--induct] [viz options]");
  HIPRINT("  Restarts the Clio runtime, replaying WAL to recover address table");
  HIPRINT("  --induct: Register this node with all existing cluster nodes");
  PrintVizUsage();
}

/** Set an environment variable portably. The CLI communicates with
 *  ConfigManager through the environment (as --ephemeral already did) because
 *  the manager does not exist yet at argument-parsing time, and because
 *  environment overrides are re-applied after every config (re)load, so the
 *  setting survives one. */
void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

/** Outcome of looking at one argument through the viz flags' eyes. */
enum class VizArg {
  kNotMine,   ///< not a viz flag; the caller should handle it
  kConsumed,  ///< consumed (i advanced past any value)
  kBadValue,  ///< a viz flag, but malformed
  // NOTE: not kError -- logging.h #defines kError as a log level, so it cannot
  // be used as an identifier anywhere in this tree.
};

/**
 * Parse one argument as a viz flag.
 *
 * The values are pushed into the environment rather than into ConfigManager
 * because ConfigManager does not exist yet (this runs before CLIO_INIT) and
 * because environment overrides are re-applied after every config load -- the
 * same reason --ephemeral uses CLIO_EPHEMERAL.
 */
VizArg ParseVizArg(int argc, char* argv[], int& i) {
  const std::string arg = argv[i];
  if (arg == "--no-viz") {
    SetEnv("CLIO_VIZ_ENABLE", "0");
    return VizArg::kConsumed;
  }
  if (arg == "--viz") {
    SetEnv("CLIO_VIZ_ENABLE", "1");
    return VizArg::kConsumed;
  }
  if (arg == "--viz-port" || arg == "--viz-bind") {
    if (i + 1 >= argc) {
      HLOG(kError, "{} requires a value", arg);
      return VizArg::kBadValue;
    }
    SetEnv(arg == "--viz-port" ? "CLIO_VIZ_PORT" : "CLIO_VIZ_BIND", argv[i + 1]);
    // Naming an endpoint means the operator wants the dashboard.
    SetEnv("CLIO_VIZ_ENABLE", "1");
    ++i;
    return VizArg::kConsumed;
  }
  return VizArg::kNotMine;
}

/**
 * Serve the dashboard for a daemon whose operator neither asked for it nor
 * refused it.
 *
 * A real daemon gets the dashboard by default; an embedded runtime (a unit test,
 * an adapter, a library user's CLIO_INIT) does not -- which is why the default
 * lives here in the daemon CLI instead of in ConfigManager, whose default is
 * off. An explicit `viz: enabled:` key or CLIO_VIZ_ENABLE always wins, and this
 * has to run after CLIO_INIT to know whether one of them spoke, by which point
 * the admin container has already started the dashboard if it was enabled --
 * hence the IsRunning() check rather than an unconditional Start().
 */
void EnableVizForDaemon() {
  auto* config = CLIO_CONFIG_MANAGER;
  if (!config) {
    return;
  }
  config->SetVizEnabledDefault(true);
  auto* viz = CLIO_VIZ;
  if (viz && !viz->IsRunning()) {
    viz->Start();
  }
}

}  // namespace

int RuntimeStart(int argc, char* argv[]) {
  bool induct = false;
  for (int i = 0; i < argc; ++i) {
    VizArg viz_arg = ParseVizArg(argc, argv, i);
    if (viz_arg == VizArg::kBadValue) {
      PrintRuntimeStartUsage();
      return 1;
    }
    if (viz_arg == VizArg::kConsumed) {
      continue;
    }
    if (std::strcmp(argv[i], "--induct") == 0) {
      induct = true;
    } else if (std::strcmp(argv[i], "--ephemeral") == 0) {
      // Skip the default compose: start bare (admin only), to be composed
      // explicitly. Communicated to ConfigManager via CLIO_EPHEMERAL, read
      // during the CLIO_INIT below.
      SetEnv("CLIO_EPHEMERAL", "1");
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

  EnableVizForDaemon();

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  RunUntilStopped();

  HLOG(kDebug, "Shutting down Clio runtime...");
  // The admin pool is NOT destroyed here — ServerFinalize()'s
  // DestroyAllContainers() does it after StopWorkers(). See the
  // "no ShutdownAdminChiMod()" note near the top of this file for why tearing
  // it down while the workers still run wedges the shutdown.
  HLOG(kDebug, "Clio runtime stopped (finalization will happen automatically)");
  return 0;
}

int RuntimeRestart(int argc, char* argv[]) {
  bool induct = false;
  for (int i = 0; i < argc; ++i) {
    VizArg viz_arg = ParseVizArg(argc, argv, i);
    if (viz_arg == VizArg::kBadValue) {
      PrintRuntimeRestartUsage();
      return 1;
    }
    if (viz_arg == VizArg::kConsumed) {
      continue;
    }
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

  EnableVizForDaemon();

  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  RunUntilStopped();

  HLOG(kDebug, "Shutting down Clio runtime...");
  // The admin pool is NOT destroyed here — ServerFinalize()'s
  // DestroyAllContainers() does it after StopWorkers(). See the
  // "no ShutdownAdminChiMod()" note near the top of this file for why tearing
  // it down while the workers still run wedges the shutdown.
  HLOG(kDebug, "Clio runtime stopped (finalization will happen automatically)");
  return 0;
}
