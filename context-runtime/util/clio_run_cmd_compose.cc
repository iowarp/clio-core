#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <clio_runtime/restart_log.h>

#include "clio_run_commands.h"

namespace {
namespace fs = std::filesystem;

void PrintComposeUsage() {
  HIPRINT("Usage: clio_run compose <start|stop|rm|list> [options]");
  HIPRINT("  start <config.yaml>    Create the pools in the compose file.");
  HIPRINT("                         Pools with 'restart: true' register the");
  HIPRINT("                         file in the restart log (~/.clio/restart_log.bin)");
  HIPRINT("                         so it is re-composed on `clio_run start`.");
  HIPRINT("  stop  <config.yaml>    Destroy the pools listed in the compose file.");
  HIPRINT("                         Leaves the restart registration intact.");
  HIPRINT("  rm    <config.yaml>    Stop the pools AND unregister the file from");
  HIPRINT("                         restart. Does NOT delete the compose file.");
  HIPRINT("  list  [--restartable]  List active containers in the local daemon.");
  HIPRINT("                         --restartable: list files registered for restart.");
}

// Resolve a compose-file path to a stable absolute form so the same file
// referenced from different working directories maps to one WAL key.
std::string AbsPath(const std::string& path) {
  std::error_code ec;
  fs::path abs = fs::weakly_canonical(fs::absolute(path, ec), ec);
  if (ec || abs.empty()) {
    return fs::absolute(path).string();
  }
  return abs.string();
}

// RAII: join the ZMQ recv thread / close the DEALER socket on every return
// path (the heap-allocated runtime singleton's dtor never runs otherwise, so
// zmq_ctx_destroy would block forever at exit).
struct ClientFinalizeGuard {
  ~ClientFinalizeGuard() {
    auto* mgr = CLIO_RUNTIME_MANAGER;
    if (mgr != nullptr) {
      mgr->ClientFinalize();
    }
  }
};

bool InitClient() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio runtime client");
    return false;
  }
  return true;
}

// Load a compose file into a local ConfigManager (avoids clobbering the
// process-wide singleton) and copy out its pool list.
bool LoadComposeFile(const std::string& path, clio::run::ComposeConfig* out) {
  clio::run::ConfigManager cfg;
  if (!cfg.LoadYaml(path)) {
    HLOG(kError, "Failed to load compose file: {}", path);
    return false;
  }
  *out = cfg.GetComposeConfig();
  return true;
}

int ComposeStart(const std::string& path) {
  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;

  clio::run::ComposeConfig compose;
  if (!LoadComposeFile(path, &compose)) {
    return 1;
  }
  if (compose.pools_.empty()) {
    HLOG(kError, "No compose section found in {}", path);
    return 1;
  }

  auto* admin = CLIO_ADMIN;
  if (admin == nullptr) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  bool any_restart = false;
  for (const auto& pool_config : compose.pools_) {
    HLOG(kInfo, "Creating pool {} (module: {})", pool_config.pool_name_,
         pool_config.mod_name_);
    auto task = admin->AsyncCompose(pool_config);
    task.Wait();
    if (task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create pool {} (module: {}), return code: {}",
           pool_config.pool_name_, pool_config.mod_name_,
           task->GetReturnCode());
      return 1;
    }
    HLOG(kSuccess, "Successfully created pool {}", pool_config.pool_name_);
    if (pool_config.restart_) {
      any_restart = true;
    }
  }

  // Register the file for restart iff it declares at least one restartable
  // pool. The WAL keys on the absolute compose-file path.
  if (any_restart) {
    clio::run::RestartLog log;
    std::string abs = AbsPath(path);
    if (log.AppendAdd(abs)) {
      log.Compact();
      HLOG(kInfo, "Registered '{}' for restart in {}", abs, log.path());
    } else {
      HLOG(kWarning, "Failed to register '{}' in restart log", abs);
    }
  }

  HLOG(kSuccess, "compose start: all {} pools created", compose.pools_.size());
  return 0;
}

// Destroy every pool listed in a compose file. Used by both stop and rm.
int DestroyComposePools(const std::string& path) {
  clio::run::ComposeConfig compose;
  if (!LoadComposeFile(path, &compose)) {
    return 1;
  }
  auto* admin = CLIO_ADMIN;
  if (admin == nullptr) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }
  for (const auto& pool_config : compose.pools_) {
    HLOG(kInfo, "Stopping pool {} (module: {})", pool_config.pool_name_,
         pool_config.mod_name_);
    auto task =
        admin->AsyncDestroyPool(clio::run::PoolQuery::Dynamic(), pool_config.pool_id_);
    task.Wait();
    if (task->GetReturnCode() != 0) {
      HLOG(kWarning, "Failed to stop pool {}, return code: {}",
           pool_config.pool_name_, task->GetReturnCode());
    } else {
      HLOG(kSuccess, "Stopped pool {}", pool_config.pool_name_);
    }
  }
  return 0;
}

int ComposeStop(const std::string& path) {
  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;
  return DestroyComposePools(path);
}

int ComposeRm(const std::string& path) {
  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;
  int rc = DestroyComposePools(path);

  // Unregister from restart regardless of stop result; a dangling rm with no
  // matching add is dropped by Compact(). The compose file itself is kept.
  clio::run::RestartLog log;
  std::string abs = AbsPath(path);
  if (log.AppendRm(abs)) {
    log.Compact();
    HLOG(kInfo, "Unregistered '{}' from restart", abs);
  } else {
    HLOG(kWarning, "Failed to unregister '{}' from restart log", abs);
  }
  return rc;
}

int ComposeList(bool restartable) {
  if (restartable) {
    // Restartable set comes purely from the WAL — no daemon query needed.
    clio::run::RestartLog log;
    std::vector<std::string> live = log.LiveSet();
    std::cout << "Restartable containers (" << live.size() << "):\n";
    for (const auto& p : live) {
      std::cout << "  " << p << "\n";
    }
    return 0;
  }

  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;
  auto* admin = CLIO_ADMIN;
  if (admin == nullptr) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }
  auto task = admin->AsyncListContainers(clio::run::PoolQuery::Local());
  task.Wait();
  if (task->GetReturnCode() != 0) {
    HLOG(kError, "ListContainers failed, return code: {}",
         task->GetReturnCode());
    return 1;
  }
  std::cout << "Active containers (" << task->pool_names_.size() << "):\n";
  for (size_t i = 0; i < task->pool_names_.size(); ++i) {
    std::cout << "  " << task->pool_names_[i] << "  (pool_id="
              << (i < task->pool_ids_.size() ? task->pool_ids_[i] : "?")
              << ")\n";
  }
  return 0;
}
}  // namespace

int Compose(int argc, char** argv) {
  if (argc < 1) {
    PrintComposeUsage();
    return 1;
  }

  std::string sub = argv[0];
  if (sub == "-h" || sub == "--help") {
    PrintComposeUsage();
    return 0;
  }

  std::vector<std::string> rest;
  rest.reserve(argc);
  for (int i = 1; i < argc; ++i) {
    rest.emplace_back(argv[i]);
  }

  if (sub == "start" || sub == "stop" || sub == "rm") {
    if (rest.empty()) {
      HLOG(kError, "compose {} requires a config file path", sub);
      PrintComposeUsage();
      return 1;
    }
    if (sub == "start") {
      return ComposeStart(rest[0]);
    }
    if (sub == "stop") {
      return ComposeStop(rest[0]);
    }
    return ComposeRm(rest[0]);
  }

  if (sub == "list") {
    bool restartable = false;
    for (const auto& a : rest) {
      if (a == "--restartable") {
        restartable = true;
      }
    }
    return ComposeList(restartable);
  }

  // Backward-compatible fallback: `compose <file>` (== start) and the old
  // `compose --unregister <file>` (== rm). Emits a deprecation warning.
  bool unregister = false;
  std::string path;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--unregister") {
      unregister = true;
    } else if (a == "-h" || a == "--help") {
      PrintComposeUsage();
      return 0;
    } else {
      path = a;
    }
  }
  if (path.empty()) {
    PrintComposeUsage();
    return 1;
  }
  HLOG(kWarning,
       "`clio_run compose <file>` is deprecated; use `clio_run compose start`");
  return unregister ? ComposeRm(path) : ComposeStart(path);
}
