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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <clio_runtime/pool_manager.h>

#include "clio_run_commands.h"

namespace {

namespace fs = std::filesystem;
using PoolLogEntry = clio::run::PoolManager::PoolLogEntry;

/** Print usage for `clio_run pool`. */
void PrintPoolUsage() {
  HIPRINT("Usage: clio-run pool <start|stop|rm> <pool> [--force]");
  HIPRINT("  Manage one pool of the running runtime; the runtime keeps running.");
  HIPRINT("  start  Re-create a stopped pool from the node pool log.");
  HIPRINT("  stop   Flush the pool's buffered data, then stop it; it stays");
  HIPRINT("         restartable (`pool start` or the next `clio-run start`).");
  HIPRINT("  rm     Stop the pool if it is running, then delete its data and");
  HIPRINT("         forget it. A stopped pool is re-created first so its data");
  HIPRINT("         can be deleted.");
  HIPRINT("  --force  stop/rm: skip the flush (buffered data is discarded).");
  HIPRINT("  <pool> is a pool name or a pool id (major.minor).");
}

/**
 * Parsed `<pool> [--force]` arguments of the pool-scoped commands.
 */
struct PoolArgs {
  std::string pool;          /**< pool name or id text; empty if absent */
  bool force = false;        /**< --force: skip the flush (NoFlush) */
};

/**
 * Parse `<pool> [--force]`. The first non-flag argument is the pool.
 * @param argc argument count for the subcommand
 * @param argv argument vector for the subcommand
 * @return parsed arguments (pool empty when none was given)
 */
PoolArgs ParsePoolArgs(int argc, char** argv) {
  PoolArgs args;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--force") {
      args.force = true;
    } else if (!a.empty() && a[0] != '-' && args.pool.empty()) {
      args.pool = a;
    }
  }
  return args;
}

// RAII: join the ZMQ recv thread / close the DEALER socket on every return
// path, so the CLI process can exit (see ComposeRm for the same pattern).
struct ClientFinalizeGuard {
  ~ClientFinalizeGuard() {
    auto* mgr = CLIO_RUNTIME_MANAGER;
    if (mgr != nullptr) {
      mgr->ClientFinalize();
    }
  }
};

/**
 * Initialize the client side of the runtime for a CLI command.
 * @return true if the client came up; false if no runtime is reachable
 */
bool InitClient() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio runtime client");
    return false;
  }
  return true;
}

/**
 * Whether a pool is active in the local runtime, and its name if so.
 * @param admin initialized admin client
 * @param pool_id pool to look for
 * @param name receives the pool's name when found (left unchanged otherwise)
 * @return true if the local runtime lists the pool as a container
 */
bool IsPoolRunning(clio::run::admin::Client *admin, clio::run::PoolId pool_id,
                   std::string *name) {
  auto task = admin->AsyncListContainers(clio::run::PoolQuery::Local());
  task.Wait();
  if (task->GetReturnCode() != 0) {
    return false;
  }
  const std::string id_text = pool_id.ToString();
  for (size_t i = 0;
       i < task->pool_ids_.size() && i < task->pool_names_.size(); ++i) {
    if (task->pool_ids_[i] == id_text) {
      *name = task->pool_names_[i];
      return true;
    }
  }
  return false;
}

/**
 * Find a pool-log entry under this host's conf dir (every node's log) by pool
 * name, or by pool id when `ref` parses as one.
 * @param ref pool name or pool id text
 * @param out receives the matching entry on success
 * @return true if an entry matched
 */
bool FindInPoolLog(const std::string &ref, PoolLogEntry *out) {
  clio::run::PoolId parsed;
  const bool is_id = clio::run::admin::Client::ParsePoolIdText(ref, &parsed);
  auto *cfg = CLIO_CONFIG_MANAGER;
  cfg->Init();
  const fs::path wal = cfg->GetConfDir() + "/wal";
  std::error_code ec;
  for (const auto &f : fs::directory_iterator(wal, ec)) {
    const std::string name = f.path().filename().string();
    if (name.rfind("pools.", 0) != 0 || f.path().extension() != ".bin") {
      continue;
    }
    for (const auto &e :
         clio::run::PoolManager::ReadPoolLogFile(f.path().string())) {
      if (e.pool_name == ref || (is_id && e.pool_id == parsed)) {
        *out = e;
        return true;
      }
    }
  }
  return false;
}

/**
 * Re-create a pool that is not running from its node pool-log entry. A
 * compose pool is sent through the compose path, so it is created with its
 * own routing (every node it spans), exactly as `compose start` would; a pool
 * created through the API is re-created on this node.
 * @param admin initialized admin client
 * @param entry the pool-log entry to replay
 * @param verb command name for messages ("start" or "rm")
 * @return true if the pool was re-created
 */
bool RecreateFromLog(clio::run::admin::Client *admin, const PoolLogEntry &entry,
                     const char *verb) {
  HLOG(kInfo, "{}: re-creating pool {} from the pool log", verb,
       entry.pool_name);
  clio::run::u32 rc = 0;
  if (entry.compose) {
    clio::run::priv::string raw(CLIO_PRIV_ALLOC, entry.chimod_params);
    auto pool_config = clio::run::Task::Deserialize<clio::run::PoolConfig>(raw);
    pool_config.restart_ = true;
    auto future = admin->AsyncCompose(pool_config);
    future.Wait();
    rc = future->GetReturnCode();
  } else {
    auto future = admin->AsyncRecreatePool(entry);
    future.Wait();
    rc = future->GetReturnCode();
  }
  if (rc != 0) {
    HLOG(kError, "{}: failed to re-create pool {} (rc={})", verb,
         entry.pool_name, rc);
    return false;
  }
  return true;
}

/**
 * Look up a pool for a pool command: resolve it, and report whether it runs.
 * @param admin initialized admin client
 * @param ref pool name or id text
 * @param pool_id receives the pool id (null if unresolved)
 * @param name receives the pool's name (ref when not running)
 * @return true if the pool is running in the local runtime
 */
bool LookupRunning(clio::run::admin::Client *admin, const std::string &ref,
                   clio::run::PoolId *pool_id, std::string *name) {
  *pool_id = admin->ResolvePoolId(ref);
  *name = ref;
  return !pool_id->IsNull() && IsPoolRunning(admin, *pool_id, name);
}

/**
 * `pool start <pool>`: re-create a stopped pool from the node pool log.
 * @param admin initialized admin client
 * @param ref pool name or id text
 * @return 0 on success (or already running), 1 on failure
 */
int PoolStart(clio::run::admin::Client *admin, const std::string &ref) {
  clio::run::PoolId pool_id;
  std::string name;
  if (LookupRunning(admin, ref, &pool_id, &name)) {
    HLOG(kSuccess, "Pool {} ({}) is already running", name,
         pool_id.ToString());
    return 0;
  }
  PoolLogEntry entry;
  if (!FindInPoolLog(ref, &entry)) {
    HLOG(kError, "start: no stopped pool '{}' in a node pool log", ref);
    return 1;
  }
  if (!RecreateFromLog(admin, entry, "start")) {
    return 1;
  }
  HLOG(kSuccess, "Started pool {} ({})", entry.pool_name,
       entry.pool_id.ToString());
  return 0;
}

/**
 * `pool stop <pool> [--force]`: flush (unless forced), then stop the pool and
 * keep it restartable.
 * @param admin initialized admin client
 * @param ref pool name or id text
 * @param force skip the flush
 * @return 0 on success, 1 on failure
 */
int PoolStop(clio::run::admin::Client *admin, const std::string &ref,
             bool force) {
  clio::run::PoolId pool_id;
  std::string name;
  if (!LookupRunning(admin, ref, &pool_id, &name)) {
    HLOG(kError, "stop: no running pool '{}'", ref);
    return 1;
  }
  const clio::run::u32 flags =
      clio::run::admin::kDestroyPoolKeepRestartable |
      (force ? clio::run::admin::kDestroyPoolNoFlush : 0u);
  std::string err;
  clio::run::u64 blobs = 0;
  clio::run::u64 bytes = 0;
  if (admin->UnplugPool(pool_id, flags, &err, &blobs, &bytes) != 0) {
    HLOG(kError, "stop: pool {} not stopped: {}", name, err);
    return 1;
  }
  HLOG(kSuccess, "Stopped pool {} ({}): flushed {} blobs, {} bytes", name,
       pool_id.ToString(), blobs, bytes);
  return 0;
}

/**
 * `pool rm <pool> [--force]`: stop the pool if it runs (flushing unless
 * forced), then delete its data and forget it.
 * @param admin initialized admin client
 * @param ref pool name or id text
 * @param force skip the flush
 * @return 0 on success, 1 on failure
 */
int PoolRemove(clio::run::admin::Client *admin, const std::string &ref,
               bool force) {
  clio::run::PoolId pool_id;
  std::string name;
  if (!LookupRunning(admin, ref, &pool_id, &name)) {
    PoolLogEntry entry;
    if (!FindInPoolLog(ref, &entry)) {
      HLOG(kError, "rm: no pool '{}' is running or in a node pool log", ref);
      return 1;
    }
    if (!RecreateFromLog(admin, entry, "rm")) {
      return 1;
    }
    pool_id = entry.pool_id;
    name = entry.pool_name;
  }
  const clio::run::u32 flags =
      clio::run::admin::kDestroyPoolDeleteData |
      (force ? clio::run::admin::kDestroyPoolNoFlush : 0u);
  std::string err;
  clio::run::u64 blobs = 0;
  clio::run::u64 bytes = 0;
  if (admin->UnplugPool(pool_id, flags, &err, &blobs, &bytes) != 0) {
    HLOG(kError, "rm: pool {} not removed: {}", name, err);
    return 1;
  }
  HLOG(kSuccess, "Removed pool {} ({}): flushed {} blobs, {} bytes; data deleted",
       name, pool_id.ToString(), blobs, bytes);
  return 0;
}

}  // namespace

int Pool(int argc, char **argv) {
  if (argc < 1) {
    PrintPoolUsage();
    return 1;
  }
  const std::string sub = argv[0];
  if (sub == "--help" || sub == "-h") {
    PrintPoolUsage();
    return 0;
  }
  if (sub != "start" && sub != "stop" && sub != "rm") {
    HLOG(kError, "pool: unknown subcommand '{}'", sub);
    PrintPoolUsage();
    return 1;
  }
  const PoolArgs args = ParsePoolArgs(argc - 1, argv + 1);
  if (args.pool.empty()) {
    PrintPoolUsage();
    return 1;
  }
  if (sub == "start" && args.force) {
    HLOG(kWarning, "pool start: --force has no effect");
  }
  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;
  auto *admin = CLIO_ADMIN;
  if (admin == nullptr) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }
  if (sub == "start") return PoolStart(admin, args.pool);
  if (sub == "stop") return PoolStop(admin, args.pool, args.force);
  return PoolRemove(admin, args.pool, args.force);
}
