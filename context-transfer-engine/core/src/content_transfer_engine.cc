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

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// Define global pointer variable in source file (outside namespace)
CTP_DEFINE_GLOBAL_PTR_VAR_CC(clio::cte::core::ContentTransferEngine,
                              g_cte_manager);

namespace clio::cte::core {

bool ContentTransferEngine::ClientInit(const clio::run::PoolQuery &pool_query) {
  // Check for race conditions - if already initialized or initializing
  if (is_initialized_) {
    return true;
  }
  if (is_initializing_) {
    return true;
  }
  auto *runtime_manager = CLIO_RUNTIME_MANAGER;
  if (runtime_manager->IsInitializing()) {
    return true;
  }

  // Set initializing flag
  is_initializing_ = true;

  // Initialize CLIO Runtime client
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    is_initializing_ = false;
    return false;
  }

  // Initialize CTE core client
  auto *cte_client = CLIO_CTE_CLIENT;

  // CLIO_CTE_POOL=major.minor (issue #886): bind this process's CTE client
  // singleton to an INTERPOSING pool — e.g. the replication chimod at 561.0,
  // which speaks the core task interface and forwards every method it does
  // not override. Pure configuration: no caller changes, no create (the
  // interposer is composed, and creating it needs its own params anyway).
  // Deliberately ONLY here, never in Client::Init/ctor — runtime-internal
  // module clients construct Client(next_pool_id) to reach the core, and
  // redirecting those would make an interposer forward to itself.
  {
    const char *pool_env = std::getenv("CLIO_CTE_POOL");
    if (pool_env != nullptr && *pool_env != '\0') {
      unsigned major = 0, minor = 0;
      if (std::sscanf(pool_env, "%u.%u", &major, &minor) == 2) {
        cte_client->Init(clio::run::PoolId(major, minor));
        // Zero-IPC reads through an interposer: the interposer pool
        // publishes no mirror of its own, so fall back to the CANONICAL
        // core pool's mirror — the pool every shipped interposer sits over.
        // Correct because interposed writes update the primary
        // synchronously and drops re-mirror empty (see AttachShmCacheOf).
        if (cte_client->AttachShmCache()) {
          HLOG(kInfo, "CTE ClientInit: SHM fast path via pool {}.{}'s mirror",
               major, minor);
        } else if (cte_client->AttachShmCacheOf(clio::cte::core::kCtePoolId)) {
          HLOG(kInfo,
               "CTE ClientInit: SHM fast path via the core pool's mirror");
        } else {
          HLOG(kDebug,
               "CTE ClientInit: SHM metadata cache unavailable; using RPC");
        }
        HLOG(kInfo, "CTE ClientInit: CLIO_CTE_POOL bound client to pool {}.{}",
             major, minor);
        is_initialized_ = true;
        is_initializing_ = false;
        return true;
      }
      HLOG(kError, "CTE ClientInit: ignoring malformed CLIO_CTE_POOL '{}'",
           pool_env);
    }
  }

  // Create CreateParams without config - configuration is now provided via clio compose
  CreateParams params;

  // Create CTE Core container using constants from core_tasks.h and specified pool_query
  auto create_task = cte_client->AsyncCreate(pool_query,
                                              clio::cte::core::kCtePoolName,
                                              clio::cte::core::kCtePoolId,
                                              params);
  create_task.Wait();

  // Check if Create operation succeeded
  clio::run::u32 return_code = create_task->GetReturnCode();
  if (return_code != 0) {
    HLOG(kError, "CTE ClientInit: Failed to create CTE pool '{}' with return code: {}",
          clio::cte::core::kCtePoolName, return_code);
    is_initializing_ = false;
    return false;
  }

  // Update client pool_id_ with the actual pool ID from the task
  cte_client->pool_id_ = create_task->new_pool_id_;

  // Attach the shared-memory metadata cache (issue #783, wired up in #817).
  // Until this call existed the cache was only ever attached by its own unit
  // tests, so `Tag::GetBlob`'s zero-IPC fast path was dead in every real
  // process. Must run AFTER pool_id_ is set: directory lookup is keyed by pool,
  // and a shared slot would attach whichever CTE pool cached last.
  //
  // The DIRECTORY overload, not the CreateTask one: the task carries a non-zero
  // root offset only for the process that actually caused pool creation, and
  // under `clio compose` the pool already exists by the time any real client
  // runs. (Reading the task's params here would also drag the config parser
  // into the client library, which does not link it.)
  //
  // Failure is not an error -- a TCP client, a remote node, or a runtime built
  // without the cache all legitimately report false, and every caller of the
  // fast path falls back to RPC.
  if (!cte_client->AttachShmCache()) {
    HLOG(kDebug, "CTE ClientInit: SHM metadata cache unavailable; using RPC");
  }

  // Mark as successfully initialized
  is_initialized_ = true;
  is_initializing_ = false;

  return true;
}

std::vector<std::string> ContentTransferEngine::TagQuery(
    const std::string &tag_re,
    clio::run::u32 max_tags,
    const clio::run::PoolQuery &pool_query) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncTagQuery(tag_re, max_tags, pool_query);
  task.Wait();

  std::vector<std::string> results = task->results_;
  return results;
}

std::vector<std::pair<std::string, std::string>> ContentTransferEngine::BlobQuery(
    const std::string &tag_re,
    const std::string &blob_re,
    clio::run::u32 max_blobs,
    const clio::run::PoolQuery &pool_query) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncBlobQuery(tag_re, blob_re, max_blobs, pool_query);
  task.Wait();

  std::vector<std::pair<std::string, std::string>> results;
  for (size_t i = 0; i < task->tag_names_.size(); ++i) {
    results.emplace_back(task->tag_names_[i], task->blob_names_[i]);
  }

  return results;
}

} // namespace clio::cte::core